//! Embedding model wrapper around fastembed-rs (local ONNX, CPU).
//!
//! We construct one `TextEmbedding` per worker thread with
//! `intra_threads=1`, then split embedding work across workers in parallel.
//! For tiny encoders this beats one session with `intra_threads=N` because
//! GEMM parallelism scales sub-linearly past ~2 threads on small matrices.

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::embed::Embed;

impl Embed for Embedder {
    fn embed_flat(&self, texts: Vec<String>, batch_size: usize) -> Result<Vec<f32>> {
        Embedder::embed_flat(self, texts, batch_size)
    }
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Embedder::embed_query(self, text)
    }
    fn default_batch_size(&self) -> usize {
        1
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn short_id(&self) -> &str {
        self.short_id
    }
    fn model_id(&self) -> &str {
        &self.model_id
    }
    fn describe(&self) -> String {
        let n = self.sessions.len();
        format!(
            "cpu/onnx · {} · {} planned session(s) (1 eager, {} lazy) × {} intra-threads · memory scales with sessions × batch size · \
             tweak: --workers N to cap parallelism, or --backend mlx for the GPU",
            self.short_id,
            n,
            n.saturating_sub(1),
            self.intra_threads
        )
    }
}

/// Canonical model id recorded in the cache `meta.json`.
#[derive(Clone)]
pub struct ModelChoice {
    pub enum_id: EmbeddingModel,
    pub canonical_id: &'static str,
    /// Short, stable slug used as the notes-ref segment
    /// (`refs/notes/vector-grep/<short_id>`). Must match `[a-z0-9-]+`.
    pub short_id: &'static str,
    pub dim: usize,
}

pub fn parse_model(name: &str) -> Result<ModelChoice> {
    Ok(match name {
        "bge-small-q" | "BAAI/bge-small-en-v1.5-quantized" => ModelChoice {
            enum_id: EmbeddingModel::BGESmallENV15Q,
            canonical_id: "BAAI/bge-small-en-v1.5-quantized",
            short_id: "bge-small-q",
            dim: 384,
        },
        "bge-small" | "BAAI/bge-small-en-v1.5" => ModelChoice {
            enum_id: EmbeddingModel::BGESmallENV15,
            canonical_id: "BAAI/bge-small-en-v1.5",
            short_id: "bge-small",
            dim: 384,
        },
        "bge-base" | "BAAI/bge-base-en-v1.5" => ModelChoice {
            enum_id: EmbeddingModel::BGEBaseENV15,
            canonical_id: "BAAI/bge-base-en-v1.5",
            short_id: "bge-base",
            dim: 768,
        },
        "minilm" | "sentence-transformers/all-MiniLM-L6-v2" => ModelChoice {
            enum_id: EmbeddingModel::AllMiniLML6V2,
            canonical_id: "sentence-transformers/all-MiniLM-L6-v2",
            short_id: "minilm",
            dim: 384,
        },
        "jina-code" | "jinaai/jina-embeddings-v2-base-code" => ModelChoice {
            enum_id: EmbeddingModel::JinaEmbeddingsV2BaseCode,
            canonical_id: "jinaai/jina-embeddings-v2-base-code",
            short_id: "jina-code",
            dim: 768,
        },
        other => anyhow::bail!("unknown model: {other}"),
    })
}

pub fn default_cache_dir() -> PathBuf {
    if let Some(d) = dirs::cache_dir() {
        d.join("git-vector-grep").join("models")
    } else {
        PathBuf::from(".git-vector-grep-models")
    }
}

const MAX_AUTO_SMALL_MODEL_WORKERS: usize = 8;
const AUTO_WORKER_GIB: f64 = 0.75;
const MIN_AUTO_HEADROOM_GIB: f64 = 1.5;

fn auto_worker_count(model: &str, cpus: usize, total_ram_gib: Option<f64>) -> usize {
    let cpus = cpus.max(1);
    // Only the small 384-D encoders use automatic multi-session inference.
    // Larger models keep the previous single-session default; users with ample
    // RAM can still opt in explicitly with --workers.
    let max_workers = match model {
        "minilm" | "bge-small" | "bge-small-q" => MAX_AUTO_SMALL_MODEL_WORKERS,
        _ => 1,
    };
    let Some(ram_gib) = total_ram_gib else {
        return 1;
    };
    // Reserve at least 1.5 GiB and, on larger machines, 25% of total RAM for
    // the OS, git, vector payloads, and concurrent processes.
    let headroom = MIN_AUTO_HEADROOM_GIB.max(ram_gib * 0.25);
    let by_ram = ((ram_gib - headroom) / AUTO_WORKER_GIB).floor().max(1.0) as usize;
    cpus.min(max_workers).min(by_ram).max(1)
}

fn parse_cgroup_memory_limit(raw: &str) -> Option<u64> {
    let raw = raw.trim();
    if raw.is_empty() || raw == "max" {
        return None;
    }
    raw.parse::<u64>().ok().filter(|&bytes| bytes > 0)
}

fn select_memory_limit(host_bytes: Option<u64>, cgroup_bytes: Option<u64>) -> Option<u64> {
    match (host_bytes, cgroup_bytes) {
        (Some(host), Some(cgroup)) => Some(host.min(cgroup)),
        (Some(host), None) => Some(host),
        (None, Some(cgroup)) => Some(cgroup),
        (None, None) => None,
    }
}

fn host_total_ram_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb = s
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?;
    kb.checked_mul(1024)
}

fn cgroup_limit_up_tree(root: &std::path::Path, relative: &str, file: &str) -> Option<u64> {
    let mut dir = root.join(relative.trim_start_matches('/'));
    let mut limit: Option<u64> = None;
    loop {
        if let Ok(raw) = std::fs::read_to_string(dir.join(file)) {
            if let Some(bytes) = parse_cgroup_memory_limit(&raw) {
                limit = Some(limit.map_or(bytes, |current| current.min(bytes)));
            }
        }
        if dir == root || !dir.pop() || !dir.starts_with(root) {
            break;
        }
    }
    limit
}

fn cgroup_memory_limit_bytes() -> Option<u64> {
    let memberships = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mut limit: Option<u64> = None;
    for line in memberships.lines() {
        let mut fields = line.splitn(3, ':');
        let hierarchy = fields.next().unwrap_or_default();
        let controllers = fields.next().unwrap_or_default();
        let relative = fields.next().unwrap_or_default();
        let found = if hierarchy == "0" && controllers.is_empty() {
            // Unified cgroup v2 hierarchy.
            cgroup_limit_up_tree(
                std::path::Path::new("/sys/fs/cgroup"),
                relative,
                "memory.max",
            )
        } else if controllers.split(',').any(|c| c == "memory") {
            // Common cgroup v1 mount layout. Also try a combined-controller
            // layout rooted directly at /sys/fs/cgroup.
            cgroup_limit_up_tree(
                std::path::Path::new("/sys/fs/cgroup/memory"),
                relative,
                "memory.limit_in_bytes",
            )
            .or_else(|| {
                cgroup_limit_up_tree(
                    std::path::Path::new("/sys/fs/cgroup"),
                    relative,
                    "memory.limit_in_bytes",
                )
            })
        } else {
            None
        };
        if let Some(bytes) = found {
            limit = Some(limit.map_or(bytes, |current| current.min(bytes)));
        }
    }
    limit
}

fn sys_total_ram_gib() -> Option<f64> {
    select_memory_limit(host_total_ram_bytes(), cgroup_memory_limit_bytes())
        .map(|bytes| bytes as f64 / 1_073_741_824.0)
}

fn lazy_worker_slots<T>(workers: usize, first: T) -> Vec<Mutex<Option<T>>> {
    let mut first = Some(first);
    (0..workers.max(1))
        .map(|_| Mutex::new(first.take()))
        .collect()
}

fn new_session(choice: &ModelChoice, intra_threads: usize) -> Result<TextEmbedding> {
    let cache_dir = default_cache_dir();
    std::fs::create_dir_all(&cache_dir).ok();
    let opts = InitOptions::new(choice.enum_id.clone())
        .with_show_download_progress(false)
        .with_cache_dir(cache_dir)
        .with_intra_threads(intra_threads.max(1));
    TextEmbedding::try_new(opts).context("failed to load embedding model")
}

/// A pool of lazily initialized `TextEmbedding` sessions, one per worker.
/// The first session is ready at construction time; remaining sessions are
/// created on first indexing work. Each slot is guarded by `Mutex` because
/// `embed()` requires `&mut self` and to make concurrent initialization safe.
pub struct Embedder {
    pub model_id: String,
    pub short_id: &'static str,
    pub dim: usize,
    pub workers: usize,
    pub intra_threads: usize,
    sessions: Vec<Mutex<Option<TextEmbedding>>>,
    choice: ModelChoice,
}

impl Embedder {
    pub fn new(model_name: &str, workers: Option<usize>) -> Result<Self> {
        let choice = parse_model(model_name)?;
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        // Batch-1 inference avoids padding unrelated code chunks to the longest
        // sequence in a batch. Multiple single-thread sessions then keep all
        // cores busy without the large activation arenas caused by batching.
        // Auto-size small models up to eight sessions with substantial RAM
        // headroom. Larger models retain a single-session default; explicit
        // --workers N remains available for manual tuning.
        let workers = workers
            .unwrap_or_else(|| auto_worker_count(choice.short_id, cpus, sys_total_ram_gib()))
            .max(1);
        let intra = (cpus / workers).max(1);
        // Initialize one session synchronously. Besides serving cached searches,
        // this completes any Hugging Face download before lazy worker sessions
        // are constructed in parallel, avoiding fresh-cache lock races.
        let first_session = new_session(&choice, intra)?;
        let sessions = lazy_worker_slots(workers, first_session);
        Ok(Embedder {
            model_id: choice.canonical_id.to_string(),
            short_id: choice.short_id,
            dim: choice.dim,
            workers,
            intra_threads: intra,
            sessions,
            choice,
        })
    }

    /// Embed many texts in parallel across worker sessions. Output is a flat
    /// L2-normalized f32 buffer of length `texts.len() * dim`.
    ///
    /// Each worker session processes one shard sequentially with its internal
    /// batch_size; we split the input across `self.workers` shards.
    pub fn embed_flat(&self, texts: Vec<String>, batch_size: usize) -> Result<Vec<f32>> {
        let n = texts.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let dim = self.dim;
        let n_workers = self.workers.min(n).max(1);
        // Even shards. Round-robin keeps similar-length runs together when
        // the caller pre-sorts by length.
        let mut shards: Vec<Vec<(usize, String)>> = (0..n_workers).map(|_| Vec::new()).collect();
        for (i, t) in texts.into_iter().enumerate() {
            shards[i % n_workers].push((i, t));
        }
        // Run shards in parallel; each shard uses its own session.
        let shard_results: Vec<Result<Vec<(usize, Vec<f32>)>>> = shards
            .into_par_iter()
            .enumerate()
            .map(|(wi, shard)| {
                let mut slot = self.sessions[wi].lock().unwrap();
                if slot.is_none() {
                    *slot = Some(new_session(&self.choice, self.intra_threads)?);
                }
                let session = slot.as_mut().expect("worker session initialized");
                let mut indices = Vec::with_capacity(shard.len());
                let mut texts_only = Vec::with_capacity(shard.len());
                for (i, t) in shard {
                    indices.push(i);
                    texts_only.push(t);
                }
                let vecs = session.embed(texts_only, Some(batch_size))?;
                let mut out = Vec::with_capacity(indices.len());
                for (idx, v) in indices.into_iter().zip(vecs.into_iter()) {
                    anyhow::ensure!(
                        v.len() == dim,
                        "embedding dim mismatch: got {} want {}",
                        v.len(),
                        dim
                    );
                    out.push((idx, v));
                }
                Ok(out)
            })
            .collect();
        // We need `&mut shards` from inside the par_iter; the loop above only
        // worked because `take` swaps with default; let's check correctness by
        // collecting errors first.
        let mut flat = vec![0f32; n * dim];
        for sr in shard_results {
            for (idx, v) in sr? {
                flat[idx * dim..(idx + 1) * dim].copy_from_slice(&v);
            }
        }
        l2_normalize_rows(&mut flat, dim);
        Ok(flat)
    }

    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        // Indexing and query embedding are sequential in the CLI, so sharing
        // the first worker avoids loading a redundant ninth model session.
        let mut slot = self.sessions[0].lock().unwrap();
        let s = slot.as_mut().expect("first worker session initialized");
        let vecs = s.embed(vec![text.to_string()], Some(1))?;
        let mut v = vecs.into_iter().next().unwrap_or_default();
        l2_normalize_rows(&mut v, self.dim);
        Ok(v)
    }
}

pub fn l2_normalize_rows(buf: &mut [f32], dim: usize) {
    let n = buf.len() / dim;
    for i in 0..n {
        let row = &mut buf[i * dim..(i + 1) * dim];
        let mut s = 0f32;
        for &x in row.iter() {
            s += x * x;
        }
        if s > 0.0 {
            let inv = 1.0 / s.sqrt();
            for x in row.iter_mut() {
                *x *= inv;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        auto_worker_count, cgroup_limit_up_tree, lazy_worker_slots, parse_cgroup_memory_limit,
        select_memory_limit,
    };

    #[test]
    fn cgroup_limit_parser_handles_v2_max_and_numeric_limits() {
        assert_eq!(parse_cgroup_memory_limit("max\n"), None);
        assert_eq!(
            parse_cgroup_memory_limit("4294967296\n"),
            Some(4_294_967_296)
        );
        assert_eq!(parse_cgroup_memory_limit("not-a-number"), None);
    }

    #[test]
    fn cgroup_limit_selection_walks_parent_limits() {
        let root = tempfile::tempdir().unwrap();
        let child = root.path().join("slice/service");
        std::fs::create_dir_all(&child).unwrap();
        std::fs::write(root.path().join("memory.max"), "17179869184\n").unwrap();
        std::fs::write(root.path().join("slice/memory.max"), "4294967296\n").unwrap();
        std::fs::write(child.join("memory.max"), "max\n").unwrap();
        assert_eq!(
            cgroup_limit_up_tree(root.path(), "slice/service", "memory.max"),
            Some(4_294_967_296)
        );
    }

    #[test]
    fn effective_memory_uses_the_smallest_available_limit() {
        assert_eq!(select_memory_limit(Some(16), Some(4)), Some(4));
        assert_eq!(select_memory_limit(Some(4), Some(16)), Some(4));
        assert_eq!(select_memory_limit(Some(16), None), Some(16));
        assert_eq!(select_memory_limit(None, Some(4)), Some(4));
        assert_eq!(select_memory_limit(None, None), None);
    }

    #[test]
    fn cgroup_limit_reduces_auto_worker_count() {
        let bytes = select_memory_limit(Some(16 * 1_073_741_824), Some(2 * 1_073_741_824)).unwrap();
        let gib = bytes as f64 / 1_073_741_824.0;
        assert_eq!(auto_worker_count("minilm", 8, Some(gib)), 1);
    }

    #[test]
    fn worker_slots_only_initialize_the_first_session_eagerly() {
        let slots = lazy_worker_slots(4, "ready");
        let loaded = slots
            .iter()
            .filter(|slot| slot.lock().unwrap().is_some())
            .count();
        assert_eq!(slots.len(), 4);
        assert_eq!(loaded, 1);
    }

    #[test]
    fn auto_workers_use_all_eight_cores_for_small_models_when_memory_allows() {
        assert_eq!(auto_worker_count("minilm", 8, Some(16.0)), 8);
        assert_eq!(auto_worker_count("bge-small", 8, Some(16.0)), 8);
        assert_eq!(auto_worker_count("bge-small-q", 8, Some(16.0)), 8);
    }

    #[test]
    fn auto_workers_are_capped_for_large_cpu_counts() {
        assert_eq!(auto_worker_count("minilm", 64, Some(64.0)), 8);
    }

    #[test]
    fn larger_models_keep_the_safe_single_session_default() {
        assert_eq!(auto_worker_count("bge-base", 8, Some(16.0)), 1);
        assert_eq!(auto_worker_count("jina-code", 8, Some(16.0)), 1);
    }

    #[test]
    fn auto_workers_leave_memory_headroom() {
        assert_eq!(auto_worker_count("minilm", 8, Some(4.0)), 3);
        assert_eq!(auto_worker_count("minilm", 8, Some(2.0)), 1);
    }

    #[test]
    fn auto_workers_fall_back_to_one_without_memory_info() {
        assert_eq!(auto_worker_count("minilm", 8, None), 1);
        assert_eq!(auto_worker_count("minilm", 0, Some(16.0)), 1);
    }
}
