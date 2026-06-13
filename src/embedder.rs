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

fn new_session(choice: &ModelChoice) -> Result<TextEmbedding> {
    let cache_dir = default_cache_dir();
    std::fs::create_dir_all(&cache_dir).ok();
    let opts = InitOptions::new(choice.enum_id.clone())
        .with_show_download_progress(false)
        .with_cache_dir(cache_dir)
        // Pin intra-op to 1; we parallelize across sessions instead.
        .with_intra_threads(1);
    TextEmbedding::try_new(opts).context("failed to load embedding model")
}

/// A pool of `TextEmbedding` sessions, one per worker thread. Sessions are
/// guarded by `Mutex` because `embed()` requires `&mut self`; we never block
/// because each worker thread sticks to its own session via rayon's job
/// scheduling, but the Mutex is needed to satisfy `Send`+`Sync`.
pub struct Embedder {
    pub model_id: String,
    pub short_id: &'static str,
    pub dim: usize,
    pub workers: usize,
    sessions: Vec<Mutex<TextEmbedding>>,
    /// One session reserved for query embedding (cheap; serial).
    query_session: Mutex<TextEmbedding>,
}

impl Embedder {
    pub fn new(model_name: &str, workers: Option<usize>) -> Result<Self> {
        let choice = parse_model(model_name)?;
        // Default: assume ~700 MB per worker at the default batch size
        // (empirically measured for MiniLM-f32 at batch=8). Cap by core count
        // and by total RAM minus 1.5 GB of OS headroom.
        let workers = workers.unwrap_or_else(|| {
            let cpus = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            let ram_gb = sys_total_ram_gb().unwrap_or(8.0);
            let by_ram = ((ram_gb - 1.5) / 0.75).floor().max(1.0) as usize;
            cpus.min(by_ram).max(1)
        }).max(1);
        // Load the model once to warm caches, then spawn N sessions in parallel.
        let query_session = Mutex::new(new_session(&choice)?);
        let sessions: Vec<Mutex<TextEmbedding>> = (0..workers)
            .into_par_iter()
            .map(|_| new_session(&choice).map(Mutex::new))
            .collect::<Result<_>>()?;
        Ok(Embedder {
            model_id: choice.canonical_id.to_string(),
            short_id: choice.short_id,
            dim: choice.dim,
            workers,
            sessions,
            query_session,
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
                let mut session = self.sessions[wi].lock().unwrap();
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
        let mut s = self.query_session.lock().unwrap();
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

/// Rough total-RAM probe via /proc/meminfo. Returns gigabytes.
fn sys_total_ram_gb() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest
                .trim()
                .split_whitespace()
                .next()?
                .parse()
                .ok()?;
            return Some(kb as f64 / 1_048_576.0);
        }
    }
    None
}
