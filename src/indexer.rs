//! Bring the cache up to date with the current repo state.
//!
//! Algorithm (cheap-first):
//!   1. `git ls-files -s` -> (path, index_blob_sha) for every tracked file.
//!   2. `git diff --name-only` -> set of paths whose working tree differs from
//!      the index. For these, we recompute the blob SHA ourselves; for the
//!      rest, the index SHA is authoritative.
//!   3. For each tracked path:
//!        - If cached file row matches (path -> blob_sha): nothing to do.
//!        - Else if cache already has chunks for this blob_sha (rename,
//!          revert, or duplicate content): upsert file row, reuse chunks.
//!        - Else: chunk + embed.
//!   4. Delete file rows for paths that vanished; prune orphan chunks.
//!
//! Reading file bytes only happens for new-content paths (where we have to
//! embed anyway).

use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use crate::chunker::{chunk_bytes, Chunk};
use crate::embedder::AnyEmbedder;
use crate::repo::{git_blob_sha1, list_tracked, looks_textual, modified_paths};
use crate::store::Store;

#[derive(Default, Debug)]
pub struct IndexStats {
    pub files_total: usize,
    pub files_unchanged: usize,
    pub files_reused_blob: usize,
    pub files_reembedded: usize,
    pub files_skipped: usize,
    pub chunks_embedded: usize,
    pub chunks_reused: usize,
    pub files_gone: usize,
    pub elapsed_ms: u128,
}

impl std::fmt::Display for IndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "files: {} total, {} unchanged, {} reused-blob, {} re-embedded, {} skipped, {} gone | chunks: {} embedded, {} reused | {}ms",
            self.files_total,
            self.files_unchanged,
            self.files_reused_blob,
            self.files_reembedded,
            self.files_skipped,
            self.files_gone,
            self.chunks_embedded,
            self.chunks_reused,
            self.elapsed_ms,
        )
    }
}

pub fn index_repo(
    root: &Path,
    cache: &mut Store,
    embedder: &mut AnyEmbedder,
    batch_size: usize,
    verbose: bool,
) -> Result<IndexStats> {
    let t0 = Instant::now();
    let mut stats = IndexStats::default();

    // 1. tracked + filter to texty paths.
    let tracked = list_tracked(root)?;
    let tracked: Vec<_> = tracked.into_iter().filter(|f| looks_textual(&f.path)).collect();
    stats.files_total = tracked.len();
    let tracked_paths: HashSet<&str> = tracked.iter().map(|f| f.path.as_str()).collect();

    // 2. modified set.
    let modified: HashSet<String> = modified_paths(root)?.into_iter().collect();

    // 3. load cache state.
    let cache_files: HashMap<String, crate::store::FileRow> = cache.load_files().clone();
    let known_blobs = cache.known_blob_shas()?;

    // Classify each tracked file.
    enum Action {
        Unchanged,
        // (blob_sha, mtime_ns, size, n_chunks) -> upsert only
        ReuseBlob(String, i64, i64, i64),
        // (chunks, vecs_placeholder) -> needs embedding; gathered after batching
        Embed(String, Vec<Chunk>, i64, i64), // (blob_sha, chunks, mtime_ns, size)
        Skip,
    }

    // To compute blob shas for modified files in parallel.
    // For unmodified files, the index sha is authoritative; no file I/O.
    struct PlannedRead {
        path: String,
        blob_sha: Option<String>, // None means "compute by reading the file"
    }
    let plan: Vec<PlannedRead> = tracked
        .iter()
        .map(|f| PlannedRead {
            path: f.path.clone(),
            blob_sha: if modified.contains(&f.path) {
                None
            } else {
                Some(f.index_blob_sha.clone())
            },
        })
        .collect();

    // Pre-fetch stat & (optionally) bytes in parallel.
    struct Loaded {
        path: String,
        mtime_ns: i64,
        size: i64,
        blob_sha: String,
        bytes: Option<Vec<u8>>, // present iff we had to read the file
        skipped: bool,
    }
    let loaded: Vec<Loaded> = plan
        .into_par_iter()
        .map(|p| {
            let full = root.join(&p.path);
            let md = match std::fs::metadata(&full) {
                Ok(m) => m,
                Err(_) => {
                    return Loaded {
                        path: p.path,
                        mtime_ns: 0,
                        size: 0,
                        blob_sha: String::new(),
                        bytes: None,
                        skipped: true,
                    };
                }
            };
            let mtime_ns = md
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos() as i64)
                .unwrap_or(0);
            let size = md.len() as i64;
            match p.blob_sha {
                Some(sha) => Loaded {
                    path: p.path,
                    mtime_ns,
                    size,
                    blob_sha: sha,
                    bytes: None,
                    skipped: false,
                },
                None => {
                    let bytes = match std::fs::read(&full) {
                        Ok(b) => b,
                        Err(_) => {
                            return Loaded {
                                path: p.path,
                                mtime_ns,
                                size,
                                blob_sha: String::new(),
                                bytes: None,
                                skipped: true,
                            };
                        }
                    };
                    let sha = git_blob_sha1(&bytes);
                    Loaded {
                        path: p.path,
                        mtime_ns,
                        size,
                        blob_sha: sha,
                        bytes: Some(bytes),
                        skipped: false,
                    }
                }
            }
        })
        .collect();

    // Now classify and (lazily) read+chunk for Embed cases.
    let mut actions: Vec<(String, Action)> = Vec::with_capacity(loaded.len());
    // Track new blob_shas we'll embed so dup content in this batch is shared.
    let mut planning_embed: HashSet<String> = HashSet::new();

    for ld in loaded {
        if ld.skipped {
            stats.files_skipped += 1;
            actions.push((ld.path, Action::Skip));
            continue;
        }

        let cached = cache_files.get(&ld.path);
        if let Some(c) = cached {
            if c.blob_sha == ld.blob_sha {
                // path unchanged. Just refresh mtime/size if they drifted.
                if c.mtime_ns != ld.mtime_ns || c.size != ld.size {
                    cache.upsert_file(
                        &ld.path,
                        &ld.blob_sha,
                        ld.mtime_ns,
                        ld.size,
                        c.n_chunks,
                    );
                }
                stats.files_unchanged += 1;
                actions.push((ld.path, Action::Unchanged));
                continue;
            }
        }

        if known_blobs.contains(&ld.blob_sha) || planning_embed.contains(&ld.blob_sha) {
            // Content already embedded somewhere (rename / dup / revert).
            let n = cache.chunk_count_for(&ld.blob_sha)?;
            actions.push((
                ld.path.clone(),
                Action::ReuseBlob(ld.blob_sha.clone(), ld.mtime_ns, ld.size, n),
            ));
            stats.files_reused_blob += 1;
            stats.chunks_reused += n as usize;
            continue;
        }

        // Need to embed. We have to read the bytes if we don't already.
        let bytes = match ld.bytes {
            Some(b) => b,
            None => match std::fs::read(root.join(&ld.path)) {
                Ok(b) => b,
                Err(_) => {
                    stats.files_skipped += 1;
                    actions.push((ld.path, Action::Skip));
                    continue;
                }
            },
        };
        let chunks = chunk_bytes(&bytes);
        if chunks.is_empty() {
            stats.files_skipped += 1;
            actions.push((ld.path, Action::Skip));
            continue;
        }
        planning_embed.insert(ld.blob_sha.clone());
        actions.push((
            ld.path.clone(),
            Action::Embed(ld.blob_sha, chunks, ld.mtime_ns, ld.size),
        ));
    }

    // 4. Run embeddings batched across all Embed entries, *deduplicated by
    // blob_sha* so identical files only embed once even within one run.
    use std::collections::HashMap;
    let mut blob_to_chunks: HashMap<String, &Vec<Chunk>> = HashMap::new();
    for (_p, a) in &actions {
        if let Action::Embed(sha, chunks, _, _) = a {
            blob_to_chunks.entry(sha.clone()).or_insert(chunks);
        }
    }

    // Flatten into a single text vector for batched embedding.
    let mut flat_texts: Vec<String> = Vec::new();
    let mut owner: Vec<(String, u32, u32)> = Vec::new(); // (blob_sha, start, end)
    // Preserve a stable order so we can slice back.
    let mut blob_order: Vec<String> = blob_to_chunks.keys().cloned().collect();
    blob_order.sort();
    let mut blob_offsets: HashMap<String, (usize, usize)> = HashMap::new(); // sha -> (start, n)
    for sha in &blob_order {
        let chunks = blob_to_chunks.get(sha).unwrap();
        let start = flat_texts.len();
        for c in chunks.iter() {
            flat_texts.push(c.text.clone());
            owner.push((sha.clone(), c.start_line, c.end_line));
        }
        blob_offsets.insert(sha.clone(), (start, chunks.len()));
    }

    let mut flat_vecs: Vec<f32> = Vec::new();
    if !flat_texts.is_empty() {
        if verbose {
            eprintln!(
                "[index] embedding {} chunks across {} unique blobs...",
                flat_texts.len(),
                blob_order.len()
            );
        }
        // Length-bucketed batching: sort by byte length so each ONNX batch has
        // similar sequence lengths, minimizing padding. Then scatter vectors
        // back to the original positions.
        let n_flat = flat_texts.len();
        let mut order: Vec<usize> = (0..n_flat).collect();
        order.sort_by_key(|&i| flat_texts[i].len());
        let sorted_texts: Vec<String> = order.iter().map(|&i| flat_texts[i].clone()).collect();
        let sorted_vecs = embedder.embed_flat(sorted_texts, batch_size)?;
        let dim = embedder.dim();
        flat_vecs = vec![0f32; n_flat * dim];
        for (sorted_pos, &orig_pos) in order.iter().enumerate() {
            flat_vecs[orig_pos * dim..(orig_pos + 1) * dim]
                .copy_from_slice(&sorted_vecs[sorted_pos * dim..(sorted_pos + 1) * dim]);
        }
    }

    // 5. Apply: write embeddings and upsert file rows for all classified files.
    for (path, action) in actions {
        match action {
            Action::Unchanged | Action::Skip => {}
            Action::ReuseBlob(sha, mt, sz, n) => {
                cache.upsert_file(&path, &sha, mt, sz, n);
            }
            Action::Embed(sha, chunks, mt, sz) => {
                let (start, n) = blob_offsets[&sha];
                // The embedding payload only needs to be written once per blob.
                // Subsequent paths sharing this sha just get a files row.
                if !cache.blob_payloads.contains_key(&sha) {
                    let lines: Vec<(u32, u32)> = chunks
                        .iter()
                        .map(|c| (c.start_line, c.end_line))
                        .collect();
                    let dim = embedder.dim();
                    let vec_slice = &flat_vecs[start * dim..(start + n) * dim];
                    cache.insert_chunks(&sha, &lines, vec_slice, dim);
                    stats.chunks_embedded += n;
                }
                cache.upsert_file(&path, &sha, mt, sz, chunks.len() as i64);
                stats.files_reembedded += 1;
            }
        }
    }

    // 6. Drop file rows for vanished paths.
    let gone: Vec<String> = cache_files
        .keys()
        .filter(|p| !tracked_paths.contains(p.as_str()))
        .cloned()
        .collect();
    stats.files_gone = gone.len();
    if !gone.is_empty() {
        cache.delete_files(&gone);
    }
    // Orphan blob pruning happens inside commit() because that's where the
    // referenced-set is materialized.

    let _ = owner; // silence unused
    stats.elapsed_ms = t0.elapsed().as_millis();
    Ok(stats)
}
