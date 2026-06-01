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
        // Needs embedding. We keep only the cheap line ranges here; the
        // chunk text bytes live in `pending_texts` and get consumed (drained)
        // during the embedding phase to free RAM.
        Embed(String, Vec<(u32, u32)>, i64, i64), // (blob_sha, ranges, mtime_ns, size)
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
    // sha -> chunk texts, populated as we plan. We DRAIN it during the
    // embedding loop and free memory as we go.
    let mut pending_texts: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
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
        let ranges: Vec<(u32, u32)> = chunks.iter().map(|c| (c.start_line, c.end_line)).collect();
        // Stash the texts in pending_texts; cheap line-ranges go into the
        // action.
        pending_texts.entry(ld.blob_sha.clone()).or_insert_with(|| {
            chunks.into_iter().map(|c| c.text).collect()
        });
        actions.push((
            ld.path.clone(),
            Action::Embed(ld.blob_sha, ranges, ld.mtime_ns, ld.size),
        ));
    }

    // 4. Run embeddings batched across `pending_texts`, in bounded groups.
    use std::collections::HashMap;

    // Plan the embedding work in BOUNDED GROUPS so peak memory stays small:
    // process at most ~CHUNK_GROUP chunks at a time, embed them, stuff the
    // resulting payloads into the cache, free the texts, repeat.
    //
    // This keeps fastembed/ORT from buffering 70k+ tokenized inputs at once
    // (which OOMs an 8 GB box).
    // Tighter group bound keeps peak memory low on tiny VMs.
    // Override with GVG_CHUNK_GROUP env var if you have RAM to spare.
    let chunk_group: usize = std::env::var("GVG_CHUNK_GROUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let dim = embedder.dim();

    let mut blob_order: Vec<String> = pending_texts.keys().cloned().collect();
    blob_order.sort();
    let total_chunks: usize = blob_order
        .iter()
        .map(|s| pending_texts.get(s).unwrap().len())
        .sum();
    if verbose && total_chunks > 0 {
        eprintln!(
            "[index] embedding {} chunks across {} unique blobs in groups of ≤{}...",
            total_chunks,
            blob_order.len(),
            chunk_group,
        );
    }

    // For each blob, remember where its vectors landed (relative to the
    // group's output buffer). We move blob_to_chunks's slices into the
    // cache as we go.
    let mut sha_to_vecs: HashMap<String, Vec<f32>> = HashMap::with_capacity(blob_order.len());

    let mut group: Vec<(String, Vec<String>)> = Vec::new(); // (sha, chunk_texts)
    let mut group_chunks: usize = 0;
    let mut groups_done = 0usize;
    let mut chunks_done = 0usize;

    let mut flush = |group: &mut Vec<(String, Vec<String>)>,
                     group_chunks: &mut usize,
                     sha_to_vecs: &mut HashMap<String, Vec<f32>>,
                     embedder: &mut AnyEmbedder|
     -> Result<()> {
        if group.is_empty() {
            return Ok(());
        }
        // Build flat text list and remember (sha, count) per entry.
        let mut texts: Vec<String> = Vec::with_capacity(*group_chunks);
        let mut counts: Vec<(String, usize)> = Vec::with_capacity(group.len());
        for (sha, mut cs) in group.drain(..) {
            let n = cs.len();
            texts.append(&mut cs);
            counts.push((sha, n));
        }
        // Length-bucket within the group only.
        let n = texts.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| texts[i].len());
        let sorted_texts: Vec<String> = order.iter().map(|&i| texts[i].clone()).collect();
        let sorted_vecs = embedder.embed_flat(sorted_texts, batch_size)?;
        // Scatter back to original (group-local) positions.
        let mut group_vecs = vec![0f32; n * dim];
        for (sp, &op) in order.iter().enumerate() {
            group_vecs[op * dim..(op + 1) * dim]
                .copy_from_slice(&sorted_vecs[sp * dim..(sp + 1) * dim]);
        }
        // Split per blob and store.
        let mut pos = 0usize;
        for (sha, n_chunks) in counts {
            let slice = group_vecs[pos * dim..(pos + n_chunks) * dim].to_vec();
            sha_to_vecs.insert(sha, slice);
            pos += n_chunks;
        }
        *group_chunks = 0;
        Ok(())
    };

    for sha in &blob_order {
        // Move texts out of pending_texts -- frees memory progressively.
        let texts: Vec<String> = pending_texts.remove(sha).unwrap_or_default();
        let n = texts.len();
        // If adding this blob would blow the group budget AND the group has
        // anything in it, flush first.
        if group_chunks > 0 && group_chunks + n > chunk_group {
            flush(&mut group, &mut group_chunks, &mut sha_to_vecs, embedder)?;
            groups_done += 1;
            chunks_done += group_chunks;
            if verbose {
                eprintln!(
                    "[index] ... group {} done ({} chunks total)",
                    groups_done, chunks_done
                );
            }
        }
        group_chunks += n;
        group.push((sha.clone(), texts));
        // If a single blob exceeds the budget, flush immediately.
        if group_chunks >= chunk_group {
            flush(&mut group, &mut group_chunks, &mut sha_to_vecs, embedder)?;
            groups_done += 1;
            chunks_done += group_chunks;
            if verbose {
                eprintln!(
                    "[index] ... group {} done ({} chunks total)",
                    groups_done, chunks_done
                );
            }
        }
    }
    flush(&mut group, &mut group_chunks, &mut sha_to_vecs, embedder)?;
    let _ = chunks_done;

    // 5. Apply: write embeddings and upsert file rows for all classified files.
    for (path, action) in actions {
        match action {
            Action::Unchanged | Action::Skip => {}
            Action::ReuseBlob(sha, mt, sz, n) => {
                cache.upsert_file(&path, &sha, mt, sz, n);
            }
            Action::Embed(sha, ranges, mt, sz) => {
                if !cache.blob_payloads.contains_key(&sha) {
                    if let Some(vec) = sha_to_vecs.remove(&sha) {
                        cache.insert_chunks(&sha, &ranges, &vec, dim);
                        stats.chunks_embedded += ranges.len();
                    }
                }
                cache.upsert_file(&path, &sha, mt, sz, ranges.len() as i64);
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

    stats.elapsed_ms = t0.elapsed().as_millis();
    Ok(stats)
}
