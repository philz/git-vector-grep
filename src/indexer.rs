//! Bring the cache up to date with the current repo state.
//!
//! The cache is an append-only union of source blob SHAs seen in any worktree.
//! No path mapping is stored: paths are reconstructed at search time from
//! `git ls-files -s`, and search only loads payloads referenced by the current
//! worktree.
//!
//! Algorithm:
//!   1. `git ls-files -s` -> (path, index_blob_sha) per tracked file.
//!   2. `git diff --name-only` -> paths whose working tree differs from the
//!      index; for these, recompute the blob SHA from disk bytes.
//!   3. target = set of unique blob_shas for tracked textual files.
//!   4. existing = `git ls-tree -r REF blobs/`.
//!   5. Embed target - existing, checkpointing each bounded group.
//!   6. Keep existing notes so other branches and interrupted runs stay cached.

use anyhow::Result;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use crate::chunker::chunk_bytes;
use crate::embed::Embed;
use crate::repo::{git_blob_sha1, list_tracked, looks_textual, modified_paths};
use crate::store::Store;

#[derive(Default, Debug)]
pub struct IndexStats {
    pub files_total: usize,
    pub blobs_unique: usize,
    pub blobs_already_cached: usize,
    pub blobs_embedded: usize,
    pub blobs_skipped: usize,
    /// Retained for API compatibility; append-only caches do not prune.
    pub blobs_pruned: usize,
    pub chunks_embedded: usize,
    pub elapsed_ms: u128,
}

impl std::fmt::Display for IndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "files: {} | blobs: {} unique, {} cached, {} embedded, {} skipped | chunks embedded: {} | {}ms",
            self.files_total,
            self.blobs_unique,
            self.blobs_already_cached,
            self.blobs_embedded,
            self.blobs_skipped,
            self.chunks_embedded,
            self.elapsed_ms,
        )
    }
}

#[derive(Clone)]
struct TrackedBlob {
    path: String,
    sha: String,
    raw_sha: bool,
}

fn list_tracked_with_blob_details(root: &Path) -> Result<Vec<TrackedBlob>> {
    let tracked = list_tracked(root)?;
    let tracked: Vec<_> = tracked
        .into_iter()
        .filter(|f| looks_textual(&f.path))
        .collect();
    let modified: HashSet<String> = modified_paths(root)?.into_iter().collect();

    Ok(tracked
        .into_par_iter()
        .filter_map(|f| {
            if modified.contains(&f.path) {
                let bytes = std::fs::read(root.join(&f.path)).ok()?;
                Some(TrackedBlob {
                    path: f.path,
                    sha: git_blob_sha1(&bytes),
                    raw_sha: true,
                })
            } else {
                Some(TrackedBlob {
                    path: f.path,
                    sha: f.index_blob_sha,
                    raw_sha: false,
                })
            }
        })
        .collect())
}

/// (path, blob_sha) for every tracked textual file in `root`.
///
/// Cheap: one `git ls-files -s` + one `git diff --name-only`; for files in
/// the diff we recompute the SHA by reading bytes off disk.
pub fn list_tracked_with_blobs(root: &Path) -> Result<Vec<(String, String)>> {
    Ok(list_tracked_with_blob_details(root)?
        .into_iter()
        .map(|f| (f.path, f.sha))
        .collect())
}

pub fn index_repo(
    root: &Path,
    cache: &mut Store,
    embedder: &dyn Embed,
    batch_size: usize,
    verbose: bool,
    quiet: bool,
) -> Result<IndexStats> {
    let t0 = Instant::now();
    let mut stats = IndexStats::default();

    // 1-2. Tracked paths + their authoritative blob SHAs.
    let tracked = list_tracked_with_blob_details(root)?;
    stats.files_total = tracked.len();

    // 3. Target set of unique blob SHAs.
    let target_blobs: HashSet<String> = tracked.iter().map(|f| f.sha.clone()).collect();
    stats.blobs_unique = target_blobs.len();

    // 4. What's already on the ref.
    let known_blobs = cache.known_blob_shas()?;
    stats.blobs_already_cached = target_blobs.intersection(&known_blobs).count();

    // 5. Blobs we need to embed = target - known.
    let mut to_embed: Vec<String> = target_blobs
        .difference(&known_blobs)
        .cloned()
        .collect();
    to_embed.sort();

    if to_embed.is_empty() {
        stats.elapsed_ms = t0.elapsed().as_millis();
        return Ok(stats);
    }

    // Pick one canonical path per blob_sha so we can read bytes off disk.
    // Prefer a path whose SHA was computed from raw working-tree bytes, since
    // that is the only case where raw hashing is valid in the presence of git
    // clean filters and line-ending normalization.
    let mut sha_to_path: HashMap<String, (String, bool)> = HashMap::new();
    for file in tracked {
        let entry = sha_to_path
            .entry(file.sha)
            .or_insert_with(|| (file.path.clone(), file.raw_sha));
        if file.raw_sha && !entry.1 {
            *entry = (file.path, true);
        }
    }

    // Read + chunk in parallel. We keep only line ranges + texts; the file
    // bytes are dropped immediately after chunking.
    struct Chunked {
        sha: String,
        ranges: Vec<(u32, u32)>,
        texts: Vec<String>,
    }
    let chunked: Vec<Chunked> = to_embed
        .par_iter()
        .filter_map(|sha| {
            let (path, verify_raw_sha) = sha_to_path.get(sha)?;
            let bytes = std::fs::read(root.join(path)).ok()?;
            // For modified paths, both the scheduled SHA and this check hash raw
            // bytes. Index SHAs may include clean filters, so do not compare
            // those against raw working-tree bytes.
            if *verify_raw_sha && git_blob_sha1(&bytes) != *sha {
                return None;
            }
            let chunks = chunk_bytes(&bytes);
            let ranges = chunks.iter().map(|c| (c.start_line, c.end_line)).collect();
            let texts = chunks.into_iter().map(|c| c.text).collect();
            Some(Chunked {
                sha: sha.clone(),
                ranges,
                texts,
            })
        })
        .collect();

    stats.blobs_skipped = to_embed.len() - chunked.len();

    // Embed in bounded groups to cap peak memory.
    let chunk_group: usize = std::env::var("GVG_CHUNK_GROUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let dim = embedder.dim();

    let total_chunks: usize = chunked.iter().map(|c| c.texts.len()).sum();
    // Announce the active backend + its resource caps + how to tweak them,
    // whenever we're about to actually embed (and not silenced).
    if total_chunks > 0 && !quiet {
        eprintln!("[embed] {} · batch {}", embedder.describe(), batch_size);
    }
    // Progress is shown unless silenced (--quiet, JSON output, or a
    // non-terminal stderr). Verbose keeps the old per-group line output.
    let show_progress = !quiet && !verbose && total_chunks > 0;
    if verbose && total_chunks > 0 {
        eprintln!(
            "[index] embedding {} chunks across {} unique blobs in groups of \u{2264}{}...",
            total_chunks,
            chunked.len(),
            chunk_group,
        );
    } else if show_progress {
        eprintln!(
            "indexing: embedding {} new chunks from {} blobs...",
            total_chunks,
            chunked.len(),
        );
    }

    // Push each blob into the current group; flush when full. Each blob's
    // resulting payload is committed to `cache` immediately to release the
    // f32s from RAM (`Store` buffers payloads as Vec<u8> which is the smallest
    // representation we'll need to keep around).
    let mut group: Vec<Chunked> = Vec::new();
    let mut group_chunks: usize = 0;
    let mut groups_done = 0usize;
    let mut chunks_done = 0usize;

    let flush = |group: &mut Vec<Chunked>,
                     group_chunks: &mut usize,
                     cache: &mut Store,
                     embedder: &dyn Embed|
     -> Result<usize> {
        if group.is_empty() {
            return Ok(0);
        }
        let mut texts: Vec<String> = Vec::with_capacity(*group_chunks);
        let mut layout: Vec<(String, Vec<(u32, u32)>, usize)> = Vec::with_capacity(group.len());
        for c in group.drain(..) {
            let n = c.texts.len();
            for t in c.texts {
                texts.push(t);
            }
            layout.push((c.sha, c.ranges, n));
        }
        let n = texts.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| texts[i].len());
        let sorted_texts: Vec<String> = order.iter().map(|&i| texts[i].clone()).collect();
        let sorted_vecs = if sorted_texts.is_empty() {
            Vec::new()
        } else {
            embedder.embed_flat(sorted_texts, batch_size)?
        };
        let mut group_vecs = vec![0f32; n * dim];
        for (sp, &op) in order.iter().enumerate() {
            group_vecs[op * dim..(op + 1) * dim]
                .copy_from_slice(&sorted_vecs[sp * dim..(sp + 1) * dim]);
        }
        let mut chunks_in_flush = 0usize;
        let mut pos = 0usize;
        for (sha, ranges, n_chunks) in layout {
            let slice = &group_vecs[pos * dim..(pos + n_chunks) * dim];
            cache.insert_chunks(&sha, &ranges, slice, dim);
            pos += n_chunks;
            chunks_in_flush += n_chunks;
        }
        // Persist every bounded group. A killed long-running index resumes from
        // its last completed group instead of recomputing everything.
        cache.commit()?;
        *group_chunks = 0;
        Ok(chunks_in_flush)
    };

    let report = |groups_done: usize, chunks_done: usize| {
        if verbose {
            eprintln!(
                "[index] ... group {} done ({} chunks total)",
                groups_done, chunks_done
            );
        } else if show_progress {
            let pct = (chunks_done as f64 / total_chunks as f64 * 100.0).min(100.0);
            eprint!(
                "\rindexing: {}/{} chunks ({:.0}%)   ",
                chunks_done, total_chunks, pct
            );
        }
    };

    for c in chunked {
        let n = c.texts.len();
        if group_chunks > 0 && group_chunks + n > chunk_group {
            let did = flush(&mut group, &mut group_chunks, cache, embedder)?;
            chunks_done += did;
            groups_done += 1;
            report(groups_done, chunks_done);
        }
        group_chunks += n;
        group.push(c);
        if group_chunks >= chunk_group {
            let did = flush(&mut group, &mut group_chunks, cache, embedder)?;
            chunks_done += did;
            groups_done += 1;
            report(groups_done, chunks_done);
        }
    }
    let did = flush(&mut group, &mut group_chunks, cache, embedder)?;
    chunks_done += did;
    if show_progress {
        eprintln!("\rindexing: {} chunks done.            ", chunks_done);
    }
    stats.chunks_embedded = chunks_done;
    stats.blobs_embedded = stats.blobs_unique - stats.blobs_already_cached - stats.blobs_skipped;

    stats.elapsed_ms = t0.elapsed().as_millis();
    Ok(stats)
}
