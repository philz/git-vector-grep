//! Bring the cache up to date with the current repo state.
//!
//! The cache is purely a function of the multiset of source blob SHAs that
//! make up the repo's tracked text files. No path mapping is stored: paths
//! are reconstructed at search time from `git ls-files -s`.
//!
//! Algorithm:
//!   1. `git ls-files -s` -> (path, index_blob_sha) per tracked file.
//!   2. `git diff --name-only` -> paths whose working tree differs from the
//!      index; for these, recompute the blob SHA from disk bytes.
//!   3. target = set of unique blob_shas for tracked textual files.
//!   4. existing = `git ls-tree -r REF blobs/`.
//!   5. Embed target - existing.
//!   6. Commit a tree containing exactly `target`'s blobs (orphans pruned).

use anyhow::Result;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use crate::chunker::chunk_bytes;
use crate::embedder::Embedder;
use crate::repo::{git_blob_sha1, list_tracked, looks_textual, modified_paths};
use crate::store::Store;

#[derive(Default, Debug)]
pub struct IndexStats {
    pub files_total: usize,
    pub blobs_unique: usize,
    pub blobs_already_cached: usize,
    pub blobs_embedded: usize,
    pub blobs_skipped: usize,
    pub blobs_pruned: usize,
    pub chunks_embedded: usize,
    pub elapsed_ms: u128,
}

impl std::fmt::Display for IndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "files: {} | blobs: {} unique, {} cached, {} embedded, {} skipped, {} pruned | chunks embedded: {} | {}ms",
            self.files_total,
            self.blobs_unique,
            self.blobs_already_cached,
            self.blobs_embedded,
            self.blobs_skipped,
            self.blobs_pruned,
            self.chunks_embedded,
            self.elapsed_ms,
        )
    }
}

/// (path, blob_sha) for every tracked textual file in `root`.
///
/// Cheap: one `git ls-files -s` + one `git diff --name-only`; for files in
/// the diff we recompute the SHA by reading bytes off disk.
pub fn list_tracked_with_blobs(root: &Path) -> Result<Vec<(String, String)>> {
    let tracked = list_tracked(root)?;
    let tracked: Vec<_> = tracked
        .into_iter()
        .filter(|f| looks_textual(&f.path))
        .collect();
    let modified: HashSet<String> = modified_paths(root)?.into_iter().collect();

    let pairs: Vec<(String, String)> = tracked
        .into_par_iter()
        .filter_map(|f| {
            if modified.contains(&f.path) {
                let bytes = std::fs::read(root.join(&f.path)).ok()?;
                let sha = git_blob_sha1(&bytes);
                Some((f.path, sha))
            } else {
                Some((f.path, f.index_blob_sha))
            }
        })
        .collect();
    Ok(pairs)
}

pub fn index_repo(
    root: &Path,
    cache: &mut Store,
    embedder: &Embedder,
    batch_size: usize,
    verbose: bool,
) -> Result<IndexStats> {
    let t0 = Instant::now();
    let mut stats = IndexStats::default();

    // 1-2. Tracked paths + their authoritative blob SHAs.
    let pairs = list_tracked_with_blobs(root)?;
    stats.files_total = pairs.len();

    // 3. Target set of unique blob SHAs.
    let target_blobs: HashSet<String> = pairs.iter().map(|(_, s)| s.clone()).collect();
    stats.blobs_unique = target_blobs.len();

    // 4. What's already on the ref.
    let known_blobs = cache.known_blob_shas()?;
    stats.blobs_already_cached = target_blobs.intersection(&known_blobs).count();
    stats.blobs_pruned = known_blobs.difference(&target_blobs).count();

    // 5. Blobs we need to embed = target - known.
    let mut to_embed: Vec<String> = target_blobs
        .difference(&known_blobs)
        .cloned()
        .collect();
    to_embed.sort();

    if to_embed.is_empty() {
        // Still need to apply pruning if the on-disk set differs from target.
        if known_blobs != target_blobs {
            cache.set_referenced(target_blobs.clone());
        }
        stats.elapsed_ms = t0.elapsed().as_millis();
        return Ok(stats);
    }

    // Pick one canonical path per blob_sha so we can read bytes off disk.
    let mut sha_to_path: HashMap<String, String> = HashMap::new();
    for (path, sha) in &pairs {
        sha_to_path.entry(sha.clone()).or_insert_with(|| path.clone());
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
            let path = sha_to_path.get(sha)?;
            let bytes = std::fs::read(root.join(path)).ok()?;
            // Defensive: working tree may have drifted under us mid-run.
            // We already trust the SHA computed earlier from the same bytes
            // (or from the index); recomputing here would be a race.
            let chunks = chunk_bytes(&bytes);
            if chunks.is_empty() {
                return None;
            }
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
    let dim = embedder.dim;

    let total_chunks: usize = chunked.iter().map(|c| c.texts.len()).sum();
    if verbose && total_chunks > 0 {
        eprintln!(
            "[index] embedding {} chunks across {} unique blobs in groups of \u{2264}{}...",
            total_chunks,
            chunked.len(),
            chunk_group,
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

    let mut flush = |group: &mut Vec<Chunked>,
                     group_chunks: &mut usize,
                     cache: &mut Store,
                     embedder: &Embedder|
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
        let sorted_vecs = embedder.embed_flat(sorted_texts, batch_size)?;
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
        *group_chunks = 0;
        Ok(chunks_in_flush)
    };

    for c in chunked {
        let n = c.texts.len();
        if group_chunks > 0 && group_chunks + n > chunk_group {
            let did = flush(&mut group, &mut group_chunks, cache, embedder)?;
            chunks_done += did;
            groups_done += 1;
            if verbose {
                eprintln!(
                    "[index] ... group {} done ({} chunks total)",
                    groups_done, chunks_done
                );
            }
        }
        group_chunks += n;
        group.push(c);
        if group_chunks >= chunk_group {
            let did = flush(&mut group, &mut group_chunks, cache, embedder)?;
            chunks_done += did;
            groups_done += 1;
            if verbose {
                eprintln!(
                    "[index] ... group {} done ({} chunks total)",
                    groups_done, chunks_done
                );
            }
        }
    }
    let did = flush(&mut group, &mut group_chunks, cache, embedder)?;
    chunks_done += did;
    stats.chunks_embedded = chunks_done;
    stats.blobs_embedded = stats.blobs_unique - stats.blobs_already_cached - stats.blobs_skipped;

    // 6. Tell the store exactly which blobs should survive the next commit.
    cache.set_referenced(target_blobs);

    stats.elapsed_ms = t0.elapsed().as_millis();
    Ok(stats)
}
