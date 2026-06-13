//! In-memory cosine search.
//!
//! We don't store path -> blob_sha; that mapping comes from `git ls-files -s`
//! on every search. One winning chunk corresponds to one blob_sha; we then
//! emit one hit per path currently pointing at that blob.

use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

use crate::indexer::list_tracked_with_blobs;
use crate::store::{unpack_payload, Store};

#[derive(Debug, Clone)]
pub struct Hit {
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub score: f32,
    pub blob_sha: String,
    pub chunk_idx: u32,
}

pub struct Index {
    pub dim: usize,
    pub matrix: Vec<f32>, // (n_rows * dim) f32
    /// meta[i] = (blob_sha, chunk_idx, start_line, end_line)
    pub meta: Vec<(String, u32, u32, u32)>,
    /// blob_sha -> all currently-tracked paths backed by that blob.
    pub blob_to_paths: HashMap<String, Vec<String>>,
}

impl Index {
    pub fn load(repo: &Path, store: &Store) -> Result<Self> {
        let payloads = store.iter_all_payloads()?;
        let mut matrix: Vec<f32> = Vec::new();
        let mut meta = Vec::new();
        let mut dim = store.dim;
        for (sha, bytes) in payloads {
            let (d, ranges, vecs) = unpack_payload(&bytes)?;
            dim = d;
            for (i, (s, e)) in ranges.iter().enumerate() {
                matrix.extend_from_slice(&vecs[i * d..(i + 1) * d]);
                meta.push((sha.clone(), i as u32, *s, *e));
            }
        }
        let mut blob_to_paths: HashMap<String, Vec<String>> = HashMap::new();
        for (path, sha) in list_tracked_with_blobs(repo)? {
            blob_to_paths.entry(sha).or_default().push(path);
        }
        Ok(Self { dim, matrix, meta, blob_to_paths })
    }

    pub fn len(&self) -> usize {
        self.meta.len()
    }

    /// Top-k hits. If a winning chunk's blob maps to multiple paths, we emit
    /// one hit per path (all at the same score). `top_k` bounds the number
    /// of chunks scanned-and-selected, not the number of emitted rows.
    pub fn search(
        &self,
        qvec: &[f32],
        k: usize,
        path_prefix: Option<&str>,
    ) -> Vec<Hit> {
        debug_assert_eq!(qvec.len(), self.dim);
        if self.meta.is_empty() {
            return Vec::new();
        }

        let dim = self.dim;
        let n = self.meta.len();

        // Pre-compute per-row passability of the path filter: a row passes
        // iff at least one path bound to its blob_sha matches the prefix.
        let row_passes: Vec<bool> = (0..n)
            .into_par_iter()
            .map(|i| {
                let sha = &self.meta[i].0;
                let Some(paths) = self.blob_to_paths.get(sha) else {
                    return false;
                };
                if let Some(p) = path_prefix {
                    paths.iter().any(|x| x.starts_with(p))
                } else {
                    true
                }
            })
            .collect();

        let scores: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                if !row_passes[i] {
                    return f32::NEG_INFINITY;
                }
                let row = &self.matrix[i * dim..(i + 1) * dim];
                dot(row, qvec)
            })
            .collect();

        let k = k.min(n);
        if k == 0 {
            return Vec::new();
        }
        let mut idx: Vec<usize> = (0..n).collect();
        idx.select_nth_unstable_by(k.saturating_sub(1).min(n - 1), |&a, &b| {
            scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut top = idx[..k].to_vec();
        top.sort_by(|&a, &b| {
            scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut out = Vec::with_capacity(k);
        for i in top {
            let s = scores[i];
            if !s.is_finite() {
                continue;
            }
            let m = &self.meta[i];
            let paths = match self.blob_to_paths.get(&m.0) {
                Some(v) => v.clone(),
                None => continue,
            };
            for path in paths {
                if let Some(p) = path_prefix {
                    if !path.starts_with(p) {
                        continue;
                    }
                }
                out.push(Hit {
                    path,
                    start_line: m.2,
                    end_line: m.3,
                    score: s,
                    blob_sha: m.0.clone(),
                    chunk_idx: m.1,
                });
            }
        }
        out
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s0 = 0f32;
    let mut s1 = 0f32;
    let mut s2 = 0f32;
    let mut s3 = 0f32;
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let j = i * 4;
        unsafe {
            s0 += a.get_unchecked(j) * b.get_unchecked(j);
            s1 += a.get_unchecked(j + 1) * b.get_unchecked(j + 1);
            s2 += a.get_unchecked(j + 2) * b.get_unchecked(j + 2);
            s3 += a.get_unchecked(j + 3) * b.get_unchecked(j + 3);
        }
    }
    let mut s = s0 + s1 + s2 + s3;
    for i in (chunks * 4)..a.len() {
        s += a[i] * b[i];
    }
    s
}
