//! In-memory cosine search: pull all chunk vectors out of SQLite into a
//! contiguous f32 matrix, then dot-product against the (already-normalized)
//! query vector. Uses rayon for the scan.

use anyhow::Result;
use rayon::prelude::*;

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
    /// meta[i] = (path, blob_sha, idx, start, end)
    pub meta: Vec<(String, String, u32, u32, u32)>,
}

impl Index {
    pub fn load(store: &Store) -> Result<Self> {
        let payloads = store.iter_all_payloads()?;
        let mut matrix: Vec<f32> = Vec::new();
        let mut meta = Vec::new();
        let mut dim = store.meta.dim;
        for (path, sha, bytes) in payloads {
            let (d, ranges, vecs) = unpack_payload(&bytes)?;
            dim = d;
            for (i, (s, e)) in ranges.iter().enumerate() {
                matrix.extend_from_slice(&vecs[i * d..(i + 1) * d]);
                meta.push((path.clone(), sha.clone(), i as u32, *s, *e));
            }
        }
        Ok(Self { dim, matrix, meta })
    }

    pub fn len(&self) -> usize {
        self.meta.len()
    }

    /// Top-k by cosine similarity (qvec assumed L2-normalized).
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

        // Score in parallel; bind path-prefix filter inside the closure.
        let scores: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                if let Some(p) = path_prefix {
                    if !self.meta[i].0.starts_with(p) {
                        return f32::NEG_INFINITY;
                    }
                }
                let row = &self.matrix[i * dim..(i + 1) * dim];
                dot(row, qvec)
            })
            .collect();

        // Top-k selection (simple partial sort).
        let k = k.min(n);
        let mut idx: Vec<usize> = (0..n).collect();
        idx.select_nth_unstable_by(k.saturating_sub(1).min(n - 1), |&a, &b| {
            scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut top = idx[..k].to_vec();
        top.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));

        top.into_iter()
            .filter_map(|i| {
                let s = scores[i];
                if !s.is_finite() {
                    return None;
                }
                let m = &self.meta[i];
                Some(Hit {
                    path: m.0.clone(),
                    start_line: m.3,
                    end_line: m.4,
                    score: s,
                    blob_sha: m.1.clone(),
                    chunk_idx: m.2,
                })
            })
            .collect()
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // Manual unroll helps autovectorization on stable Rust without portable-simd.
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
