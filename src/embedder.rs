//! Embedding model wrapper around fastembed-rs.

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::path::PathBuf;

/// Identifier we record in the cache `meta` table so a model change wipes
/// stale vectors.
pub const DEFAULT_MODEL_ID: &str = "BAAI/bge-small-en-v1.5";
pub const DEFAULT_DIM: usize = 384;

pub struct Embedder {
    pub model_id: String,
    pub dim: usize,
    inner: TextEmbedding,
}

pub fn parse_model(name: &str) -> Result<(EmbeddingModel, &'static str, usize)> {
    Ok(match name {
        "jina-code" | "jinaai/jina-embeddings-v2-base-code" => (
            EmbeddingModel::JinaEmbeddingsV2BaseCode,
            "jinaai/jina-embeddings-v2-base-code",
            768,
        ),
        "jina-en" | "jinaai/jina-embeddings-v2-base-en" => (
            EmbeddingModel::JinaEmbeddingsV2BaseEN,
            "jinaai/jina-embeddings-v2-base-en",
            768,
        ),
        "bge-small" | "BAAI/bge-small-en-v1.5" => (
            EmbeddingModel::BGESmallENV15,
            "BAAI/bge-small-en-v1.5",
            384,
        ),
        "bge-small-q" => (
            EmbeddingModel::BGESmallENV15Q,
            "BAAI/bge-small-en-v1.5-quantized",
            384,
        ),
        "minilm-q" => (
            EmbeddingModel::AllMiniLML6V2Q,
            "sentence-transformers/all-MiniLM-L6-v2-quantized",
            384,
        ),
        "bge-base" | "BAAI/bge-base-en-v1.5" => (
            EmbeddingModel::BGEBaseENV15,
            "BAAI/bge-base-en-v1.5",
            768,
        ),
        "minilm" | "sentence-transformers/all-MiniLM-L6-v2" => (
            EmbeddingModel::AllMiniLML6V2,
            "sentence-transformers/all-MiniLM-L6-v2",
            384,
        ),
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

impl Embedder {
    pub fn new(model_name: &str, threads: Option<usize>) -> Result<Self> {
        let (model_enum, id, dim) = parse_model(model_name)?;
        let cache_dir = default_cache_dir();
        std::fs::create_dir_all(&cache_dir).ok();

        let mut opts = InitOptions::new(model_enum)
            .with_show_download_progress(true)
            .with_cache_dir(cache_dir);
        if let Some(t) = threads {
            opts = opts.with_intra_threads(t);
        }
        let inner = TextEmbedding::try_new(opts)
            .context("failed to load embedding model")?;
        Ok(Embedder {
            model_id: id.to_string(),
            dim,
            inner,
        })
    }

    /// Returns embeddings as a flat Vec<f32> of length `texts.len() * dim`.
    /// Output is L2-normalized (fastembed normalizes for these models).
    pub fn embed_flat<S>(&mut self, texts: Vec<S>, batch_size: usize) -> Result<Vec<f32>>
    where
        S: AsRef<str> + Send + Sync,
    {
        let n = texts.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let vecs = self.inner.embed(texts, Some(batch_size))?;
        let mut flat = Vec::with_capacity(n * self.dim);
        for v in &vecs {
            anyhow::ensure!(
                v.len() == self.dim,
                "embedding dim mismatch: got {}, want {}",
                v.len(),
                self.dim
            );
            flat.extend_from_slice(v);
        }
        // Belt-and-suspenders: normalize.
        l2_normalize_rows(&mut flat, self.dim);
        Ok(flat)
    }

    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        let v = self.embed_flat(vec![text.to_string()], 1)?;
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
