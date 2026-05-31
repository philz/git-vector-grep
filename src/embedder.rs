//! Embedding model wrapper around fastembed-rs.

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::path::PathBuf;

/// Identifier we record in the cache `meta` table so a model change wipes
/// stale vectors.
pub const DEFAULT_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
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

/// Minimal OpenAI-compatible embeddings client (works with any /v1/embeddings
/// endpoint, including https://llm.int.exe.xyz/v1).
///
/// We model it as a parallel Embedder type with the same shape as the local
/// one. We split inputs into HTTP-friendly batches and parallelize across
/// them with rayon.
pub struct RemoteEmbedder {
    pub model_id: String,
    pub dim: usize,
    base_url: String,
    api_key: Option<String>,
    requested_dim: Option<usize>,
}

#[derive(serde::Serialize)]
struct EmbReq<'a> {
    model: &'a str,
    input: Vec<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}
#[derive(serde::Deserialize)]
struct EmbItem { embedding: Vec<f32> }
#[derive(serde::Deserialize)]
struct EmbResp { data: Vec<EmbItem> }

impl RemoteEmbedder {
    /// Convenience: configure for the exe.dev LLM gateway.
    pub fn exe_gateway(model: &str, dim: usize) -> Self {
        Self {
            model_id: model.to_string(),
            dim,
            base_url: "https://llm.int.exe.xyz/v1".to_string(),
            api_key: None,
            requested_dim: Some(dim),
        }
    }

    pub fn embed_flat<S>(&self, texts: Vec<S>, batch_size: usize) -> Result<Vec<f32>>
    where
        S: AsRef<str> + Send + Sync,
    {
        let n = texts.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let strs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
        let chunks: Vec<&[&str]> = strs.chunks(batch_size).collect();
        let dim = self.dim;
        // HTTP latency, not CPU, bounds us. Spin up many concurrent requests
        // via a dedicated rayon pool sized far above num_cpus.
        let n_workers = std::env::var("GVG_REMOTE_CONCURRENCY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16usize);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_workers)
            .build()
            .map_err(|e| anyhow::anyhow!("rayon pool: {e}"))?;
        use rayon::prelude::*;
        let batch_vecs: Vec<Vec<f32>> = pool.install(|| {
            chunks
                .par_iter()
                .map(|batch| self.embed_one_batch(batch))
                .collect::<Result<_>>()
        })?;
        let mut flat = Vec::with_capacity(n * dim);
        for v in batch_vecs {
            flat.extend(v);
        }
        l2_normalize_rows(&mut flat, dim);
        Ok(flat)
    }

    fn embed_one_batch(&self, batch: &[&str]) -> Result<Vec<f32>> {
        let url = format!("{}/embeddings", self.base_url);
        let body = EmbReq {
            model: &self.model_id,
            input: batch.to_vec(),
            dimensions: self.requested_dim,
        };
        let body_v = serde_json::to_value(&body)?;
        let mut last_err: Option<String> = None;
        let mut resp: Option<EmbResp> = None;
        for attempt in 0..4 {
            let mut req = ureq::post(&url).set("content-type", "application/json");
            if let Some(k) = &self.api_key {
                req = req.set("authorization", &format!("Bearer {}", k));
            }
            match req.send_json(body_v.clone()) {
                Ok(r) => match r.into_json::<EmbResp>() {
                    Ok(parsed) => { resp = Some(parsed); break; }
                    Err(e) => last_err = Some(format!("decode: {e}")),
                },
                Err(e) => last_err = Some(format!("http: {e}")),
            }
            let backoff_ms = 200u64 * (1u64 << attempt);
            std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
        }
        let resp = resp.ok_or_else(|| anyhow::anyhow!(
            "embeddings request failed after retries: {}", last_err.unwrap_or_default()
        ))?;
        let mut out = Vec::with_capacity(batch.len() * self.dim);
        for item in resp.data {
            anyhow::ensure!(
                item.embedding.len() == self.dim,
                "remote returned dim {}, expected {}",
                item.embedding.len(),
                self.dim
            );
            out.extend(item.embedding);
        }
        Ok(out)
    }

    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_flat(vec![text.to_string()], 1)
    }
}

pub enum AnyEmbedder {
    Local(Embedder),
    Remote(RemoteEmbedder),
}

impl AnyEmbedder {
    pub fn model_id(&self) -> &str {
        match self {
            AnyEmbedder::Local(e) => &e.model_id,
            AnyEmbedder::Remote(e) => &e.model_id,
        }
    }
    pub fn dim(&self) -> usize {
        match self {
            AnyEmbedder::Local(e) => e.dim,
            AnyEmbedder::Remote(e) => e.dim,
        }
    }
    pub fn embed_flat<S>(&mut self, texts: Vec<S>, batch_size: usize) -> Result<Vec<f32>>
    where
        S: AsRef<str> + Send + Sync,
    {
        match self {
            AnyEmbedder::Local(e) => e.embed_flat(texts, batch_size),
            AnyEmbedder::Remote(e) => e.embed_flat(texts, batch_size),
        }
    }
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        match self {
            AnyEmbedder::Local(e) => e.embed_query(text),
            AnyEmbedder::Remote(e) => e.embed_query(text),
        }
    }
}
