//! Backend for Apple's `afm embed` server — Apple NaturalLanguage contextual
//! embeddings (ANE-native), served over an OpenAI-compatible HTTP endpoint.
//!
//!   afm embed --port 9998        # apple-nl-contextual-en, 512-dim
//!
//! This is the "real" ANE path: Apple's own optimized model, not the
//! partition-and-fall-back-to-CPU behavior of the ONNX CoreML EP. We POST
//! batches to `/v1/embeddings`; `--sessions` controls how many requests are
//! in flight at once (the server can pipeline across cores/ANE).

use anyhow::{anyhow, Context, Result};
use git_vector_grep::embedder::l2_normalize_rows;
use rayon::prelude::*;
use serde_json::json;

use super::Backend;

pub struct AfmBackend {
    url: String,
    model: String,
    dim: usize,
    concurrency: usize,
    label: String,
    agent: ureq::Agent,
}

fn embed_request(
    agent: &ureq::Agent,
    url: &str,
    model: &str,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    let body = json!({ "input": texts, "model": model });
    let resp: serde_json::Value = agent
        .post(url)
        .set("Content-Type", "application/json")
        .send_json(body)
        .map_err(|e| anyhow!("afm request failed: {e}"))?
        .into_json()
        .context("parse afm response")?;
    let data = resp
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| anyhow!("afm response missing data[]"))?;
    let mut out = Vec::with_capacity(data.len());
    for item in data {
        let emb = item
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or_else(|| anyhow!("afm item missing embedding"))?;
        out.push(emb.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect());
    }
    Ok(out)
}

impl AfmBackend {
    pub fn new(url: &str, model: &str, concurrency: usize) -> Result<Self> {
        let url = if url.contains("/v1/") {
            url.to_string()
        } else {
            format!("{}/v1/embeddings", url.trim_end_matches('/'))
        };
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(120))
            .build();
        // Probe dimensionality.
        let probe = embed_request(&agent, &url, model, &["probe".to_string()])?;
        let dim = probe.first().map(|v| v.len()).unwrap_or(0);
        anyhow::ensure!(dim > 0, "afm returned empty embedding");
        Ok(AfmBackend {
            label: format!("afm-http[{} d{} c{}]", model, dim, concurrency.max(1)),
            url,
            model: model.to_string(),
            dim,
            concurrency: concurrency.max(1),
            agent,
        })
    }
}

impl Backend for AfmBackend {
    fn dim(&self) -> usize {
        self.dim
    }
    fn label(&self) -> &str {
        &self.label
    }

    fn embed(&self, texts: &[String], batch_size: usize) -> Result<Vec<f32>> {
        let n = texts.len();
        let dim = self.dim;
        if n == 0 {
            return Ok(Vec::new());
        }
        // Index the batches so results land in the right place regardless of
        // completion order.
        let batches: Vec<(usize, &[String])> = texts
            .chunks(batch_size.max(1))
            .enumerate()
            .map(|(bi, b)| (bi * batch_size.max(1), b))
            .collect();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.concurrency)
            .build()
            .context("build afm request pool")?;

        let results: Vec<Result<(usize, Vec<Vec<f32>>)>> = pool.install(|| {
            batches
                .par_iter()
                .map(|(start, batch)| {
                    let vecs = embed_request(&self.agent, &self.url, &self.model, batch)?;
                    Ok((*start, vecs))
                })
                .collect()
        });

        let mut flat = vec![0f32; n * dim];
        for r in results {
            let (start, vecs) = r?;
            for (k, v) in vecs.into_iter().enumerate() {
                anyhow::ensure!(v.len() == dim, "afm dim mismatch: {} vs {}", v.len(), dim);
                let idx = start + k;
                flat[idx * dim..(idx + 1) * dim].copy_from_slice(&v);
            }
        }
        // afm may already L2-normalize; do it anyway so cosine == dot, matching
        // the other backends.
        l2_normalize_rows(&mut flat, dim);
        Ok(flat)
    }
}
