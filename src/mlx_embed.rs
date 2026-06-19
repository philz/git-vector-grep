//! Apple-GPU embedding backend via mlx-rs.
//!
//! Implements a standard BERT encoder (bge-small / MiniLM-class) directly with
//! mlx-rs ops and runs it on the Metal GPU. Weights load from the same HF
//! safetensors the rest of the ecosystem uses; tokenization is the pure-Rust
//! `tokenizers` crate. Pooling is mean or CLS per model, then L2-normalize.
//!
//! Only compiled on Apple Silicon (`cfg(mlx)`, set by build.rs).

use anyhow::{anyhow, Context, Result};
use mlx_rs::error::Exception;
use mlx_rs::ops;
use mlx_rs::{fast, Array, Device};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Map `Result<_, Exception>` into `anyhow` (Exception isn't Send+Sync-friendly).
trait MlxRes<T> {
    fn m(self) -> Result<T>;
}
impl<T> MlxRes<T> for Result<T, Exception> {
    fn m(self) -> Result<T> {
        self.map_err(|e| anyhow!("mlx: {e}"))
    }
}

#[derive(Clone, Copy)]
enum Pooling {
    Mean,
    Cls,
}

pub struct MlxModelSpec {
    pub short_id: &'static str,
    pub repo: &'static str,
    pub dim: usize,
    pooling: Pooling,
}

/// Resolve a model name to its MLX spec. Returns short_id/dim without loading.
pub fn parse_mlx_model(name: &str) -> Result<MlxModelSpec> {
    Ok(match name {
        "bge-small" | "BAAI/bge-small-en-v1.5" => MlxModelSpec {
            short_id: "mlx-bge-small",
            repo: "BAAI/bge-small-en-v1.5",
            dim: 384,
            pooling: Pooling::Cls,
        },
        "minilm" | "sentence-transformers/all-MiniLM-L6-v2" => MlxModelSpec {
            short_id: "mlx-minilm",
            repo: "sentence-transformers/all-MiniLM-L6-v2",
            dim: 384,
            pooling: Pooling::Mean,
        },
        "bge-base" | "BAAI/bge-base-en-v1.5" => MlxModelSpec {
            short_id: "mlx-bge-base",
            repo: "BAAI/bge-base-en-v1.5",
            dim: 768,
            pooling: Pooling::Cls,
        },
        other => anyhow::bail!("unknown mlx model: {other} (try bge-small | minilm | bge-base)"),
    })
}

struct BertConfig {
    layers: usize,
    heads: usize,
    eps: f32,
}

pub struct MlxEmbedder {
    short_id: &'static str,
    model_id: String,
    dim: usize,
    pooling: Pooling,
    cfg: BertConfig,
    w: HashMap<String, Array>,
    tokenizer: Tokenizer,
    max_tokens: usize,
    // mlx-rs Arrays are not Sync; serialize GPU access (the GPU is one device).
    lock: Mutex<()>,
}

// All GPU access goes through `self.lock`; the raw mlx pointers never race.
unsafe impl Send for MlxEmbedder {}
unsafe impl Sync for MlxEmbedder {}

/// Cap MLX's reusable-buffer cache. MLX runs in *unified* memory on Apple
/// silicon, so an uncapped buffer cache grows into system RAM and can OOM the
/// machine over a large, varying-shape workload. This bounds it hard.
fn cap_mlx_memory(cache_bytes: usize) {
    unsafe {
        let mut prev: usize = 0;
        mlx_sys::mlx_set_cache_limit(&mut prev as *mut usize, cache_bytes);
    }
}

/// Release cached GPU buffers (called between groups so peak memory stays flat).
fn clear_mlx_cache() {
    unsafe {
        mlx_sys::mlx_clear_cache();
    }
}

fn cache_dir() -> PathBuf {
    crate::embedder::default_cache_dir()
}

fn fetch(repo: &str, file: &str) -> Result<PathBuf> {
    if std::env::var("HF_ENDPOINT").map(|v| v.is_empty()).unwrap_or(true) {
        std::env::set_var("HF_ENDPOINT", "https://huggingface.co");
    }
    use hf_hub::api::sync::ApiBuilder;
    let api = ApiBuilder::new().with_cache_dir(cache_dir()).build()?;
    api.model(repo.to_string())
        .get(file)
        .with_context(|| format!("download {repo}/{file}"))
}

impl MlxEmbedder {
    pub fn new(model_name: &str) -> Result<Self> {
        let spec = parse_mlx_model(model_name)?;
        // GPU by default for all compute.
        Device::set_default(&Device::gpu());
        // Hard cap on MLX's unified-memory buffer cache (512 MB is ample for
        // batched BERT inference and keeps peak RSS bounded).
        cap_mlx_memory(512 * 1024 * 1024);

        let weights = fetch(spec.repo, "model.safetensors")?;
        let cfg_path = fetch(spec.repo, "config.json")?;
        let tok_path = fetch(spec.repo, "tokenizer.json")?;

        let w = Array::load_safetensors(weights.to_string_lossy().as_ref())
            .map_err(|e| anyhow!("load safetensors: {e}"))?;

        let cfg_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&cfg_path)?)?;
        let cfg = BertConfig {
            layers: cfg_json["num_hidden_layers"].as_u64().unwrap_or(12) as usize,
            heads: cfg_json["num_attention_heads"].as_u64().unwrap_or(12) as usize,
            eps: cfg_json["layer_norm_eps"].as_f64().unwrap_or(1e-12) as f32,
        };

        let tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("load tokenizer: {e}"))?;

        Ok(MlxEmbedder {
            short_id: spec.short_id,
            model_id: spec.repo.to_string(),
            dim: spec.dim,
            pooling: spec.pooling,
            cfg,
            w,
            tokenizer,
            max_tokens: 512,
            lock: Mutex::new(()),
        })
    }

    fn g(&self, name: &str) -> Result<&Array> {
        self.w.get(name).ok_or_else(|| anyhow!("missing tensor {name}"))
    }

    /// Linear: x @ W^T + b  (W is [out, in]).
    fn linear(&self, x: &Array, prefix: &str) -> Result<Array> {
        let w = self.g(&format!("{prefix}.weight"))?;
        let b = self.g(&format!("{prefix}.bias"))?;
        ops::addmm(b, x, w.t(), None, None).m()
    }

    fn layer_norm(&self, x: &Array, prefix: &str) -> Result<Array> {
        let w = self.g(&format!("{prefix}.weight"))?;
        let b = self.g(&format!("{prefix}.bias"))?;
        fast::layer_norm(x, Some(w), Some(b), self.cfg.eps).m()
    }

    /// Embed one already-tokenized, padded batch. ids/mask are [b, s] flattened.
    fn forward(&self, ids: &[i32], mask: &[i32], b: usize, s: usize) -> Result<Vec<f32>> {
        let _guard = self.lock.lock().unwrap();
        let h = self.dim as i32;
        let bi = b as i32;
        let si = s as i32;

        // --- embeddings: word + position + token_type, then LayerNorm ---
        let ids_arr = Array::from_slice(ids, &[(b * s) as i32]);
        let word = self.g("embeddings.word_embeddings.weight")?.take_axis(&ids_arr, 0)
            .m()?
            .reshape(&[bi, si, h])
            .m()?;
        let positions: Vec<i32> = (0..s as i32).collect();
        let pos_idx = Array::from_slice(&positions, &[si]);
        let pos = self.g("embeddings.position_embeddings.weight")?.take_axis(&pos_idx, 0).m()?;
        let type0 = Array::from_slice(&[0i32], &[1]);
        let typ = self.g("embeddings.token_type_embeddings.weight")?.take_axis(&type0, 0).m()?;
        let emb = ops::add(&ops::add(&word, &pos).m()?, &typ).m()?;
        let mut x = self.layer_norm(&emb, "embeddings.LayerNorm")?;

        // --- additive attention mask [b,1,1,s]: 0 keep, -1e9 pad ---
        let mask_f: Vec<f32> = mask.iter().map(|&m| if m == 0 { -1e9 } else { 0.0 }).collect();
        let amask = Array::from_slice(&mask_f, &[bi, 1, 1, si]);

        let nh = self.cfg.heads as i32;
        let hd = h / nh;
        let scale = 1.0f32 / (hd as f32).sqrt();

        for l in 0..self.cfg.layers {
            let p = format!("encoder.layer.{l}");
            // self-attention
            let q = self.to_heads(&self.linear(&x, &format!("{p}.attention.self.query"))?, bi, si, nh, hd)?;
            let k = self.to_heads(&self.linear(&x, &format!("{p}.attention.self.key"))?, bi, si, nh, hd)?;
            let v = self.to_heads(&self.linear(&x, &format!("{p}.attention.self.value"))?, bi, si, nh, hd)?;
            let attn = fast::scaled_dot_product_attention(&q, &k, &v, scale, &amask).m()?;
            // [b,nh,s,hd] -> [b,s,h]
            let attn = ops::transpose_axes(&attn, &[0, 2, 1, 3]).m()?.reshape(&[bi, si, h]).m()?;
            let attn_out = self.linear(&attn, &format!("{p}.attention.output.dense"))?;
            x = self.layer_norm(&ops::add(&x, &attn_out).m()?, &format!("{p}.attention.output.LayerNorm"))?;
            // feed-forward
            let inter = mlx_rs::nn::gelu(&self.linear(&x, &format!("{p}.intermediate.dense"))?).m()?;
            let ffn = self.linear(&inter, &format!("{p}.output.dense"))?;
            x = self.layer_norm(&ops::add(&x, &ffn).m()?, &format!("{p}.output.LayerNorm"))?;
        }

        // --- pooling ---
        let pooled = match self.pooling {
            Pooling::Cls => {
                let zero = Array::from_slice(&[0i32], &[1]);
                x.take_axis(&zero, 1).m()?.reshape(&[bi, h]).m()?
            }
            Pooling::Mean => {
                let mask_keep: Vec<f32> = mask.iter().map(|&m| m as f32).collect();
                let m2 = Array::from_slice(&mask_keep, &[bi, si, 1]);
                let summed = ops::sum_axes(&ops::multiply(&x, &m2).m()?, &[1], false).m()?;
                let counts = ops::sum_axes(&m2, &[1], false).m()?;
                ops::divide(&summed, &counts).m()?
            }
        };
        // L2 normalize
        let norm = ops::sqrt(&ops::sum_axes(&ops::multiply(&pooled, &pooled).m()?, &[1], true).m()?).m()?;
        let normed = ops::divide(&pooled, &norm).m()?;
        normed.eval().m()?;
        Ok(normed.as_slice::<f32>().to_vec())
    }

    fn to_heads(&self, x: &Array, b: i32, s: i32, nh: i32, hd: i32) -> Result<Array> {
        // [b,s,h] -> [b,nh,s,hd]
        ops::transpose_axes(&x.reshape(&[b, s, nh, hd]).m()?, &[0, 2, 1, 3]).m()
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<f32>> {
        let encs = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let s = encs
            .iter()
            .map(|e| e.get_ids().len().min(self.max_tokens))
            .max()
            .unwrap_or(1)
            .max(1);
        let b = texts.len();
        let mut ids = vec![0i32; b * s];
        let mut mask = vec![0i32; b * s];
        for (i, e) in encs.iter().enumerate() {
            let eids = e.get_ids();
            let n = eids.len().min(s);
            for j in 0..n {
                ids[i * s + j] = eids[j] as i32;
                mask[i * s + j] = 1;
            }
        }
        self.forward(&ids, &mask, b, s)
    }
}

impl crate::embed::Embed for MlxEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
    fn short_id(&self) -> &str {
        self.short_id
    }
    fn model_id(&self) -> &str {
        &self.model_id
    }
    fn describe(&self) -> String {
        format!(
            "mlx/apple-gpu · {} · GPU buffer-cache cap 512 MB · ~0.4 GB peak · \
             tweak: --backend cpu to force CPU, --batch-size N for batch size",
            self.short_id
        )
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_batch(&[text])
    }

    fn embed_flat(&self, texts: Vec<String>, batch_size: usize) -> Result<Vec<f32>> {
        let n = texts.len();
        let dim = self.dim;
        if n == 0 {
            return Ok(Vec::new());
        }
        // Sort by length so each padded batch wastes little compute.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| texts[i].len());

        let mut flat = vec![0f32; n * dim];
        for batch in order.chunks(batch_size.max(1)) {
            let bt: Vec<&str> = batch.iter().map(|&i| texts[i].as_str()).collect();
            let vecs = self.embed_batch(&bt)?;
            for (bi, &orig) in batch.iter().enumerate() {
                flat[orig * dim..(orig + 1) * dim]
                    .copy_from_slice(&vecs[bi * dim..(bi + 1) * dim]);
            }
        }
        // Release cached GPU buffers so memory doesn't accumulate across the
        // indexer's groups.
        clear_mlx_cache();
        Ok(flat)
    }
}
