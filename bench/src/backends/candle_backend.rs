//! Candle + Metal backend for jina-embeddings-v2-base-code.
//!
//! Runs the BERT/Jina encoder on the Apple GPU via candle's Metal backend.
//! candle's `jina_bert::BertModel::forward` takes only `input_ids` (no attention
//! mask), so we minimize padding error by sorting inputs by length and padding
//! each batch to its own max; pooling is masked mean over real tokens.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::jina_bert::{BertModel, Config};
use tokenizers::Tokenizer;

use super::Backend;

pub struct CandleBackend {
    label: String,
    device: Device,
    model: BertModel,
    tokenizer: Tokenizer,
    dim: usize,
    max_tokens: usize,
    dtype: DType,
}

impl CandleBackend {
    pub fn new(model: &str, f16: bool) -> Result<Self> {
        let dtype = if f16 { DType::F16 } else { DType::F32 };
        // candle-transformers 0.8 `jina_bert` matches jina v2 base-EN. The
        // base-CODE checkpoint adds QK-LayerNorm + a renamed GLU MLP, so it
        // needs a custom model (see jina_code module). jina-en is the same
        // 768-dim cost class and a faithful throughput proxy.
        let repo_id = match model {
            "jina-en" | "jinaai/jina-embeddings-v2-base-en" => {
                "jinaai/jina-embeddings-v2-base-en"
            }
            "jina-code" | "jinaai/jina-embeddings-v2-base-code" => {
                "jinaai/jina-embeddings-v2-base-code"
            }
            other => anyhow::bail!("candle backend supports jina-en | jina-code, not {other}"),
        };

        let device = Device::new_metal(0).context("create Metal device")?;

        // Fetch weights + tokenizer + config from the HF hub. Reuse fastembed's
        // model cache (already HF-hub-layout) so tokenizer/config aren't
        // re-downloaded. Force a real endpoint: the shell sets HF_ENDPOINT="",
        // which hf-hub would otherwise treat as an (invalid) base URL.
        // hf-hub reads the endpoint from $HF_ENDPOINT; this shell exports it as
        // "", which parses as an invalid base URL. Repair it if blank.
        if std::env::var("HF_ENDPOINT").map(|v| v.is_empty()).unwrap_or(true) {
            std::env::set_var("HF_ENDPOINT", "https://huggingface.co");
        }
        use hf_hub::api::sync::ApiBuilder;
        let api = ApiBuilder::new()
            .with_cache_dir(git_vector_grep::embedder::default_cache_dir())
            .build()?;
        let repo = api.model(repo_id.to_string());
        let tok_path = repo.get("tokenizer.json").context("download tokenizer.json")?;
        let cfg_path = repo.get("config.json").context("download config.json")?;
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .context("download model weights")?;

        // Architecture is jina v2 base; only the vocab differs for the code model.
        let mut config = Config::v2_base();
        let cfg_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&cfg_path)?)?;
        if let Some(v) = cfg_json.get("vocab_size").and_then(|v| v.as_u64()) {
            config.vocab_size = v as usize;
        }
        if let Some(v) = cfg_json.get("max_position_embeddings").and_then(|v| v.as_u64()) {
            config.max_position_embeddings = v as usize;
        }

        let vb = if weights_path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], dtype, &device)?
            }
        } else {
            VarBuilder::from_pth(&weights_path, dtype, &device)?
        };
        let model = BertModel::new(vb, &config).context("build jina BertModel")?;

        let mut tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        // We do our own padding per batch; disable the tokenizer's.
        tokenizer.with_padding(None);

        Ok(CandleBackend {
            label: format!(
                "candle-metal[{} {}]",
                model_short(repo_id),
                if f16 { "f16" } else { "f32" }
            ),
            device,
            model,
            tokenizer,
            dim: config.hidden_size,
            max_tokens: 512,
            dtype,
        })
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<f32>> {
        // Tokenize, truncating long chunks; pad to this batch's max length.
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(self.max_tokens))
            .max()
            .unwrap_or(1)
            .max(1);
        let b = texts.len();

        let mut ids = vec![0u32; b * max_len];
        let mut mask = vec![0f32; b * max_len];
        for (i, enc) in encodings.iter().enumerate() {
            let e_ids = enc.get_ids();
            let n = e_ids.len().min(max_len);
            ids[i * max_len..i * max_len + n].copy_from_slice(&e_ids[..n]);
            for m in &mut mask[i * max_len..i * max_len + n] {
                *m = 1.0;
            }
        }

        let input_ids = Tensor::from_vec(ids, (b, max_len), &self.device)?;
        // Mask must match the model dtype (F16/F32) for broadcast ops.
        let mask_t = Tensor::from_vec(mask, (b, max_len), &self.device)?.to_dtype(self.dtype)?;

        // [b, seq, hidden]
        let hidden = self.model.forward(&input_ids)?;
        // Masked mean pool over the sequence dimension.
        let mask3 = mask_t.unsqueeze(2)?; // [b, seq, 1]
        let summed = hidden.broadcast_mul(&mask3)?.sum(1)?; // [b, hidden]
        let counts = mask_t.sum(1)?.unsqueeze(1)?; // [b, 1]
        let mean = summed.broadcast_div(&counts)?; // [b, hidden]
        // L2 normalize. Pool in F32 for numerical stability under F16 weights.
        let mean = mean.to_dtype(DType::F32)?;
        let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normed = mean.broadcast_div(&norm)?;
        let out: Vec<f32> = normed.flatten_all()?.to_vec1()?;
        Ok(out)
    }
}

fn model_short(repo: &str) -> &str {
    repo.rsplit('/').next().unwrap_or(repo)
}

impl Backend for CandleBackend {
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
        // Sort by length so each padded batch wastes little compute.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| texts[i].len());

        let mut flat = vec![0f32; n * dim];
        for batch in order.chunks(batch_size.max(1)) {
            let batch_texts: Vec<&str> = batch.iter().map(|&i| texts[i].as_str()).collect();
            let vecs = self.embed_batch(&batch_texts)?;
            for (bi, &orig) in batch.iter().enumerate() {
                flat[orig * dim..(orig + 1) * dim]
                    .copy_from_slice(&vecs[bi * dim..(bi + 1) * dim]);
            }
        }
        Ok(flat)
    }
}
