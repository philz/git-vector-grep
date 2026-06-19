//! Direct-`ort` backend with FIXED input shapes, for the CoreML EP (ANE/GPU).
//!
//! The CoreML execution provider compiles a CoreML model per unique input shape
//! and runs poorly (or crashes) on dynamic shapes. fastembed feeds it dynamic,
//! `BatchLongest`-padded batches, so every batch is a new shape — fatal.
//!
//! Here we bypass fastembed: tokenize with FIXED padding to `[batch, seq_len]`,
//! so CoreML compiles exactly one model and the ANE/GPU can run it at speed.
//! The last partial batch is padded with empty rows to keep the shape constant.

use anyhow::{anyhow, Context, Result};
use git_vector_grep::embedder::{l2_normalize_rows, parse_model};
use ort::session::Session;
use ort::value::Tensor;
use std::path::PathBuf;
use std::sync::Mutex;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use super::{Backend, CoreMlUnits};

/// `ort::Error` is not `Send`/`Sync`, so it can't flow through `?` into
/// `anyhow::Error`. Stringify it at the boundary.
fn oe<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("ort: {e}")
}

/// Execution target for the fixed-shape session.
#[derive(Clone, Copy, Debug)]
pub enum DirectProvider {
    Cpu,
    CoreMl(CoreMlUnits),
}

pub struct OnnxDirectBackend {
    label: String,
    dim: usize,
    seq_len: usize,
    batch: usize,
    has_token_type: bool,
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

fn resolve_model_files(canonical_id: &str) -> Result<(PathBuf, PathBuf)> {
    // Reuse fastembed's already-populated HF cache; no network needed.
    if std::env::var("HF_ENDPOINT").map(|v| v.is_empty()).unwrap_or(true) {
        std::env::set_var("HF_ENDPOINT", "https://huggingface.co");
    }
    use hf_hub::api::sync::ApiBuilder;
    let api = ApiBuilder::new()
        .with_cache_dir(git_vector_grep::embedder::default_cache_dir())
        .build()?;
    let repo = api.model(canonical_id.to_string());
    let onnx = repo
        .get("onnx/model.onnx")
        .or_else(|_| repo.get("model.onnx"))
        .context("locate model.onnx")?;
    let tok = repo.get("tokenizer.json").context("locate tokenizer.json")?;
    Ok((onnx, tok))
}

impl OnnxDirectBackend {
    pub fn new(
        model: &str,
        provider: DirectProvider,
        seq_len: usize,
        batch: usize,
    ) -> Result<Self> {
        let choice = parse_model(model)?;
        let (onnx_path, tok_path) = resolve_model_files(choice.canonical_id)?;

        let mut builder = Session::builder().map_err(oe)?;
        let prov_label = match provider {
            DirectProvider::Cpu => "cpu".to_string(),
            DirectProvider::CoreMl(units) => {
                use ort::ep::coreml::{ComputeUnits, ModelFormat};
                use ort::ep::CoreML;
                let cu = match units {
                    CoreMlUnits::All => ComputeUnits::All,
                    CoreMlUnits::Ane => ComputeUnits::CPUAndNeuralEngine,
                    CoreMlUnits::Gpu => ComputeUnits::CPUAndGPU,
                    CoreMlUnits::Cpu => ComputeUnits::CPUOnly,
                };
                let ep = CoreML::default()
                    .with_compute_units(cu)
                    .with_model_format(ModelFormat::MLProgram)
                    .with_static_input_shapes(true)
                    .build();
                builder = builder.with_execution_providers([ep]).map_err(oe)?;
                format!("coreml:{:?}", units)
            }
        };
        let session = builder.commit_from_file(&onnx_path).map_err(oe)?;

        let has_token_type = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");

        // Fixed-length tokenization: pad and truncate to exactly seq_len.
        let mut tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("load tokenizer: {e}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(seq_len),
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: seq_len,
                ..Default::default()
            }))
            .map_err(|e| anyhow!("set truncation: {e}"))?;

        Ok(OnnxDirectBackend {
            label: format!(
                "onnx-direct[{} {} L{} B{}]",
                choice.short_id, prov_label, seq_len, batch
            ),
            dim: choice.dim,
            seq_len,
            batch,
            has_token_type,
            session: Mutex::new(session),
            tokenizer,
        })
    }

    /// Run exactly `self.batch` rows (caller pads short final batches).
    fn run_fixed_batch(&self, texts: &[&str], out: &mut [f32]) -> Result<()> {
        let b = self.batch;
        let l = self.seq_len;
        debug_assert_eq!(texts.len(), b);
        let encs = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("encode: {e}"))?;

        let mut ids = vec![0i64; b * l];
        let mut mask = vec![0i64; b * l];
        let mut types = vec![0i64; b * l];
        for (i, e) in encs.iter().enumerate() {
            let eids = e.get_ids();
            let em = e.get_attention_mask();
            let et = e.get_type_ids();
            for j in 0..l {
                ids[i * l + j] = eids[j] as i64;
                mask[i * l + j] = em[j] as i64;
                types[i * l + j] = et[j] as i64;
            }
        }

        let shape = vec![b as i64, l as i64];
        let mut session = self.session.lock().unwrap();
        let mk = |data: Vec<i64>| Tensor::from_array((shape.clone(), data)).map_err(oe);
        let outputs = if self.has_token_type {
            session
                .run(ort::inputs![
                    "input_ids" => mk(ids)?,
                    "attention_mask" => mk(mask.clone())?,
                    "token_type_ids" => mk(types)?,
                ])
                .map_err(oe)?
        } else {
            session
                .run(ort::inputs![
                    "input_ids" => mk(ids)?,
                    "attention_mask" => mk(mask.clone())?,
                ])
                .map_err(oe)?
        };

        // Find the [b, l, dim] hidden-state output and masked-mean-pool it (CPU).
        let dim = self.dim;
        let mut pooled = false;
        for (_name, val) in outputs.iter() {
            let Ok((sh, data)) = val.try_extract_tensor::<f32>() else {
                continue;
            };
            if sh.len() != 3 || sh[2] as usize != dim {
                continue;
            }
            for bi in 0..b {
                let row = &mut out[bi * dim..(bi + 1) * dim];
                let mut cnt = 0f32;
                for j in 0..l {
                    if mask[bi * l + j] == 0 {
                        continue;
                    }
                    cnt += 1.0;
                    let base = (bi * l + j) * dim;
                    for d in 0..dim {
                        row[d] += data[base + d];
                    }
                }
                if cnt > 0.0 {
                    let inv = 1.0 / cnt;
                    for x in row.iter_mut() {
                        *x *= inv;
                    }
                }
            }
            pooled = true;
            break;
        }
        if !pooled {
            return Err(anyhow!("no [b,l,dim] output found"));
        }
        Ok(())
    }
}

impl Backend for OnnxDirectBackend {
    fn dim(&self) -> usize {
        self.dim
    }
    fn label(&self) -> &str {
        &self.label
    }

    fn embed(&self, texts: &[String], _batch_size: usize) -> Result<Vec<f32>> {
        let n = texts.len();
        let dim = self.dim;
        if n == 0 {
            return Ok(Vec::new());
        }
        // Sort by length so each fixed batch wastes minimal real compute.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| texts[i].len());

        let mut flat = vec![0f32; n * dim];
        let b = self.batch;
        let mut buf = vec![0f32; b * dim];
        let empty = String::new();
        for batch in order.chunks(b) {
            let mut row_texts: Vec<&str> = batch.iter().map(|&i| texts[i].as_str()).collect();
            let actual = row_texts.len();
            while row_texts.len() < b {
                row_texts.push(empty.as_str()); // pad to fixed batch shape
            }
            for x in buf.iter_mut() {
                *x = 0.0;
            }
            self.run_fixed_batch(&row_texts, &mut buf)?;
            for (k, &orig) in batch.iter().enumerate().take(actual) {
                flat[orig * dim..(orig + 1) * dim]
                    .copy_from_slice(&buf[k * dim..(k + 1) * dim]);
            }
        }
        l2_normalize_rows(&mut flat, dim);
        Ok(flat)
    }
}
