//! Embedding backends under test. Each produces L2-normalized row-major f32
//! vectors so results are directly comparable (cosine == dot product).

use anyhow::{Context, Result};
use git_vector_grep::embedder::{l2_normalize_rows, parse_model, ModelChoice};
use std::path::PathBuf;
use std::sync::Mutex;

#[cfg(feature = "candle")]
pub mod candle_backend;
pub mod afm_http;
pub mod onnx_direct;

pub trait Backend: Send + Sync {
    /// Embed all `texts`, returning a flat `texts.len() * dim` normalized buffer.
    fn embed(&self, texts: &[String], batch_size: usize) -> Result<Vec<f32>>;
    fn dim(&self) -> usize;
    fn label(&self) -> &str;
}

fn cache_dir() -> PathBuf {
    git_vector_grep::embedder::default_cache_dir()
}

/// Which ONNX execution provider to register, and how.
#[derive(Clone, Debug)]
pub enum OnnxProvider {
    Cpu,
    CoreMl {
        units: CoreMlUnits,
        mlprogram: bool,
        static_shapes: bool,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum CoreMlUnits {
    All,
    Ane,
    Gpu,
    Cpu,
}

/// ONNX backend (fastembed). One or more `TextEmbedding` sessions; work is
/// sharded across them with rayon. For CoreML, a single session is correct —
/// the execution provider owns the GPU/ANE and extra sessions only contend.
pub struct OnnxBackend {
    label: String,
    dim: usize,
    sessions: Vec<Mutex<fastembed::TextEmbedding>>,
    /// Texts handed to each `embed()` call. fastembed retains every batch's
    /// full hidden-state tensor until the whole call returns, so feeding it the
    /// entire corpus at once costs tens of GB. We cap each call to this many
    /// texts to bound peak memory — exactly what the shipping indexer does.
    group: usize,
}

impl OnnxBackend {
    pub fn new(
        model: &str,
        provider: OnnxProvider,
        n_sessions: usize,
        intra_threads: usize,
        group: usize,
    ) -> Result<Self> {
        let choice: ModelChoice = parse_model(model)?;
        let n_sessions = n_sessions.max(1);
        let label = match &provider {
            OnnxProvider::Cpu => format!(
                "onnx-cpu[{}x s{} t{}]",
                choice.short_id, n_sessions, intra_threads
            ),
            OnnxProvider::CoreMl {
                units,
                mlprogram,
                static_shapes,
            } => format!(
                "onnx-coreml[{} {:?} {} {}]",
                choice.short_id,
                units,
                if *mlprogram { "mlprog" } else { "nn" },
                if *static_shapes { "static" } else { "dyn" }
            ),
        };

        // Build sessions. For a single session (CoreML), build inline on the
        // calling thread — CoreML/Metal objects have thread affinity and must be
        // created and used on the same thread, so we never hand them to rayon.
        let sessions = if n_sessions == 1 {
            vec![Mutex::new(build_session(&choice, &provider, intra_threads)?)]
        } else {
            use rayon::prelude::*;
            (0..n_sessions)
                .into_par_iter()
                .map(|_| build_session(&choice, &provider, intra_threads).map(Mutex::new))
                .collect::<Result<Vec<_>>>()?
        };

        let group = group.max(1);

        Ok(OnnxBackend {
            label,
            dim: choice.dim,
            sessions,
            group,
        })
    }
}

fn build_session(
    choice: &ModelChoice,
    provider: &OnnxProvider,
    intra_threads: usize,
) -> Result<fastembed::TextEmbedding> {
    use fastembed::{InitOptions, TextEmbedding};
    std::fs::create_dir_all(cache_dir()).ok();
    let mut opts = InitOptions::new(choice.enum_id.clone())
        .with_show_download_progress(false)
        .with_cache_dir(cache_dir())
        .with_intra_threads(intra_threads.max(1));

    if let OnnxProvider::CoreMl {
        units,
        mlprogram,
        static_shapes,
    } = provider
    {
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
            .with_model_format(if *mlprogram {
                ModelFormat::MLProgram
            } else {
                ModelFormat::NeuralNetwork
            })
            .with_static_input_shapes(*static_shapes)
            .build();
        opts = opts.with_execution_providers(vec![ep]);
    }

    TextEmbedding::try_new(opts).context("failed to init fastembed session")
}

impl Backend for OnnxBackend {
    fn dim(&self) -> usize {
        self.dim
    }
    fn label(&self) -> &str {
        &self.label
    }

    fn embed(&self, texts: &[String], batch_size: usize) -> Result<Vec<f32>> {
        use rayon::prelude::*;
        let n = texts.len();
        let dim = self.dim;
        if n == 0 {
            return Ok(Vec::new());
        }
        // Single session (CoreML): run inline on the calling thread to respect
        // CoreML/Metal thread affinity. No rayon.
        if self.sessions.len() == 1 {
            let mut session = self.sessions[0].lock().unwrap();
            let mut flat = vec![0f32; n * dim];
            let mut pos = 0usize;
            for chunk in texts.chunks(self.group) {
                let vecs = session.embed(chunk.to_vec(), Some(batch_size))?;
                for v in vecs {
                    anyhow::ensure!(v.len() == dim, "dim mismatch: {} vs {}", v.len(), dim);
                    flat[pos * dim..(pos + 1) * dim].copy_from_slice(&v);
                    pos += 1;
                }
            }
            l2_normalize_rows(&mut flat, dim);
            return Ok(flat);
        }

        let n_workers = self.sessions.len().min(n).max(1);
        // Round-robin shards keep similar-length runs together when the caller
        // pre-sorts by length (less padding per batch).
        let mut shards: Vec<Vec<(usize, String)>> = (0..n_workers).map(|_| Vec::new()).collect();
        for (i, t) in texts.iter().enumerate() {
            shards[i % n_workers].push((i, t.clone()));
        }
        let results: Vec<Result<Vec<(usize, Vec<f32>)>>> = shards
            .into_par_iter()
            .enumerate()
            .map(|(wi, shard)| {
                let mut session = self.sessions[wi].lock().unwrap();
                let (idxs, texts_only): (Vec<usize>, Vec<String>) = shard.into_iter().unzip();
                // Embed in bounded groups: fastembed keeps every batch's hidden
                // state alive until the call returns, so a whole-shard call is
                // unbounded. One group at a time keeps peak memory flat.
                let mut out: Vec<Vec<f32>> = Vec::with_capacity(idxs.len());
                for chunk in texts_only.chunks(self.group) {
                    let vecs = session.embed(chunk.to_vec(), Some(batch_size))?;
                    out.extend(vecs);
                }
                Ok(idxs.into_iter().zip(out.into_iter()).collect())
            })
            .collect();

        let mut flat = vec![0f32; n * dim];
        for r in results {
            for (idx, v) in r? {
                anyhow::ensure!(v.len() == dim, "dim mismatch: {} vs {}", v.len(), dim);
                flat[idx * dim..(idx + 1) * dim].copy_from_slice(&v);
            }
        }
        l2_normalize_rows(&mut flat, dim);
        Ok(flat)
    }
}
