//! Run one embedding backend over a corpus and report throughput + peak RSS.
//!
//!   bench --corpus bench/corpus/exe.bin --backend onnx-cpu   --model minilm
//!   bench --corpus bench/corpus/exe.bin --backend onnx-coreml --model jina-code --coreml-units ane
//!   bench --corpus bench/corpus/exe.bin --backend candle-metal --model jina-code   # needs --features candle
//!
//! Always runs under a memory watchdog (default 16 GB) that aborts the process
//! before it can exhaust RAM. Optionally dumps a slice of output vectors so a
//! separate run can verify two backends agree (cosine ~= 1.0).

use anyhow::{bail, Result};
use clap::Parser;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use vgg_bench::backends::{Backend, CoreMlUnits, OnnxBackend, OnnxProvider};
use vgg_bench::mem::Watchdog;

#[derive(Parser)]
#[command(about = "Benchmark an embedding backend over a corpus")]
struct Args {
    /// Corpus file produced by the `corpus` binary.
    #[arg(long)]
    corpus: PathBuf,
    /// Backend: onnx-cpu | onnx-coreml | candle-metal
    #[arg(long)]
    backend: String,
    /// Model: minilm | bge-small | bge-small-q | bge-base | jina-code
    #[arg(long, default_value = "minilm")]
    model: String,
    /// Per-call batch size.
    #[arg(long, default_value_t = 16)]
    batch: usize,
    /// ONNX CPU: number of parallel sessions (ignored for CoreML/candle).
    #[arg(long, default_value_t = 0)]
    sessions: usize,
    /// ONNX CPU: intra-op threads per session.
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// CoreML compute units: all | ane | gpu | cpu
    #[arg(long, default_value = "all")]
    coreml_units: String,
    /// CoreML: use NeuralNetwork format instead of MLProgram.
    #[arg(long)]
    coreml_nn: bool,
    /// CoreML: pad to a fixed batch (static input shapes — helps the ANE).
    #[arg(long)]
    coreml_static: bool,
    /// candle: load weights in F16 (much faster on Apple GPU).
    #[arg(long)]
    f16: bool,
    /// onnx-direct: fixed sequence length (CoreML needs a constant shape).
    #[arg(long, default_value_t = 256)]
    seq_len: usize,
    /// afm-http: base URL of the `afm embed` server.
    #[arg(long, default_value = "http://127.0.0.1:9998")]
    afm_url: String,
    /// Cap the corpus to the first N texts (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// Memory budget in GB; process aborts if RSS crosses it.
    #[arg(long, default_value_t = 16.0)]
    budget_gb: f64,
    /// Sort texts by length before embedding (less padding per batch).
    #[arg(long)]
    sort: bool,
    /// Warm-up run over the first `warmup` texts before timing (CoreML compiles
    /// the model on first use; warming gives a fair steady-state number).
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    /// Dump the first N output vectors here for a correctness comparison.
    #[arg(long)]
    dump: Option<PathBuf>,
    /// Compare this run's first-N vectors against a reference dump (cosine).
    #[arg(long)]
    compare: Option<PathBuf>,
}

fn build_backend(args: &Args) -> Result<Box<dyn Backend>> {
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    match args.backend.as_str() {
        "onnx-cpu" => {
            let sessions = if args.sessions == 0 { cpus } else { args.sessions };
            Ok(Box::new(OnnxBackend::new(
                &args.model,
                OnnxProvider::Cpu,
                sessions,
                args.threads,
            )?))
        }
        "onnx-coreml" => {
            let units = match args.coreml_units.as_str() {
                "all" => CoreMlUnits::All,
                "ane" => CoreMlUnits::Ane,
                "gpu" => CoreMlUnits::Gpu,
                "cpu" => CoreMlUnits::Cpu,
                other => bail!("unknown --coreml-units {other}"),
            };
            Ok(Box::new(OnnxBackend::new(
                &args.model,
                OnnxProvider::CoreMl {
                    units,
                    mlprogram: !args.coreml_nn,
                    static_shapes: args.coreml_static,
                },
                1,
                args.threads,
            )?))
        }
        "afm-http" => {
            use vgg_bench::backends::afm_http::AfmBackend;
            // The afm server speaks Apple NL models, not our ONNX model ids.
            let afm_model = if args.model.contains("apple") || args.model.contains("nl") {
                args.model.clone()
            } else {
                "apple-nl-contextual-en".to_string()
            };
            let concurrency = if args.sessions == 0 { 8 } else { args.sessions };
            Ok(Box::new(AfmBackend::new(&args.afm_url, &afm_model, concurrency)?))
        }
        "onnx-direct" => {
            use vgg_bench::backends::onnx_direct::{DirectProvider, OnnxDirectBackend};
            let provider = match args.coreml_units.as_str() {
                "none" | "cpu-noep" => DirectProvider::Cpu,
                "all" => DirectProvider::CoreMl(CoreMlUnits::All),
                "ane" => DirectProvider::CoreMl(CoreMlUnits::Ane),
                "gpu" => DirectProvider::CoreMl(CoreMlUnits::Gpu),
                "cpu" => DirectProvider::CoreMl(CoreMlUnits::Cpu),
                other => bail!("unknown --coreml-units {other}"),
            };
            Ok(Box::new(OnnxDirectBackend::new(
                &args.model,
                provider,
                args.seq_len,
                args.batch,
            )?))
        }
        "candle-metal" => {
            #[cfg(feature = "candle")]
            {
                Ok(Box::new(vgg_bench::backends::candle_backend::CandleBackend::new(&args.model, args.f16)?))
            }
            #[cfg(not(feature = "candle"))]
            {
                bail!("candle-metal requires building with --features candle")
            }
        }
        other => bail!("unknown backend {other}"),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let budget = if args.budget_gb > 0.0 {
        Some((args.budget_gb * 1e9) as u64)
    } else {
        None
    };
    let wd = Watchdog::spawn(budget, Duration::from_millis(150));

    let mut texts = vgg_bench::corpus::read(&args.corpus)?;
    if args.limit > 0 && args.limit < texts.len() {
        texts.truncate(args.limit);
    }
    if args.sort {
        texts.sort_by_key(|t| t.len());
    }
    let n = texts.len();
    let total_bytes: usize = texts.iter().map(|t| t.len()).sum();

    eprintln!(
        "corpus: {} texts ({:.1} MB), backend={}, model={}, batch={}",
        n,
        total_bytes as f64 / 1e6,
        args.backend,
        args.model,
        args.batch
    );

    let t_init = Instant::now();
    let backend = build_backend(&args)?;
    let init_s = t_init.elapsed().as_secs_f64();
    eprintln!("init: {:.2}s ({})", init_s, backend.label());

    if args.warmup > 0 {
        let w = args.warmup.min(n);
        let t = Instant::now();
        let _ = backend.embed(&texts[..w], args.batch)?;
        eprintln!("warmup: {} texts in {:.2}s", w, t.elapsed().as_secs_f64());
    }

    let t0 = Instant::now();
    let vecs = backend.embed(&texts, args.batch)?;
    let secs = t0.elapsed().as_secs_f64();

    let dim = backend.dim();
    let peak_gb = wd.peak_bytes() as f64 / 1e9;
    let rate = n as f64 / secs;

    println!("──────────────────────────────────────────────");
    println!("backend     : {}", backend.label());
    println!("model/dim   : {} / {}", args.model, dim);
    println!("texts       : {}", n);
    println!("embed time  : {:.2} s", secs);
    println!("throughput  : {:.0} chunks/s", rate);
    println!("init time   : {:.2} s", init_s);
    println!("peak RSS    : {:.2} GB", peak_gb);
    println!("──────────────────────────────────────────────");

    if let Some(path) = &args.dump {
        dump_vectors(path, &vecs, dim, 256)?;
        eprintln!("dumped first {} vectors -> {}", 256.min(n), path.display());
    }
    if let Some(path) = &args.compare {
        let reference = load_vectors(path)?;
        let m = (reference.len() / dim).min(n);
        let mut sum = 0f64;
        let mut worst = 1f64;
        for i in 0..m {
            let a = &vecs[i * dim..(i + 1) * dim];
            let b = &reference[i * dim..(i + 1) * dim];
            let mut dot = 0f64;
            for k in 0..dim {
                dot += (a[k] * b[k]) as f64;
            }
            sum += dot;
            if dot < worst {
                worst = dot;
            }
        }
        let mean = sum / m as f64;
        println!("correctness : mean cosine {:.5}, worst {:.5} (n={})", mean, worst, m);
    }

    Ok(())
}

fn dump_vectors(path: &PathBuf, vecs: &[f32], dim: usize, n: usize) -> Result<()> {
    use std::io::Write;
    let n = n.min(vecs.len() / dim);
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&(dim as u32).to_le_bytes())?;
    f.write_all(&(n as u32).to_le_bytes())?;
    f.write_all(bytes_of(&vecs[..n * dim]))?;
    f.flush()?;
    Ok(())
}

fn load_vectors(path: &PathBuf) -> Result<Vec<f32>> {
    use std::io::Read;
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut hdr = [0u8; 8];
    f.read_exact(&mut hdr)?;
    let n = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
    let dim = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as usize;
    let mut buf = vec![0u8; n * dim * 4];
    f.read_exact(&mut buf)?;
    let mut out = vec![0f32; n * dim];
    for (i, chunk) in buf.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(out)
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
