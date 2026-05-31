//! git-vector-grep: fast CPU vector grep over a git repo.

mod chunker;
mod embedder;
mod indexer;
mod repo;
mod search;
mod store;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::embedder::{AnyEmbedder, Embedder, RemoteEmbedder};
use crate::store::Store;
use crate::indexer::index_repo;
use crate::repo::find_repo_root;
use crate::search::Index;

#[derive(Parser, Debug)]
#[command(name = "git-vector-grep", version, about = "Vector grep over a git repo")]
struct Cli {
    /// Path inside the git repo (default: cwd).
    #[arg(long, global = true)]
    repo: Option<PathBuf>,

    /// Embedding model: jina-code (default), jina-en, bge-small, bge-base, minilm.
    #[arg(long, global = true, default_value = "minilm")]
    model: String,

    /// Override ONNX intra-op threads (default: all cores).
    #[arg(long, global = true)]
    threads: Option<usize>,

    /// Use the exe.dev LLM gateway for embeddings instead of local ONNX.
    /// Much faster on small VMs; requires network. The cache is keyed by
    /// `--remote-model` so it won't collide with local-model caches.
    #[arg(long, global = true)]
    remote: bool,

    /// Remote embedding model id (default: openai/text-embedding-3-small).
    #[arg(long, global = true, default_value = "openai/text-embedding-3-small")]
    remote_model: String,

    /// Remote embedding dim (default 512: a Matryoshka truncation of
    /// text-embedding-3-small that keeps most of the quality at 1/3 the
    /// storage).
    #[arg(long, global = true, default_value_t = 512)]
    remote_dim: usize,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// (Re)build the embedding cache to match the current repo state.
    Index {
        #[arg(short, long)]
        verbose: bool,
        #[arg(long, default_value_t = 64)]
        batch_size: usize,
    },
    /// Search the repo. Will refresh the cache first unless --no-auto-index.
    Search {
        /// Query string.
        query: Vec<String>,
        #[arg(short = 'k', long, default_value_t = 10)]
        top_k: usize,
        /// Restrict matches to paths starting with this prefix.
        #[arg(long)]
        path: Option<String>,
        /// Print the matching chunk text.
        #[arg(long)]
        show: bool,
        /// Skip the incremental reindex.
        #[arg(long)]
        no_auto_index: bool,
        /// Emit JSON.
        #[arg(long)]
        json: bool,
        #[arg(short, long)]
        verbose: bool,
        #[arg(long, default_value_t = 64)]
        batch_size: usize,
    },
    /// Print cache stats.
    Stats {},
    /// Delete the cache.
    Clear {},
}

fn build_embedder(cli: &Cli) -> Result<AnyEmbedder> {
    if cli.remote {
        Ok(AnyEmbedder::Remote(RemoteEmbedder::exe_gateway(
            &cli.remote_model,
            cli.remote_dim,
        )))
    } else {
        Ok(AnyEmbedder::Local(Embedder::new(&cli.model, cli.threads)?))
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let start_dir = cli.repo.clone().unwrap_or_else(|| std::env::current_dir().unwrap());
    let root = find_repo_root(&start_dir)?;

    match cli.cmd {
        Cmd::Index { verbose, batch_size } => {
            let t0 = Instant::now();
            let mut emb = build_embedder(&cli)?;
            let mut s = Store::open(&root, emb.model_id(), emb.dim())?;
            if verbose {
                eprintln!(
                    "[index] repo={} model={} dim={} ref={}",
                    root.display(), emb.model_id(), emb.dim(), store::REF_NAME
                );
            }
            let stats = index_repo(&root, &mut s, &mut emb, batch_size, verbose)?;
            s.commit()?;
            eprintln!("[index] {}", stats);
            eprintln!("[index] wall: {:.2}s", t0.elapsed().as_secs_f64());
        }
        Cmd::Search {
            ref query,
            top_k,
            ref path,
            show,
            no_auto_index,
            json,
            verbose,
            batch_size,
        } => {
            let t0 = Instant::now();
            let q = query.join(" ");
            anyhow::ensure!(!q.is_empty(), "query is empty");

            let mut emb = build_embedder(&cli)?;
            let mut s = Store::open(&root, emb.model_id(), emb.dim())?;

            if !no_auto_index {
                let stats = index_repo(&root, &mut s, &mut emb, batch_size, verbose)?;
                if verbose {
                    eprintln!("[index] {}", stats);
                }
                s.commit()?;
            }

            let t_load = Instant::now();
            let idx = Index::load(&s)?;
            if verbose {
                eprintln!(
                    "[search] loaded {} vectors in {:.3}s",
                    idx.len(),
                    t_load.elapsed().as_secs_f64()
                );
            }

            let t_q = Instant::now();
            let qv = emb.embed_query(&q)?;
            if verbose {
                eprintln!("[search] query embedded in {:.3}s", t_q.elapsed().as_secs_f64());
            }

            let t_s = Instant::now();
            let hits = idx.search(&qv, top_k, path.as_deref());
            if verbose {
                eprintln!(
                    "[search] top-{} in {:.1}ms (total {:.2}s)",
                    top_k,
                    t_s.elapsed().as_secs_f64() * 1000.0,
                    t0.elapsed().as_secs_f64()
                );
            }

            if json {
                #[derive(serde::Serialize)]
                struct HitJson<'a> {
                    path: &'a str,
                    start_line: u32,
                    end_line: u32,
                    score: f32,
                }
                let out: Vec<HitJson> = hits
                    .iter()
                    .map(|h| HitJson {
                        path: &h.path,
                        start_line: h.start_line,
                        end_line: h.end_line,
                        score: h.score,
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&out)?);
            } else {
                for h in &hits {
                    println!(
                        "{:.4}  {}:{}-{}",
                        h.score, h.path, h.start_line, h.end_line
                    );
                    if show {
                        let full = root.join(&h.path);
                        if let Ok(text) = std::fs::read_to_string(&full) {
                            let lines: Vec<&str> = text.split('\n').collect();
                            let s = (h.start_line as usize).saturating_sub(1);
                            let e = (h.end_line as usize).min(lines.len());
                            for ln in &lines[s..e] {
                                println!("    {}", ln);
                            }
                            println!();
                        }
                    }
                }
            }
        }
        Cmd::Stats {} => {
            let (id, dim) = if cli.remote {
                (cli.remote_model.as_str(), cli.remote_dim)
            } else {
                let (_e, id, dim) = embedder::parse_model(&cli.model)?;
                (id, dim)
            };
            let s = Store::open(&root, id, dim).context("open store")?;
            let n_files = s.files.len();
            let known = s.known_blob_shas()?;
            let n_blobs = known.len();
            // Sum chunks across all payloads in one batch.
            let mut n_chunks: u64 = 0;
            let blobs_size = {
                let payloads = s.iter_all_payloads()?;
                let mut total = 0u64;
                for (_p, _sha, b) in &payloads {
                    total += b.len() as u64;
                    if let Some(n) = crate::store::peek_n(b) {
                        n_chunks += n as u64;
                    }
                }
                total
            };
            println!("ref:      {}", store::REF_NAME);
            println!("model:    {}", id);
            println!("files:    {}", n_files);
            println!("chunks:   {}", n_chunks);
            println!("blobs:    {} unique", n_blobs);
            println!("payload:  {:.1} MB (uncompressed; git packs further)", blobs_size as f64 / 1e6);
        }
        Cmd::Clear {} => {
            let out = std::process::Command::new("git")
                .arg("-C").arg(&root)
                .args(["update-ref", "-d", store::REF_NAME])
                .output()?;
            if out.status.success() {
                println!("deleted ref {}", store::REF_NAME);
            } else {
                eprintln!("git update-ref: {}", String::from_utf8_lossy(&out.stderr));
            }
        }
    }
    Ok(())
}
