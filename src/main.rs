//! git-vector-grep: fast CPU vector grep over a git repo.

mod chunker;
mod embedder;
mod indexer;
mod repo;
mod search;
mod store;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;

use crate::embedder::Embedder;
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

    /// Number of parallel embedding workers (default: all cores).
    /// Each worker holds its own ONNX session with intra_threads=1.
    #[arg(long, global = true)]
    workers: Option<usize>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// (Re)build the embedding cache to match the current repo state.
    Index {
        #[arg(short, long)]
        verbose: bool,
        /// ONNX batch size per worker. Lower if you OOM; 8 is conservative.
        #[arg(long, default_value_t = 16)]
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
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
    },
    /// Print cache stats.
    Stats {},
    /// Delete the cache.
    Clear {},
    /// Push the cache ref to a remote. History is linear so plain pushes
    /// fast-forward; pass --force to clobber a divergent remote.
    Push {
        /// Remote name (default: origin).
        #[arg(long, default_value = "origin")]
        remote: String,
        /// Force-push (last-writer-wins). The cache is reproducible.
        #[arg(long)]
        force: bool,
    },
    /// Fetch the cache ref from a remote.
    Pull {
        /// Remote name (default: origin).
        #[arg(long, default_value = "origin")]
        remote: String,
    },
    /// Add a fetch refspec so plain `git fetch` picks up the cache ref.
    ConfigRemote {
        /// Remote name (default: origin).
        #[arg(long, default_value = "origin")]
        remote: String,
    },
    /// Collapse history of the cache ref into a single commit and suggest
    /// `git gc --prune=now` to reclaim space.
    Gc {},
}

fn build_embedder(cli: &Cli) -> Result<Embedder> {
    Embedder::new(&cli.model, cli.workers)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let start_dir = cli.repo.clone().unwrap_or_else(|| std::env::current_dir().unwrap());
    let root = find_repo_root(&start_dir)?;

    match cli.cmd {
        Cmd::Index { verbose, batch_size } => {
            let t0 = Instant::now();
            let emb = build_embedder(&cli)?;
            let mut s = Store::open(&root, &emb.model_id, emb.dim)?;
            if verbose {
                eprintln!(
                    "[index] repo={} model={} dim={} workers={} ref={}",
                    root.display(), emb.model_id, emb.dim, emb.workers, store::REF_NAME
                );
            }
            let stats = index_repo(&root, &mut s, &emb, batch_size, verbose)?;
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

            let emb = build_embedder(&cli)?;
            let mut s = Store::open(&root, &emb.model_id, emb.dim)?;

            if !no_auto_index {
                let stats = index_repo(&root, &mut s, &emb, batch_size, verbose)?;
                if verbose {
                    eprintln!("[index] {}", stats);
                }
                s.commit()?;
            }

            let t_load = Instant::now();
            let idx = Index::load(&root, &s)?;
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
                    blob_sha: &'a str,
                    chunk_idx: u32,
                }
                let out: Vec<HitJson> = hits
                    .iter()
                    .map(|h| HitJson {
                        path: &h.path,
                        start_line: h.start_line,
                        end_line: h.end_line,
                        score: h.score,
                        blob_sha: &h.blob_sha,
                        chunk_idx: h.chunk_idx,
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
            let choice = embedder::parse_model(&cli.model)?;
            let s = Store::open(&root, choice.canonical_id, choice.dim).context("open store")?;
            let id = choice.canonical_id;
            let pairs = crate::indexer::list_tracked_with_blobs(&root)?;
            let n_files = pairs.len();
            let known = s.known_blob_shas()?;
            let n_blobs = known.len();
            let mut n_chunks: u64 = 0;
            let blobs_size = {
                let payloads = s.iter_all_payloads()?;
                let mut total = 0u64;
                for (_sha, b) in &payloads {
                    total += b.len() as u64;
                    if let Some(n) = crate::store::peek_n(b) {
                        n_chunks += n as u64;
                    }
                }
                total
            };
            println!("ref:      {}", store::REF_NAME);
            println!("model:    {}", id);
            println!("files:    {} tracked (textual)", n_files);
            println!("chunks:   {}", n_chunks);
            println!("blobs:    {} unique", n_blobs);
            println!("payload:  {:.1} MB (uncompressed; git packs further)", blobs_size as f64 / 1e6);
        }
        Cmd::Push { remote, force } => {
            let refspec = format!("{0}:{0}", store::REF_NAME);
            let mut args: Vec<&str> = vec!["-C"];
            let root_s = root.to_string_lossy().into_owned();
            args.push(&root_s);
            args.push("push");
            if force {
                args.push("--force");
            }
            args.push(&remote);
            args.push(&refspec);
            let status = std::process::Command::new("git").args(&args).status()?;
            if !status.success() {
                anyhow::bail!("git push failed");
            }
        }
        Cmd::Pull { remote } => {
            let refspec = format!("+{0}/*:{0}/*", "refs/vector-grep");
            let status = std::process::Command::new("git")
                .arg("-C").arg(&root)
                .args(["fetch", &remote, &refspec])
                .status()?;
            if !status.success() {
                anyhow::bail!("git fetch failed");
            }
        }
        Cmd::ConfigRemote { remote } => {
            let key = format!("remote.{}.fetch", remote);
            let val = "+refs/vector-grep/*:refs/vector-grep/*";
            // Check if already present; only add if missing.
            let existing = std::process::Command::new("git")
                .arg("-C").arg(&root)
                .args(["config", "--get-all", &key])
                .output()?;
            let body = String::from_utf8_lossy(&existing.stdout);
            if body.lines().any(|l| l.trim() == val) {
                println!("already configured");
            } else {
                let status = std::process::Command::new("git")
                    .arg("-C").arg(&root)
                    .args(["config", "--add", &key, val])
                    .status()?;
                if !status.success() {
                    anyhow::bail!("git config --add failed");
                }
                println!("added fetch refspec to remote.{}", remote);
            }
        }
        Cmd::Gc {} => {
            // Rewrite the ref to a single commit pointing at the current tree.
            let head = std::process::Command::new("git")
                .arg("-C").arg(&root)
                .args(["rev-parse", "--verify", "--quiet", store::REF_NAME])
                .output()?;
            if !head.status.success() {
                println!("no cache ref to gc");
                return Ok(());
            }
            let tip = String::from_utf8(head.stdout)?.trim().to_string();
            let tree = std::process::Command::new("git")
                .arg("-C").arg(&root)
                .args(["rev-parse", &format!("{}^{{tree}}", tip)])
                .output()?;
            anyhow::ensure!(tree.status.success(), "git rev-parse tree failed");
            let tree = String::from_utf8(tree.stdout)?.trim().to_string();
            let ct = std::process::Command::new("git")
                .arg("-C").arg(&root)
                .env("GIT_AUTHOR_NAME", "git-vector-grep")
                .env("GIT_AUTHOR_EMAIL", "vgrep@local")
                .env("GIT_COMMITTER_NAME", "git-vector-grep")
                .env("GIT_COMMITTER_EMAIL", "vgrep@local")
                .args(["commit-tree", &tree, "-m", "git-vector-grep gc"])
                .output()?;
            anyhow::ensure!(ct.status.success(), "git commit-tree failed: {}", String::from_utf8_lossy(&ct.stderr));
            let new_tip = String::from_utf8(ct.stdout)?.trim().to_string();
            let up = std::process::Command::new("git")
                .arg("-C").arg(&root)
                .args(["update-ref", store::REF_NAME, &new_tip, &tip])
                .status()?;
            anyhow::ensure!(up.success(), "git update-ref failed");
            println!("collapsed history: {} -> {}", &tip[..12], &new_tip[..12]);
            println!("run `git -C {} gc --prune=now` to reclaim space", root.display());
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
