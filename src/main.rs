//! git-vector-grep: fast CPU vector grep over a git repo.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;

use git_vector_grep::{embedder, indexer, search, store};
use embedder::Embedder;
use store::Store;
use indexer::index_repo;
use git_vector_grep::repo::find_repo_root;
use search::Index;

#[derive(Parser, Debug)]
#[command(
    name = "git-vector-grep",
    version,
    about = "Semantic (vector) code search over a git repo: embeds your tracked \
             text files locally on CPU, caches the vectors in git, and ranks \
             chunks by meaning rather than exact keywords.",
    args_conflicts_with_subcommands = true
)]
struct Cli {
    /// Path inside the git repo. Defaults to the current directory.
    #[arg(long, global = true)]
    repo: Option<PathBuf>,

    /// Embedding model. One of: minilm, bge-small, bge-small-q, bge-base, jina-code.
    #[arg(long, global = true, default_value = "minilm")]
    model: String,

    /// Number of ONNX sessions (CPU backend). Default 1: a single session that
    /// uses all cores via intra-op threads, in one memory arena (~1 GB). Each
    /// extra session adds another full model + arena, so raise this only to
    /// trade memory for parallelism on a big-RAM machine.
    #[arg(long, global = true)]
    workers: Option<usize>,

    /// Embedding backend: auto | cpu | mlx. `auto` uses the Apple GPU (mlx) on
    /// Apple Silicon when the model supports it, else falls back to CPU/ONNX.
    /// MLX caches live under `mlx-*` notes refs (they coexist with CPU caches).
    #[arg(long, global = true, default_value = "auto")]
    backend: String,

    #[command(subcommand)]
    cmd: Option<Cmd>,

    /// Search query (the default action when no subcommand is given).
    #[command(flatten)]
    search: SearchArgs,
}

#[derive(clap::Args, Debug, Default)]
struct SearchArgs {
    /// Query string.
    query: Vec<String>,
    /// Number of results to return.
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
    /// Suppress indexing progress output.
    #[arg(short, long)]
    quiet: bool,
    #[arg(short, long)]
    verbose: bool,
    /// ONNX batch size per worker. Lower if you OOM.
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// (Re)build the embedding cache to match the current repo state.
    Index {
        #[arg(short, long)]
        verbose: bool,
        /// Suppress indexing progress output.
        #[arg(short, long)]
        quiet: bool,
        /// ONNX batch size per worker. Lower if you OOM; 8 is conservative.
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
    },
    /// Search the repo. Will refresh the cache first unless --no-auto-index.
    Search(SearchArgs),
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

/// Whether to use the MLX (Apple GPU) backend for this invocation.
fn want_mlx(cli: &Cli) -> bool {
    match cli.backend.as_str() {
        "mlx" => true,
        "cpu" => false,
        _ => {
            // auto: use MLX when compiled in and the model has an MLX variant.
            #[cfg(mlx)]
            {
                git_vector_grep::mlx_embed::parse_mlx_model(&cli.model).is_ok()
            }
            #[cfg(not(mlx))]
            {
                false
            }
        }
    }
}

fn build_embedder(cli: &Cli) -> Result<Box<dyn git_vector_grep::embed::Embed>> {
    if want_mlx(cli) {
        #[cfg(mlx)]
        {
            match git_vector_grep::mlx_embed::MlxEmbedder::new(&cli.model) {
                Ok(e) => return Ok(Box::new(e)),
                Err(e) if cli.backend == "auto" => {
                    eprintln!("[mlx] unavailable ({e}); falling back to CPU/ONNX");
                }
                Err(e) => return Err(e),
            }
        }
        #[cfg(not(mlx))]
        if cli.backend == "mlx" {
            anyhow::bail!("--backend mlx is only available on Apple Silicon builds");
        }
    }
    Ok(Box::new(Embedder::new(&cli.model, cli.workers)?))
}

/// Resolve (short_id, dim) without loading a model — for stats/clear/gc.
fn resolve_spec(cli: &Cli) -> Result<(String, usize)> {
    if want_mlx(cli) {
        #[cfg(mlx)]
        {
            let s = git_vector_grep::mlx_embed::parse_mlx_model(&cli.model)?;
            return Ok((s.short_id.to_string(), s.dim));
        }
    }
    let c = embedder::parse_model(&cli.model)?;
    Ok((c.short_id.to_string(), c.dim))
}

fn main() -> Result<()> {
    let mut cli = Cli::parse();

    // No subcommand and no query: print help instead of erroring.
    if cli.cmd.is_none() && cli.search.query.is_empty() {
        use clap::CommandFactory;
        Cli::command().print_help()?;
        println!();
        return Ok(());
    }

    let start_dir = cli.repo.clone().unwrap_or_else(|| std::env::current_dir().unwrap());
    let root = find_repo_root(&start_dir)?;

    let cmd = cli.cmd.take().unwrap_or_else(|| Cmd::Search(std::mem::take(&mut cli.search)));

    match cmd {
        Cmd::Index { verbose, quiet, batch_size } => {
            let t0 = Instant::now();
            let emb = build_embedder(&cli)?;
            let mut s = Store::open(&root, emb.short_id(), emb.dim())?;
            if verbose {
                eprintln!(
                    "[index] repo={} model={} dim={} backend={} ref={}",
                    root.display(), emb.model_id(), emb.dim(), cli.backend, s.ref_name
                );
            }
            let stats = index_repo(&root, &mut s, emb.as_ref(), batch_size, verbose, quiet)?;
            s.commit()?;
            if !quiet {
                eprintln!("[index] {}", stats);
                eprintln!("[index] wall: {:.2}s", t0.elapsed().as_secs_f64());
            }
        }
        Cmd::Search(args) => {
            let SearchArgs {
                query,
                top_k,
                path,
                show,
                no_auto_index,
                json,
                quiet,
                verbose,
                batch_size,
            } = args;
            let t0 = Instant::now();
            let q = query.join(" ");
            anyhow::ensure!(!q.is_empty(), "query is empty (usage: git-vector-grep <search terms>)");

            // JSON output should stay machine-clean; silence progress then.
            let quiet = quiet || json;

            let emb = build_embedder(&cli)?;
            let mut s = Store::open(&root, emb.short_id(), emb.dim())?;

            if !no_auto_index {
                let stats = index_repo(&root, &mut s, emb.as_ref(), batch_size, verbose, quiet)?;
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
            let (short_id, dim) = resolve_spec(&cli)?;
            let s = Store::open(&root, &short_id, dim).context("open store")?;
            let pairs = indexer::list_tracked_with_blobs(&root)?;
            let n_files = pairs.len();
            let known = s.known_blob_shas()?;
            let n_blobs = known.len();
            let mut n_chunks: u64 = 0;
            let blobs_size = {
                let payloads = s.iter_all_payloads()?;
                let mut total = 0u64;
                for (_sha, b) in &payloads {
                    total += b.len() as u64;
                    if let Some(n) = store::peek_n(b) {
                        n_chunks += n as u64;
                    }
                }
                total
            };
            println!("ref:      {}", s.ref_name);
            println!("model:    {} (backend: {})", short_id, cli.backend);
            println!("dim:      {}", dim);
            println!("files:    {} tracked (textual)", n_files);
            println!("chunks:   {}", n_chunks);
            println!("blobs:    {} unique", n_blobs);
            println!("payload:  {:.1} MB (uncompressed; git packs further)", blobs_size as f64 / 1e6);
        }
        Cmd::Push { remote, force } => {
            // Push *all* model caches for this repo. Each model lives at a
            // distinct ref under refs/notes/vector-grep/<short_id>, so a
            // single namespace refspec pushes the lot.
            let refspec = format!("{0}/*:{0}/*", store::NOTES_REF_PREFIX);
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
            let refspec = format!("+{0}/*:{0}/*", store::NOTES_REF_PREFIX);
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
            let val = format!("+{0}/*:{0}/*", store::NOTES_REF_PREFIX);
            // Also wire union merge so `git notes merge` Just Works for
            // the two-clients-pushing-disjoint-blobs case.
            let _ = std::process::Command::new("git")
                .arg("-C").arg(&root)
                .args(["config", "notes.mergeStrategy", "union"])
                .status();
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
                    .args(["config", "--add", &key, &val])
                    .status()?;
                if !status.success() {
                    anyhow::bail!("git config --add failed");
                }
                println!("added fetch refspec to remote.{}", remote);
            }
        }
        Cmd::Gc {} => {
            // Collapse history for the *current model's* cache ref.
            let (short_id, _dim) = resolve_spec(&cli)?;
            let ref_name = store::ref_for(&short_id);
            let head = std::process::Command::new("git")
                .arg("-C").arg(&root)
                .args(["rev-parse", "--verify", "--quiet", &ref_name])
                .output()?;
            if !head.status.success() {
                println!("no cache ref to gc: {}", ref_name);
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
                .args(["update-ref", &ref_name, &new_tip, &tip])
                .status()?;
            anyhow::ensure!(up.success(), "git update-ref failed");
            println!("collapsed history of {}: {} -> {}", ref_name, &tip[..12], &new_tip[..12]);
            println!("run `git -C {} gc --prune=now` to reclaim space", root.display());
        }
        Cmd::Clear {} => {
            // Delete just the current model's ref.
            let (short_id, _dim) = resolve_spec(&cli)?;
            let ref_name = store::ref_for(&short_id);
            let out = std::process::Command::new("git")
                .arg("-C").arg(&root)
                .args(["update-ref", "-d", &ref_name])
                .output()?;
            if out.status.success() {
                println!("deleted ref {}", ref_name);
            } else {
                eprintln!("git update-ref: {}", String::from_utf8_lossy(&out.stderr));
            }
        }
    }
    Ok(())
}
