//! Extract a benchmark corpus from a git repo using the real chunker.
//!
//!   corpus --repo ~/src/exe --out bench/corpus/exe.bin
//!
//! Mirrors `git-vector-grep index`'s workload: tracked textual files, deduped
//! by blob SHA, chunked with the production chunker. The resulting chunk-text
//! count matches what indexing would embed.

use anyhow::Result;
use clap::Parser;
use git_vector_grep::chunker::chunk_bytes;
use git_vector_grep::repo::{find_repo_root, git_blob_sha1, list_tracked, looks_textual, modified_paths};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Extract a stable embedding benchmark corpus from a git repo")]
struct Args {
    /// Repo to extract from.
    #[arg(long)]
    repo: PathBuf,
    /// Output corpus file.
    #[arg(long)]
    out: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = find_repo_root(&args.repo)?;

    // (path, blob_sha) for tracked textual files, recomputing SHA for files
    // whose working tree differs from the index.
    let tracked: Vec<_> = list_tracked(&root)?
        .into_iter()
        .filter(|f| looks_textual(&f.path))
        .collect();
    let modified: HashSet<String> = modified_paths(&root)?.into_iter().collect();
    let pairs: Vec<(String, String)> = tracked
        .into_par_iter()
        .filter_map(|f| {
            if modified.contains(&f.path) {
                let bytes = std::fs::read(root.join(&f.path)).ok()?;
                Some((f.path.clone(), git_blob_sha1(&bytes)))
            } else {
                Some((f.path, f.index_blob_sha))
            }
        })
        .collect();

    // One canonical path per unique blob SHA.
    let mut sha_to_path: HashMap<String, String> = HashMap::new();
    for (path, sha) in &pairs {
        sha_to_path.entry(sha.clone()).or_insert_with(|| path.clone());
    }
    let blobs: Vec<(String, String)> = sha_to_path.into_iter().collect();

    let texts: Vec<String> = blobs
        .par_iter()
        .flat_map_iter(|(_sha, path)| {
            let bytes = std::fs::read(root.join(path)).unwrap_or_default();
            chunk_bytes(&bytes).into_iter().map(|c| c.text)
        })
        .collect();

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    vgg_bench::corpus::write(&args.out, &texts)?;

    let bytes: usize = texts.iter().map(|t| t.len()).sum();
    println!(
        "wrote {} chunks ({} unique blobs, {} files) -> {}  [{:.1} MB text]",
        texts.len(),
        blobs.len(),
        pairs.len(),
        args.out.display(),
        bytes as f64 / 1e6,
    );
    Ok(())
}
