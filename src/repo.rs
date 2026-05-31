//! Git interaction. We shell out to `git` -- it's already installed, fast,
//! and gives us the index-cached blob SHA for free.

use anyhow::{bail, Context, Result};
use sha1::{Digest, Sha1};
use std::path::{Path, PathBuf};
use std::process::Command;

/// One tracked file as reported by `git ls-files -s -z`.
#[derive(Debug, Clone)]
pub struct TrackedFile {
    pub path: String,
    /// Blob SHA from the index. For an unmodified working tree this is the
    /// content hash. If the working tree differs from the index, we recompute.
    pub index_blob_sha: String,
}

pub fn find_repo_root(start: &Path) -> Result<PathBuf> {
    let out = Command::new("git")
        .args(["-C"])
        .arg(start)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .context("failed to invoke `git`; is it installed?")?;
    if !out.status.success() {
        bail!(
            "not inside a git repo: {} ({})",
            start.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let s = String::from_utf8(out.stdout)?.trim().to_string();
    Ok(PathBuf::from(s))
}

/// Run `git ls-files -s -z` and parse it.
/// Output records: `<mode> SP <sha> SP <stage> TAB <path> NUL`.
pub fn list_tracked(root: &Path) -> Result<Vec<TrackedFile>> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-s", "-z"])
        .output()
        .context("git ls-files failed")?;
    if !out.status.success() {
        bail!(
            "git ls-files: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let mut files = Vec::new();
    for rec in out.stdout.split(|&b| b == 0) {
        if rec.is_empty() {
            continue;
        }
        // Find the TAB that separates `<mode> <sha> <stage>` from the path.
        let tab = rec.iter().position(|&b| b == b'\t');
        let Some(tab) = tab else { continue };
        let head = std::str::from_utf8(&rec[..tab])?;
        let path = std::str::from_utf8(&rec[tab + 1..])?.to_string();
        let mut it = head.split_whitespace();
        let _mode = it.next();
        let sha = it.next().unwrap_or("").to_string();
        let _stage = it.next();
        if sha.is_empty() {
            continue;
        }
        files.push(TrackedFile {
            path,
            index_blob_sha: sha,
        });
    }
    Ok(files)
}

/// Detect paths whose working-tree content differs from the index.
/// Returns a set-as-Vec of relative paths.
pub fn modified_paths(root: &Path) -> Result<Vec<String>> {
    // -z: NUL terminator, --name-only, only tracked files differing from index.
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["diff", "--name-only", "-z"])
        .output()
        .context("git diff failed")?;
    if !out.status.success() {
        bail!("git diff: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    let mut v = Vec::new();
    for rec in out.stdout.split(|&b| b == 0) {
        if rec.is_empty() {
            continue;
        }
        v.push(std::str::from_utf8(rec)?.to_string());
    }
    Ok(v)
}

/// Compute git's blob SHA1 for the given bytes:
/// `sha1(b"blob " + len.to_string() + b"\0" + content)`.
pub fn git_blob_sha1(content: &[u8]) -> String {
    let mut h = Sha1::new();
    let hdr = format!("blob {}\0", content.len());
    h.update(hdr.as_bytes());
    h.update(content);
    let digest = h.finalize();
    hex(&digest)
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

/// Extensions / filenames we treat as "likely textual code-or-docs".
/// Skip common machine-generated / huge / not-useful-to-grep paths.
fn is_excluded(path: &str) -> bool {
    const SUBSTR: &[&str] = &[
        "/node_modules/", "/vendor/", "/target/", "/dist/", "/build/",
        "/.git/", "/.next/", "/.venv/", "/__pycache__/",
        ".min.js", ".min.css", ".map", ".lock",
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "go.sum", "Cargo.lock", "poetry.lock", "uv.lock", "composer.lock",
    ];
    SUBSTR.iter().any(|p| path.contains(p))
}

pub fn looks_textual(path: &str) -> bool {
    if is_excluded(path) {
        return false;
    }
    let lower = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    const NO_EXT_NAMES: &[&str] = &[
        "makefile", "dockerfile", "jenkinsfile", "readme", "license",
        "changelog", "authors", "contributors", "copying", "notice",
    ];
    if NO_EXT_NAMES.contains(&lower.as_str()) {
        return true;
    }
    let Some(dot) = lower.rfind('.') else { return false };
    let ext = &lower[dot + 1..];
    matches!(ext,
        "py" | "pyx" | "pyi" | "ipynb" |
        "js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" |
        "go" | "rs" | "java" | "kt" | "kts" | "scala" | "swift" | "m" | "mm" |
        "c" | "cc" | "cpp" | "cxx" | "h" | "hh" | "hpp" | "hxx" |
        "rb" | "php" | "cs" | "fs" | "fsx" |
        "sh" | "bash" | "zsh" | "fish" | "ps1" |
        "sql" | "proto" | "thrift" | "graphql" |
        "yaml" | "yml" | "toml" | "ini" | "cfg" | "json" | "jsonc" |
        "md" | "mdx" | "rst" | "txt" | "adoc" |
        "html" | "htm" | "xml" | "svg" | "css" | "scss" | "sass" | "less" |
        "dockerfile" | "makefile" | "mk" | "cmake" |
        "tf" | "hcl" | "nix" | "lua" | "r" | "jl" | "hs" | "elm" |
        "ex" | "exs" | "erl" | "clj" | "cljs" | "edn"
    )
}
