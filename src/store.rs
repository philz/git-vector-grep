//! Embedding storage backed by a hidden git ref.
//!
//! Layout under `refs/vector-grep/index`:
//!
//!   meta.json                    {"model_id":"...","dim":N,"schema":1}
//!   blobs/<2hex>/<rest>.bin      content-addressed payload
//!   files/<2hex>/<rest>.json     manifest: which paths map to which blob_sha
//!
//! The `blobs/<sha>.bin` file is the cache *value* for a given git blob SHA
//! (which we already use as the cache key). Storing it under that SHA gives
//! us free renames, free branch-switches, and `git gc` packs everything.
//!
//! `files/` records the (path -> blob_sha, mtime_ns, size, n_chunks) mapping
//! so we can decide what to (re-)embed without reading file bytes. It's one
//! small JSON file per source path; we batch them in a single fast-import
//! stream.
//!
//! Each `<sha>.bin` payload is:
//!
//!   magic   : [u8;4]    = b"VGRP"
//!   version : u16 LE    = 1
//!   dim     : u16 LE
//!   n       : u32 LE    (number of chunks)
//!   ranges  : [u32;2*n] LE  (start_line, end_line per chunk)
//!   vecs    : [f32; n*dim] LE  (L2-normalized)
//!
//! Writes go through `git fast-import`; reads through `git cat-file --batch`.

use anyhow::{anyhow, bail, Context, Result};
use bytemuck::{cast_slice, cast_slice_mut};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

pub const REF_NAME: &str = "refs/vector-grep/index";
pub const SCHEMA: u16 = 1;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Meta {
    pub model_id: String,
    pub dim: usize,
    pub schema: u16,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileRow {
    pub blob_sha: String,
    pub mtime_ns: i64,
    pub size: i64,
    pub n_chunks: i64,
}

/// In-memory representation of the entire embedding ref. Loaded once per
/// invocation; we materialize a new tree at the end if anything changed.
pub struct Store {
    pub repo: PathBuf,
    pub meta: Meta,
    /// path -> FileRow (loaded from `files/`)
    pub files: HashMap<String, FileRow>,
    /// blob_sha -> payload bytes (loaded lazily on demand via load_payload).
    /// We only populate this for blobs we actually need.
    payload_cache: HashMap<String, Vec<u8>>,
    /// For materializing the new tree at write time, we need to know each
    /// path entry's existing git oid so we can keep unchanged entries cheaply.
    /// (Optimization left for later; first pass rewrites the whole tree.)
    pub blob_payloads: HashMap<String, Vec<u8>>,
    /// True if anything mutated (files/, blobs/, meta).
    pub dirty: bool,
}

impl Store {
    /// Load the existing ref (if any) and validate meta against (model_id,dim).
    /// If meta mismatches, the entire ref is treated as empty (will be rewritten).
    pub fn open(repo: &Path, model_id: &str, dim: usize) -> Result<Self> {
        let mut s = Store {
            repo: repo.to_path_buf(),
            meta: Meta {
                model_id: model_id.to_string(),
                dim,
                schema: SCHEMA,
            },
            files: HashMap::new(),
            payload_cache: HashMap::new(),
            blob_payloads: HashMap::new(),
            dirty: false,
        };
        // Resolve the ref. If it doesn't exist, we're done.
        let head = s.rev_parse(REF_NAME)?;
        let Some(_head_sha) = head else {
            return Ok(s);
        };
        // Check meta.
        match s.cat_blob(&format!("{}:meta.json", REF_NAME))? {
            Some(bytes) => {
                let m: Meta = serde_json::from_slice(&bytes)
                    .context("parsing meta.json from ref")?;
                if m.model_id != model_id || m.dim != dim || m.schema != SCHEMA {
                    // Stale ref. Treat as empty; caller will rewrite.
                    s.dirty = true;
                    return Ok(s);
                }
            }
            None => {
                s.dirty = true;
                return Ok(s);
            }
        }
        // Load files/ manifest entries via `git ls-tree -r`.
        let out = run_git(repo, &["ls-tree", "-r", REF_NAME, "files/"])?;
        if !out.is_empty() {
            for line in std::str::from_utf8(&out)?.lines() {
                // "100644 blob <sha>\tfiles/<2>/<rest>.json"
                let (meta_part, path) = match line.split_once('\t') {
                    Some(p) => p,
                    None => continue,
                };
                let mut it = meta_part.split_whitespace();
                let _mode = it.next();
                let kind = it.next();
                let oid = it.next().unwrap_or("");
                if kind != Some("blob") || oid.is_empty() {
                    continue;
                }
                let bytes = match s.cat_blob(oid)? {
                    Some(b) => b,
                    None => continue,
                };
                #[derive(serde::Deserialize)]
                struct OnDisk {
                    path: String,
                    blob_sha: String,
                    mtime_ns: i64,
                    size: i64,
                    n_chunks: i64,
                }
                let od: OnDisk = match serde_json::from_slice(&bytes) {
                    Ok(p) => p,
                    Err(_) => { let _ = path; continue; }
                };
                s.files.insert(od.path, FileRow {
                    blob_sha: od.blob_sha,
                    mtime_ns: od.mtime_ns,
                    size: od.size,
                    n_chunks: od.n_chunks,
                });
            }
        }
        Ok(s)
    }

    /// `path` -> FileRow, like `Cache::load_files`.
    pub fn load_files(&self) -> &HashMap<String, FileRow> {
        &self.files
    }

    /// All `blob_sha` values currently stored in `blobs/`.
    pub fn known_blob_shas(&self) -> Result<std::collections::HashSet<String>> {
        let out = run_git(&self.repo, &["ls-tree", "-r", REF_NAME, "blobs/"])?;
        let mut set = std::collections::HashSet::new();
        if out.is_empty() {
            return Ok(set);
        }
        for line in std::str::from_utf8(&out)?.lines() {
            let (_meta, path) = match line.split_once('\t') {
                Some(p) => p,
                None => continue,
            };
            // blobs/<2hex>/<rest>.bin
            if !path.ends_with(".bin") {
                continue;
            }
            let rest = path.strip_prefix("blobs/").unwrap_or(path);
            let rest = rest.trim_end_matches(".bin");
            let sha = rest.replacen('/', "", 1);
            set.insert(sha);
        }
        Ok(set)
    }

    /// Load the payload for a given blob_sha as raw bytes (cached in memory).
    pub fn load_payload(&mut self, blob_sha: &str) -> Result<Option<Vec<u8>>> {
        if let Some(b) = self.payload_cache.get(blob_sha) {
            return Ok(Some(b.clone()));
        }
        let p = format!("{}:blobs/{}/{}.bin", REF_NAME, &blob_sha[..2], &blob_sha[2..]);
        match self.cat_blob(&p)? {
            Some(b) => {
                self.payload_cache.insert(blob_sha.to_string(), b.clone());
                Ok(Some(b))
            }
            None => Ok(None),
        }
    }

    /// Iterate over every (path, payload_bytes) currently in the store, in
    /// stable order. Used by the in-memory index loader.
    pub fn iter_all_payloads(&self) -> Result<Vec<(String, String, Vec<u8>)>> {
        // First: get (path -> blob_sha) for every file row, stably ordered.
        let mut entries: Vec<(String, String)> = self
            .files
            .iter()
            .map(|(p, r)| (p.clone(), r.blob_sha.clone()))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        // Unique blob shas in stable order.
        let mut wanted: Vec<String> = entries.iter().map(|(_, s)| s.clone()).collect();
        wanted.sort();
        wanted.dedup();

        // Batch-fetch via one `git cat-file --batch` invocation.
        let cat = batch_cat_blobs(
            &self.repo,
            wanted.iter().map(|s| {
                format!("{}:blobs/{}/{}.bin", REF_NAME, &s[..2], &s[2..])
            }).collect(),
        )?;
        let mut sha_to_payload: HashMap<String, Vec<u8>> = HashMap::new();
        for (sha, bytes) in wanted.into_iter().zip(cat) {
            if let Some(b) = bytes {
                sha_to_payload.insert(sha, b);
            }
        }

        // Re-emit per path.
        let mut out = Vec::with_capacity(entries.len());
        for (path, sha) in entries {
            if let Some(b) = sha_to_payload.get(&sha) {
                out.push((path, sha, b.clone()));
            }
        }
        Ok(out)
    }

    /// Mutators -- buffered in memory; flushed by `commit()`.
    pub fn upsert_file(
        &mut self,
        path: &str,
        blob_sha: &str,
        mtime_ns: i64,
        size: i64,
        n_chunks: i64,
    ) {
        self.files.insert(
            path.to_string(),
            FileRow {
                blob_sha: blob_sha.to_string(),
                mtime_ns,
                size,
                n_chunks,
            },
        );
        self.dirty = true;
    }

    pub fn delete_files(&mut self, paths: &[String]) {
        for p in paths {
            self.files.remove(p);
        }
        if !paths.is_empty() {
            self.dirty = true;
        }
    }

    pub fn insert_chunks(
        &mut self,
        blob_sha: &str,
        ranges: &[(u32, u32)],
        vecs: &[f32],
        dim: usize,
    ) {
        let payload = pack_payload(dim, ranges, vecs);
        self.blob_payloads.insert(blob_sha.to_string(), payload);
        self.dirty = true;
    }

    pub fn chunk_count_for(&mut self, blob_sha: &str) -> Result<i64> {
        if let Some(b) = self.blob_payloads.get(blob_sha) {
            return Ok(decode_n(b).unwrap_or(0) as i64);
        }
        match self.load_payload(blob_sha)? {
            Some(b) => Ok(decode_n(&b).unwrap_or(0) as i64),
            None => Ok(0),
        }
    }

    /// Materialize a new tree containing meta.json + files/ + blobs/ and
    /// update the ref. Uses a single `git fast-import` subprocess.
    pub fn commit(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        let mut fi = Command::new("git")
            .arg("-C")
            .arg(&self.repo)
            .args(["fast-import", "--quiet", "--force", "--date-format=raw"])
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("spawn git fast-import")?;
        let stdin = fi.stdin.as_mut().unwrap();

        // Mark allocator. fast-import requires :N where N is a positive integer.
        let mut next_mark: u32 = 0;
        let mut mk = || { next_mark += 1; format!(":{}", next_mark) };
        // Blob: meta.json
        let meta_bytes = serde_json::to_vec(&self.meta)?;
        let meta_mark = mk();
        write_blob_with_mark(stdin, &meta_mark, &meta_bytes)?;

        // Blob: every files/<path-hash>.json
        let mut file_entries: Vec<(String, String)> = Vec::with_capacity(self.files.len());
        for (path, row) in &self.files {
            #[derive(serde::Serialize)]
            struct OnDisk<'a> {
                path: &'a str,
                #[serde(flatten)]
                row: &'a FileRow,
            }
            let json = serde_json::to_vec(&OnDisk { path, row })?;
            let mark = mk();
            write_blob_with_mark(stdin, &mark, &json)?;
            file_entries.push((path_to_files_subpath(path), mark));
        }

        // Blob: every blobs/<sha>.bin
        // We need to keep blobs that exist on the ref but for which we don't
        // have an in-memory payload (untouched this run). Fetch their oids
        // from the existing tree.
        let existing_blob_paths = self.existing_blob_tree_paths()?;
        let mut blob_entries: Vec<(String, BlobSource)> = Vec::new();
        let mut wrote_shas = std::collections::HashSet::new();
        for (sha, bytes) in &self.blob_payloads {
            let mark = mk();
            write_blob_with_mark(stdin, &mark, bytes)?;
            blob_entries.push((blob_subpath(sha), BlobSource::Mark(mark)));
            wrote_shas.insert(sha.clone());
        }
        // Carry over existing blobs we didn't touch, but only for shas that
        // are still referenced by at least one file row (orphan pruning).
        let referenced: std::collections::HashSet<String> = self
            .files
            .values()
            .map(|r| r.blob_sha.clone())
            .collect();
        for (path, oid) in existing_blob_paths {
            // path like "blobs/aa/bb...bin"
            let sha = sha_from_blob_path(&path).unwrap_or_default();
            if sha.is_empty() {
                continue;
            }
            if !referenced.contains(&sha) {
                continue; // prune orphan
            }
            if wrote_shas.contains(&sha) {
                continue; // overwritten this run
            }
            blob_entries.push((path, BlobSource::Oid(oid)));
        }

        // Commit: write a commit pointing to a new tree.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        writeln!(stdin, "commit {}", REF_NAME)?;
        writeln!(stdin, "committer git-vector-grep <vgrep@local> {} +0000", now)?;
        let msg = b"git-vector-grep snapshot\n";
        writeln!(stdin, "data {}", msg.len())?;
        stdin.write_all(msg)?;
        // No `from` -- replace the ref's history entirely each commit. The
        // tree dedupes against pack objects, so this is cheap.
        writeln!(stdin, "deleteall")?;
        writeln!(stdin, "M 100644 {} meta.json", meta_mark)?;
        for (path, mark) in &file_entries {
            writeln!(stdin, "M 100644 {} {}", mark, path)?;
        }
        for (path, src) in &blob_entries {
            match src {
                BlobSource::Mark(m) => writeln!(stdin, "M 100644 {} {}", m, path)?,
                BlobSource::Oid(o) => writeln!(stdin, "M 100644 {} {}", o, path)?,
            }
        }
        writeln!(stdin, "done")?;
        drop(fi.stdin.take());
        let status = fi.wait_with_output()?;
        if !status.status.success() {
            bail!(
                "git fast-import failed: {}",
                String::from_utf8_lossy(&status.stderr)
            );
        }
        self.dirty = false;
        Ok(())
    }

    fn existing_blob_tree_paths(&self) -> Result<Vec<(String, String)>> {
        // (path, oid) for every existing blobs/* entry.
        let head = self.rev_parse(REF_NAME)?;
        if head.is_none() {
            return Ok(Vec::new());
        }
        let out = run_git(&self.repo, &["ls-tree", "-r", REF_NAME, "blobs/"])?;
        let mut v = Vec::new();
        for line in std::str::from_utf8(&out)?.lines() {
            let (meta_part, path) = match line.split_once('\t') {
                Some(p) => p,
                None => continue,
            };
            let mut it = meta_part.split_whitespace();
            let _mode = it.next();
            let kind = it.next();
            let oid = it.next().unwrap_or("");
            if kind == Some("blob") && !oid.is_empty() {
                v.push((path.to_string(), oid.to_string()));
            }
        }
        Ok(v)
    }

    fn rev_parse(&self, name: &str) -> Result<Option<String>> {
        let out = Command::new("git")
            .arg("-C")
            .arg(&self.repo)
            .args(["rev-parse", "--verify", "--quiet", name])
            .output()?;
        if !out.status.success() {
            return Ok(None);
        }
        let s = String::from_utf8(out.stdout)?.trim().to_string();
        if s.is_empty() {
            Ok(None)
        } else {
            Ok(Some(s))
        }
    }

    fn cat_blob(&self, spec: &str) -> Result<Option<Vec<u8>>> {
        let out = Command::new("git")
            .arg("-C")
            .arg(&self.repo)
            .args(["cat-file", "-p", spec])
            .output()?;
        if !out.status.success() {
            return Ok(None);
        }
        Ok(Some(out.stdout))
    }
}

enum BlobSource {
    Mark(String),
    Oid(String),
}

fn run_git(repo: &Path, args: &[&str]) -> Result<Vec<u8>> {
    let out = Command::new("git").arg("-C").arg(repo).args(args).output()?;
    if !out.status.success() {
        // Treat missing ref as empty.
        let err = String::from_utf8_lossy(&out.stderr);
        if err.contains("unknown revision") || err.contains("Not a valid object") {
            return Ok(Vec::new());
        }
        bail!("git {:?} failed: {}", args, err);
    }
    Ok(out.stdout)
}

/// Run `git cat-file --batch` for many object specs at once.
/// Returns Some(bytes) per input or None if the object is missing.
fn batch_cat_blobs(repo: &Path, specs: Vec<String>) -> Result<Vec<Option<Vec<u8>>>> {
    if specs.is_empty() {
        return Ok(Vec::new());
    }
    let mut child = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["cat-file", "--batch"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    // Send all queries.
    let send_specs = specs.clone();
    let sender = std::thread::spawn(move || -> std::io::Result<()> {
        for s in &send_specs {
            stdin.write_all(s.as_bytes())?;
            stdin.write_all(b"\n")?;
        }
        drop(stdin);
        Ok(())
    });
    let mut reader = BufReader::new(stdout);
    let mut out = Vec::with_capacity(specs.len());
    for _ in 0..specs.len() {
        let mut header = String::new();
        reader.read_line(&mut header)?;
        let header = header.trim_end();
        // header is either "<oid> <type> <size>" or "<spec> missing"
        if header.ends_with(" missing") {
            out.push(None);
            continue;
        }
        let mut it = header.split_whitespace();
        let _oid = it.next();
        let _ty = it.next();
        let sz: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
        let mut buf = vec![0u8; sz];
        reader.read_exact(&mut buf)?;
        let mut nl = [0u8; 1];
        let _ = reader.read_exact(&mut nl);
        out.push(Some(buf));
    }
    let _ = sender.join();
    let _ = child.wait();
    Ok(out)
}

fn write_blob_with_mark(
    stdin: &mut std::process::ChildStdin,
    mark: &str,
    bytes: &[u8],
) -> Result<()> {
    writeln!(stdin, "blob")?;
    writeln!(stdin, "mark {}", mark)?;
    writeln!(stdin, "data {}", bytes.len())?;
    stdin.write_all(bytes)?;
    Ok(())
}

fn blob_subpath(blob_sha: &str) -> String {
    format!("blobs/{}/{}.bin", &blob_sha[..2], &blob_sha[2..])
}

fn sha_from_blob_path(p: &str) -> Option<String> {
    let rest = p.strip_prefix("blobs/")?.strip_suffix(".bin")?;
    let (a, b) = rest.split_once('/')?;
    Some(format!("{a}{b}"))
}

/// `path` is the source-file path inside the user's repo. We can't store
/// it verbatim under `files/` because it may contain `..`, leading slashes,
/// or characters that git's tree format dislikes. Instead we name the file
/// by its SHA1, and store the original path inside the JSON.
fn path_to_files_subpath(path: &str) -> String {
    use sha1::Digest;
    let mut h = sha1::Sha1::new();
    h.update(path.as_bytes());
    let d = h.finalize();
    let hex = hex_lower(&d);
    format!("files/{}/{}.json", &hex[..2], &hex[2..])
}

fn hex_lower(b: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(b.len() * 2);
    for &x in b {
        s.push(H[(x >> 4) as usize] as char);
        s.push(H[(x & 0xf) as usize] as char);
    }
    s
}

// ---------- payload encoding ----------

fn pack_payload(dim: usize, ranges: &[(u32, u32)], vecs: &[f32]) -> Vec<u8> {
    let n = ranges.len();
    debug_assert_eq!(vecs.len(), n * dim);
    let mut out = Vec::with_capacity(4 + 2 + 2 + 4 + n * 8 + n * dim * 4);
    out.extend_from_slice(b"VGRP");
    out.extend_from_slice(&(SCHEMA as u16).to_le_bytes());
    out.extend_from_slice(&(dim as u16).to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    for (s, e) in ranges {
        out.extend_from_slice(&s.to_le_bytes());
        out.extend_from_slice(&e.to_le_bytes());
    }
    let vec_bytes: &[u8] = cast_slice(vecs);
    out.extend_from_slice(vec_bytes);
    out
}

/// Returns (dim, ranges, f32 slice).
pub fn unpack_payload(bytes: &[u8]) -> Result<(usize, Vec<(u32, u32)>, Vec<f32>)> {
    if bytes.len() < 12 || &bytes[..4] != b"VGRP" {
        bail!("bad payload magic");
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != SCHEMA {
        bail!("payload schema {} != {}", version, SCHEMA);
    }
    let dim = u16::from_le_bytes([bytes[6], bytes[7]]) as usize;
    let n = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let ranges_bytes = 8 * n;
    let vec_bytes = 4 * dim * n;
    if bytes.len() < 12 + ranges_bytes + vec_bytes {
        bail!("payload truncated");
    }
    let mut ranges = Vec::with_capacity(n);
    for i in 0..n {
        let off = 12 + i * 8;
        let s = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let e = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
        ranges.push((s, e));
    }
    let mut vecs = vec![0f32; n * dim];
    let dst: &mut [u8] = cast_slice_mut(vecs.as_mut_slice());
    let src = &bytes[12 + ranges_bytes..12 + ranges_bytes + vec_bytes];
    dst.copy_from_slice(src);
    Ok((dim, ranges, vecs))
}

pub fn peek_n(bytes: &[u8]) -> Option<u32> { decode_n(bytes) }

fn decode_n(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < 12 || &bytes[..4] != b"VGRP" {
        return None;
    }
    Some(u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]))
}
