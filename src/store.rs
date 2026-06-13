//! Embedding storage backed by a hidden git ref.
//!
//! Layout under `refs/vector-grep/index`:
//!
//!   meta.json                    {"model_id":"...","dim":N,"schema":1}
//!   blobs/<2hex>/<rest>.bin      content-addressed payload
//!
//! The `blobs/<sha>.bin` file is the cache *value* for a given git blob SHA
//! (which we already use as the cache key). Storing it under that SHA gives
//! us free renames, free branch-switches, dedup of identical files, and
//! `git gc` packs everything.
//!
//! There is intentionally NO `files/` manifest. The mapping from path to
//! blob SHA lives in git's own index (`git ls-files -s`) and is rebuilt on
//! every invocation in ~14 ms; storing it twice would only cause
//! branch-switch churn.
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

use anyhow::{bail, Context, Result};
use bytemuck::{cast_slice, cast_slice_mut};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub const REF_NAME: &str = "refs/vector-grep/index";
pub const SCHEMA: u16 = 1;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Meta {
    pub model_id: String,
    pub dim: usize,
    pub schema: u16,
}

/// In-memory representation of the embedding ref. We load whichever payloads
/// the caller asks for and buffer any new payloads to write, then materialize
/// a new tree in `commit()`.
pub struct Store {
    pub repo: PathBuf,
    pub meta: Meta,
    /// Cached in-memory copies of payloads we've already fetched (lazy).
    payload_cache: HashMap<String, Vec<u8>>,
    /// New payloads produced this run (keyed by blob_sha). Written to git
    /// at commit time and shadow these out of the on-disk view.
    pub blob_payloads: HashMap<String, Vec<u8>>,
    /// Blob SHAs that should survive the next `commit()`. Anything in the
    /// existing tree that isn't in here is pruned as an orphan. Defaults to
    /// "keep everything" (caller didn't set it) so callers that only want
    /// to read can't accidentally truncate the cache.
    referenced: Option<HashSet<String>>,
    /// True if anything mutated (meta replaced, payloads added, or
    /// referenced-set explicitly changed the on-disk view).
    pub dirty: bool,
    /// True if the existing ref was found with a different model/dim. The
    /// next commit will drop every existing blob.
    stale: bool,
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
            payload_cache: HashMap::new(),
            blob_payloads: HashMap::new(),
            referenced: None,
            dirty: false,
            stale: false,
        };
        let head = s.rev_parse(REF_NAME)?;
        let Some(_head_sha) = head else {
            return Ok(s);
        };
        match s.cat_blob(&format!("{}:meta.json", REF_NAME))? {
            Some(bytes) => {
                let m: Meta = serde_json::from_slice(&bytes)
                    .context("parsing meta.json from ref")?;
                if m.model_id != model_id || m.dim != dim || m.schema != SCHEMA {
                    s.stale = true;
                    s.dirty = true;
                }
            }
            None => {
                s.stale = true;
                s.dirty = true;
            }
        }
        Ok(s)
    }

    /// All `blob_sha` values currently stored in `blobs/` on the ref.
    /// Returns an empty set if the ref was opened with a stale meta.
    pub fn known_blob_shas(&self) -> Result<HashSet<String>> {
        if self.stale {
            return Ok(HashSet::new());
        }
        let out = run_git(&self.repo, &["ls-tree", "-r", REF_NAME, "blobs/"])?;
        let mut set = HashSet::new();
        if out.is_empty() {
            return Ok(set);
        }
        for line in std::str::from_utf8(&out)?.lines() {
            let (_meta, path) = match line.split_once('\t') {
                Some(p) => p,
                None => continue,
            };
            if let Some(sha) = sha_from_blob_path(path) {
                set.insert(sha);
            }
        }
        Ok(set)
    }

    /// Load the payload for a given blob_sha as raw bytes (cached in memory).
    /// Looks in the pending-writes buffer first, then on the ref.
    pub fn load_payload(&mut self, blob_sha: &str) -> Result<Option<Vec<u8>>> {
        if let Some(b) = self.blob_payloads.get(blob_sha) {
            return Ok(Some(b.clone()));
        }
        if let Some(b) = self.payload_cache.get(blob_sha) {
            return Ok(Some(b.clone()));
        }
        if self.stale {
            return Ok(None);
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

    /// Stream every (blob_sha, payload_bytes) currently in the store, in
    /// blob_sha order. Pulls everything via a single `git cat-file --batch`.
    pub fn iter_all_payloads(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let mut shas: Vec<String> = self.known_blob_shas()?.into_iter().collect();
        shas.sort();
        let specs: Vec<String> = shas
            .iter()
            .map(|s| format!("{}:blobs/{}/{}.bin", REF_NAME, &s[..2], &s[2..]))
            .collect();
        let cat = batch_cat_blobs(&self.repo, specs)?;
        let mut out = Vec::with_capacity(shas.len());
        for (sha, bytes) in shas.into_iter().zip(cat) {
            if let Some(b) = bytes {
                out.push((sha, b));
            }
        }
        Ok(out)
    }

    /// Buffer a payload to be written on the next commit.
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

    /// Set the *exact* set of blob SHAs that should be present after the
    /// next commit. Anything in the existing tree not in this set is
    /// pruned. The caller MUST include shas that are still in use on disk
    /// but were neither re-embedded nor reused this run.
    pub fn set_referenced(&mut self, set: HashSet<String>) {
        self.referenced = Some(set);
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

    /// Materialize a new tree containing meta.json + blobs/ and update the
    /// ref. The new commit is a child of the previous tip (linear history).
    pub fn commit(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        // Resolve the previous tip before we start (so fast-import's `from`
        // can chain onto it).
        let parent = if !self.stale { self.rev_parse(REF_NAME)? } else { None };

        let mut fi = Command::new("git")
            .arg("-C")
            .arg(&self.repo)
            .args(["fast-import", "--quiet", "--force", "--date-format=raw"])
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("spawn git fast-import")?;
        let stdin = fi.stdin.as_mut().unwrap();

        let mut next_mark: u32 = 0;
        let mut mk = || {
            next_mark += 1;
            format!(":{}", next_mark)
        };
        let meta_bytes = serde_json::to_vec(&self.meta)?;
        let meta_mark = mk();
        write_blob_with_mark(stdin, &meta_mark, &meta_bytes)?;

        // Decide which existing blobs to carry over. If the caller didn't
        // set a referenced-set, keep everything that's already there (safe
        // for read-only/refresh runs).
        let existing_blob_paths = if self.stale {
            Vec::new()
        } else {
            self.existing_blob_tree_paths()?
        };
        let wrote_shas: HashSet<String> = self.blob_payloads.keys().cloned().collect();

        let mut blob_entries: Vec<(String, BlobSource)> = Vec::new();
        for (sha, bytes) in &self.blob_payloads {
            let mark = mk();
            write_blob_with_mark(stdin, &mark, bytes)?;
            blob_entries.push((blob_subpath(sha), BlobSource::Mark(mark)));
        }
        for (path, oid) in existing_blob_paths {
            let Some(sha) = sha_from_blob_path(&path) else { continue };
            if let Some(ref ref_set) = self.referenced {
                if !ref_set.contains(&sha) {
                    continue;
                }
            }
            if wrote_shas.contains(&sha) {
                continue; // overwritten this run
            }
            blob_entries.push((path, BlobSource::Oid(oid)));
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        writeln!(stdin, "commit {}", REF_NAME)?;
        writeln!(stdin, "committer git-vector-grep <vgrep@local> {} +0000", now)?;
        let msg = b"git-vector-grep snapshot\n";
        writeln!(stdin, "data {}", msg.len())?;
        stdin.write_all(msg)?;
        // Chain onto the previous tip (linear history) so `git push` is a
        // fast-forward. If there's no previous tip, the new commit is a root.
        if let Some(p) = parent {
            writeln!(stdin, "from {}", p)?;
        }
        writeln!(stdin, "deleteall")?;
        writeln!(stdin, "M 100644 {} meta.json", meta_mark)?;
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
        self.stale = false;
        Ok(())
    }

    fn existing_blob_tree_paths(&self) -> Result<Vec<(String, String)>> {
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

    pub fn rev_parse(&self, name: &str) -> Result<Option<String>> {
        let out = Command::new("git")
            .arg("-C")
            .arg(&self.repo)
            .args(["rev-parse", "--verify", "--quiet", name])
            .output()?;
        if !out.status.success() {
            return Ok(None);
        }
        let s = String::from_utf8(out.stdout)?.trim().to_string();
        if s.is_empty() { Ok(None) } else { Ok(Some(s)) }
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
        let err = String::from_utf8_lossy(&out.stderr);
        if err.contains("unknown revision") || err.contains("Not a valid object") {
            return Ok(Vec::new());
        }
        bail!("git {:?} failed: {}", args, err);
    }
    Ok(out.stdout)
}

/// Run `git cat-file --batch` for many object specs at once.
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

pub fn peek_n(bytes: &[u8]) -> Option<u32> {
    decode_n(bytes)
}

fn decode_n(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < 12 || &bytes[..4] != b"VGRP" {
        return None;
    }
    Some(u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]))
}
