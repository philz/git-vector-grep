//! Embedding storage as a git notes ref.
//!
//! Layout under `refs/notes/vector-grep/<model-short-id>`:
//!
//!   <2hex>/<rest38>            payload, attached to the *source* blob OID
//!
//! This is exactly the conventional `git notes` tree layout (default 2/38
//! fanout): each note is a regular blob whose tree path is the SHA1 of the
//! object it annotates. So:
//!
//!   git notes --ref=refs/notes/vector-grep/minilm list
//!   git notes --ref=refs/notes/vector-grep/minilm show <blob_sha>
//!
//! ...work out of the box. `git notes ... merge --strategy=union` is our
//! conflict resolution for two clients that indexed disjoint blobs.
//!
//! There is intentionally NO `meta.json`. The schema (magic + version + dim)
//! is in every payload's header; the model is in the ref name. Switching
//! models doesn't invalidate anything because each model has its own ref.
//!
//! Each note payload is:
//!
//!   magic   : [u8;4]    = b"VGRP"
//!   version : u16 LE    = 1
//!   dim     : u16 LE
//!   n       : u32 LE    (number of chunks)
//!   ranges  : [u32;2*n] LE  (start_line, end_line per chunk)
//!   vecs    : [f32; n*dim] LE  (L2-normalized)
//!
//! Writes go through `git fast-import`, inheriting the previous notes tree and
//! adding only new payloads; reads go through `git cat-file --batch`.

use anyhow::{bail, Context, Result};
use bytemuck::{cast_slice, cast_slice_mut};
use fs2::FileExt;
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub const NOTES_REF_PREFIX: &str = "refs/notes/vector-grep";
pub const SCHEMA: u16 = 1;

/// Build the full notes ref name for a model short id, e.g. "minilm".
pub fn ref_for(short_id: &str) -> String {
    format!("{}/{}", NOTES_REF_PREFIX, short_id)
}

/// In-memory representation of one model's notes ref.
pub struct Store {
    pub repo: PathBuf,
    pub short_id: String,
    pub ref_name: String,
    pub dim: usize,
    /// Pending new note payloads, keyed by source blob_sha. Flushed by `commit()`.
    pub blob_payloads: HashMap<String, Vec<u8>>,
    /// Invalid note paths to remove on the next commit.
    deleted_shas: HashSet<String>,
    /// True if anything mutated.
    pub dirty: bool,
}

impl Store {
    pub fn open(repo: &Path, short_id: &str, dim: usize) -> Result<Self> {
        Ok(Store {
            repo: repo.to_path_buf(),
            short_id: short_id.to_string(),
            ref_name: ref_for(short_id),
            dim,
            blob_payloads: HashMap::new(),
            deleted_shas: HashSet::new(),
            dirty: false,
        })
    }

    /// Source blob SHAs currently annotated by this notes ref.
    pub fn known_blob_shas(&self) -> Result<HashSet<String>> {
        let out = run_git(&self.repo, &["ls-tree", "-r", &self.ref_name])?;
        let mut set = HashSet::new();
        if out.is_empty() {
            return Ok(set);
        }
        for line in std::str::from_utf8(&out)?.lines() {
            let (_meta, path) = match line.split_once('\t') {
                Some(p) => p,
                None => continue,
            };
            if let Some(sha) = sha_from_note_path(path) {
                set.insert(sha);
            }
        }
        Ok(set)
    }

    /// Stream payloads for a selected set of source blob SHAs, in SHA order.
    /// Missing or incompatible notes are skipped.
    pub fn payloads_for(&self, selected: &HashSet<String>) -> Result<Vec<(String, Vec<u8>)>> {
        let mut shas: Vec<String> = selected.iter().cloned().collect();
        shas.sort();
        let specs: Vec<String> = shas
            .iter()
            .map(|s| format!("{}:{}/{}", self.ref_name, &s[..2], &s[2..]))
            .collect();
        let cat = batch_cat_blobs(&self.repo, specs)?;
        let mut out = Vec::with_capacity(shas.len());
        for (sha, bytes) in shas.into_iter().zip(cat) {
            if let Some(b) = bytes {
                if payload_valid(&b, self.dim) {
                    out.push((sha, b));
                }
            }
        }
        Ok(out)
    }

    /// Stream every payload currently in this notes ref, in blob SHA order.
    pub fn iter_all_payloads(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let shas = self.known_blob_shas()?;
        self.payloads_for(&shas)
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

    /// Kept for source compatibility with pre-append-only callers.
    #[deprecated(note = "the cache is append-only; references are no longer pruned")]
    pub fn set_referenced(&mut self, _set: HashSet<String>) {}

    pub fn remove_blobs(&mut self, shas: impl IntoIterator<Item = String>) {
        for sha in shas {
            self.blob_payloads.remove(&sha);
            self.deleted_shas.insert(sha);
            self.dirty = true;
        }
    }

    /// Materialize a new commit on `self.ref_name`, linear history.
    pub fn commit(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        // Worktrees share the notes ref through their common git directory.
        // Serialize ref snapshots so concurrent checkpoints cannot lose notes.
        let _lock = self.commit_lock()?;
        let parent = self.rev_parse(&self.ref_name)?;

        let mut fi = Command::new("git")
            .arg("-C")
            .arg(&self.repo)
            .args(["fast-import", "--quiet", "--force", "--done", "--date-format=raw"])
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

        let mut entries: Vec<(String, String)> = Vec::new();
        for (sha, bytes) in &self.blob_payloads {
            let mark = mk();
            write_blob_with_mark(stdin, &mark, bytes)?;
            entries.push((note_subpath(sha), mark));
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        writeln!(stdin, "commit {}", self.ref_name)?;
        writeln!(stdin, "committer git-vector-grep <vgrep@local> {} +0000", now)?;
        let msg = format!("git-vector-grep snapshot ({})\n", self.short_id);
        writeln!(stdin, "data {}", msg.len())?;
        stdin.write_all(msg.as_bytes())?;
        if let Some(p) = parent {
            writeln!(stdin, "from {}", p)?;
        }
        for sha in &self.deleted_shas {
            writeln!(stdin, "D {}", note_subpath(sha))?;
        }
        for (path, mark) in &entries {
            writeln!(stdin, "M 100644 {} {}", mark, path)?;
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
        self.blob_payloads.clear();
        self.deleted_shas.clear();
        self.dirty = false;
        Ok(())
    }

    fn commit_lock(&self) -> Result<File> {
        let out = Command::new("git")
            .arg("-C")
            .arg(&self.repo)
            .args(["rev-parse", "--git-common-dir"])
            .output()
            .context("find git common directory")?;
        if !out.status.success() {
            bail!(
                "git rev-parse --git-common-dir failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        let common = PathBuf::from(String::from_utf8(out.stdout)?.trim());
        let common = if common.is_absolute() { common } else { self.repo.join(common) };
        let path = common.join(format!("vector-grep-{}.lock", self.short_id));
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("open index lock {}", path.display()))?;
        file.lock_exclusive()
            .with_context(|| format!("lock index cache {}", path.display()))?;
        Ok(file)
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
}

fn run_git(repo: &Path, args: &[&str]) -> Result<Vec<u8>> {
    let out = Command::new("git").arg("-C").arg(repo).args(args).output()?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        if err.contains("unknown revision")
            || err.contains("Not a valid object")
            || err.contains("does not exist")
        {
            return Ok(Vec::new());
        }
        bail!("git {:?} failed: {}", args, err);
    }
    Ok(out.stdout)
}

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
    let read_result = (|| -> Result<Vec<Option<Vec<u8>>>> {
        let mut out = Vec::with_capacity(specs.len());
        for _ in 0..specs.len() {
            let mut header = String::new();
            if reader.read_line(&mut header)? == 0 {
                bail!("git cat-file ended before returning every payload");
            }
            let header = header.trim_end();
            if header.ends_with(" missing") {
                out.push(None);
                continue;
            }
            let mut it = header.split_whitespace();
            let _oid = it.next().context("git cat-file response missing object id")?;
            let ty = it.next().context("git cat-file response missing object type")?;
            if ty != "blob" {
                bail!("git cat-file returned unexpected object type: {}", ty);
            }
            let sz: usize = it
                .next()
                .context("git cat-file response missing object size")?
                .parse()
                .context("git cat-file returned invalid object size")?;
            let mut buf = vec![0u8; sz];
            reader.read_exact(&mut buf)?;
            let mut nl = [0u8; 1];
            reader.read_exact(&mut nl)?;
            if nl[0] != b'\n' {
                bail!("git cat-file payload missing terminator");
            }
            out.push(Some(buf));
        }
        Ok(out)
    })();
    drop(reader);
    let sender_result = sender.join();
    let status = child.wait_with_output()?;
    match sender_result {
        Ok(result) => result.context("send git cat-file requests")?,
        Err(_) => bail!("git cat-file request thread panicked"),
    }
    if !status.status.success() {
        bail!(
            "git cat-file failed: {}",
            String::from_utf8_lossy(&status.stderr)
        );
    }
    read_result
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

fn note_subpath(blob_sha: &str) -> String {
    format!("{}/{}", &blob_sha[..2], &blob_sha[2..])
}

fn sha_from_note_path(p: &str) -> Option<String> {
    // Default notes fanout: "<2hex>/<38hex>". We accept that exactly.
    let (a, b) = p.split_once('/')?;
    if a.len() != 2 || b.len() != 38 {
        return None;
    }
    if !a.chars().chain(b.chars()).all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
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

fn payload_valid(bytes: &[u8], expected_dim: usize) -> bool {
    if bytes.len() < 12 || &bytes[..4] != b"VGRP" {
        return false;
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    let dim = u16::from_le_bytes([bytes[6], bytes[7]]) as usize;
    let n = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let Some(expected_len) = n
        .checked_mul(8 + expected_dim.saturating_mul(4))
        .and_then(|payload| payload.checked_add(12))
    else {
        return false;
    };
    version == SCHEMA && dim == expected_dim && bytes.len() == expected_len
}

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
    if bytes.len() != 12 + ranges_bytes + vec_bytes {
        bail!("payload length mismatch");
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
    if bytes.len() < 12 || &bytes[..4] != b"VGRP" {
        return None;
    }
    Some(u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]))
}
