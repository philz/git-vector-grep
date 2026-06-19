//! On-disk benchmark corpus: a flat list of chunk texts.
//!
//! Format (little-endian):
//!   magic   [u8; 8]  = "VGGCORP1"
//!   n       u64
//!   then `n` records, each: len u32, bytes[len] (UTF-8, not NUL-terminated)
//!
//! Produced by the `corpus` binary from a repo using the *real* chunker, so the
//! workload is identical to what `git-vector-grep index` would embed.

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"VGGCORP1";

pub fn write(path: &Path, texts: &[String]) -> Result<()> {
    let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut w = BufWriter::new(f);
    w.write_all(MAGIC)?;
    w.write_all(&(texts.len() as u64).to_le_bytes())?;
    for t in texts {
        let b = t.as_bytes();
        w.write_all(&(b.len() as u32).to_le_bytes())?;
        w.write_all(b)?;
    }
    w.flush()?;
    Ok(())
}

pub fn read(path: &Path) -> Result<Vec<String>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut r = BufReader::new(f);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).context("read magic")?;
    if &magic != MAGIC {
        bail!("not a corpus file (bad magic): {}", path.display());
    }
    let mut n_buf = [0u8; 8];
    r.read_exact(&mut n_buf)?;
    let n = u64::from_le_bytes(n_buf) as usize;
    let mut out = Vec::with_capacity(n);
    let mut len_buf = [0u8; 4];
    for _ in 0..n {
        r.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        r.read_exact(&mut buf)?;
        out.push(String::from_utf8(buf).context("corpus text not UTF-8")?);
    }
    Ok(out)
}
