//! SQLite-backed embedding cache, keyed by git blob SHA.
//!
//! Two tables:
//!   files(path PK, blob_sha, mtime_ns, size, n_chunks)
//!   chunks(blob_sha, idx, start_line, end_line, vec BLOB) -- PK (blob_sha, idx)
//!
//! All vectors are float32 little-endian, length == dim.
//! The cache stores a meta row that records (model_id, dim, schema_version)
//! and self-wipes on mismatch.

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashMap;
use std::path::Path;

pub const SCHEMA_VERSION: i64 = 1;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
  path        TEXT PRIMARY KEY,
  blob_sha    TEXT NOT NULL,
  mtime_ns    INTEGER NOT NULL,
  size        INTEGER NOT NULL,
  n_chunks    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS files_blob_idx ON files(blob_sha);
CREATE TABLE IF NOT EXISTS chunks (
  blob_sha    TEXT NOT NULL,
  idx         INTEGER NOT NULL,
  start_line  INTEGER NOT NULL,
  end_line    INTEGER NOT NULL,
  vec         BLOB NOT NULL,
  PRIMARY KEY (blob_sha, idx)
);
"#;

pub struct Cache {
    pub conn: Connection,
    pub model_id: String,
    pub dim: usize,
}

#[derive(Debug, Clone)]
pub struct FileRow {
    pub blob_sha: String,
    pub mtime_ns: i64,
    pub size: i64,
    pub n_chunks: i64,
}

impl Cache {
    pub fn open(path: &Path, model_id: &str, dim: usize) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let conn = Connection::open(path).context("open sqlite cache")?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA temp_store=MEMORY;
             PRAGMA mmap_size=268435456;",
        )?;
        conn.execute_batch(SCHEMA)?;
        let mut c = Cache {
            conn,
            model_id: model_id.to_string(),
            dim,
        };
        c.check_meta()?;
        Ok(c)
    }

    fn check_meta(&mut self) -> Result<()> {
        let mut stmt = self.conn.prepare("SELECT key, value FROM meta")?;
        let rows: HashMap<String, String> = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
            .collect::<Result<_, _>>()?;
        drop(stmt);
        let want = [
            ("model_id", self.model_id.clone()),
            ("dim", self.dim.to_string()),
            ("schema", SCHEMA_VERSION.to_string()),
        ];
        if rows.is_empty() {
            for (k, v) in &want {
                self.conn.execute(
                    "INSERT INTO meta(key,value) VALUES(?,?)",
                    params![k, v],
                )?;
            }
            return Ok(());
        }
        let mismatch = want.iter().any(|(k, v)| rows.get(*k) != Some(v));
        if mismatch {
            self.conn.execute_batch(
                "DELETE FROM chunks; DELETE FROM files; DELETE FROM meta;",
            )?;
            for (k, v) in &want {
                self.conn.execute(
                    "INSERT INTO meta(key,value) VALUES(?,?)",
                    params![k, v],
                )?;
            }
        }
        Ok(())
    }

    pub fn load_files(&self) -> Result<HashMap<String, FileRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT path, blob_sha, mtime_ns, size, n_chunks FROM files",
        )?;
        let mut out = HashMap::new();
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            let path: String = r.get(0)?;
            out.insert(
                path,
                FileRow {
                    blob_sha: r.get(1)?,
                    mtime_ns: r.get(2)?,
                    size: r.get(3)?,
                    n_chunks: r.get(4)?,
                },
            );
        }
        Ok(out)
    }

    /// Set of blob_sha values that already have at least one chunk vector.
    pub fn known_blob_shas(&self) -> Result<std::collections::HashSet<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT blob_sha FROM chunks")?;
        let mut set = std::collections::HashSet::new();
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            set.insert(r.get::<_, String>(0)?);
        }
        Ok(set)
    }

    pub fn chunk_count_for(&self, blob_sha: &str) -> Result<i64> {
        let n: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM chunks WHERE blob_sha=?",
                params![blob_sha],
                |r| r.get(0),
            )
            .optional()?
            .unwrap_or(0);
        Ok(n)
    }

    pub fn upsert_file(
        &mut self,
        path: &str,
        blob_sha: &str,
        mtime_ns: i64,
        size: i64,
        n_chunks: i64,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO files(path, blob_sha, mtime_ns, size, n_chunks)
             VALUES (?,?,?,?,?)
             ON CONFLICT(path) DO UPDATE SET
               blob_sha=excluded.blob_sha,
               mtime_ns=excluded.mtime_ns,
               size=excluded.size,
               n_chunks=excluded.n_chunks",
            params![path, blob_sha, mtime_ns, size, n_chunks],
        )?;
        Ok(())
    }

    pub fn delete_files(&mut self, paths: &[String]) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare("DELETE FROM files WHERE path=?")?;
            for p in paths {
                stmt.execute(params![p])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Insert chunks for a given blob_sha. `vecs` is dim*n_chunks long.
    pub fn insert_chunks(
        &mut self,
        blob_sha: &str,
        chunks: &[(u32, u32)], // (start_line, end_line)
        vecs: &[f32],
        dim: usize,
    ) -> Result<()> {
        debug_assert_eq!(vecs.len(), chunks.len() * dim);
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO chunks(blob_sha, idx, start_line, end_line, vec)
                 VALUES (?,?,?,?,?)",
            )?;
            for (i, (s, e)) in chunks.iter().enumerate() {
                let v = &vecs[i * dim..(i + 1) * dim];
                let bytes: &[u8] = bytemuck::cast_slice(v);
                stmt.execute(params![blob_sha, i as i64, *s as i64, *e as i64, bytes])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn prune_orphans(&mut self) -> Result<i64> {
        let n = self.conn.execute(
            "DELETE FROM chunks WHERE blob_sha NOT IN (SELECT DISTINCT blob_sha FROM files)",
            [],
        )? as i64;
        Ok(n)
    }

    pub fn db_size(path: &Path) -> u64 {
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    }
}
