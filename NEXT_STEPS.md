# git-vector-grep -- next steps

Ordered by ROI, not by ease.

Retired (already shipped):
- Drop `files/` manifest. Cache is now keyed purely by source blob SHA.
- Linear history + `push`/`pull`/`config-remote`/`gc` subcommands.
- Switch to git notes layout: `refs/notes/vector-grep/<model-short-id>`.
  Drops `meta.json`. Each model gets its own ref (multi-model coexistence).
  `git notes list`, `git notes show`, `git notes merge --strategy=union`
  all work directly. Union-merge solves the two-devs-pushing-disjoint-blobs
  case with zero custom code; `config-remote` enables it automatically.

## 1. Speed up search by avoiding `git cat-file --batch` on every query

**Why.** On exe.git, search end-to-end is ~1.5 s; ~1.0 s of that is
streaming all 73k vector payloads out of pack via
`git cat-file --batch`. The actual cosine scan is ~3 ms. We're paying
the full load cost on every invocation even when nothing changed.

**How.**

Materialize a flat shadow file under `.git/vector-grep/<short_id>.bin`.
On every index commit, after writing the ref, also write a single
contiguous `[u32 n_chunks][f32 dim*n]` matrix plus a parallel array
of `(blob_sha, chunk_idx, start_line, end_line)`. Stamp it with the
ref's commit OID. On search, mmap and dot-product directly. If the
ref OID matches the shadow's stamp, skip the rebuild; otherwise
regenerate from `git cat-file`. Expected: ~50 ms cold search.

---

## 2. Bound cold-index memory more tightly

**Why.** We're sized against "big enough VM to spawn N independent ORT
sessions," which is wasteful. On the 8-CPU / 8 GB box we get 8 workers
but have to run at `batch_size=8` to fit, and even then peak RSS is
5.7 GB. On a 4 GB box the current defaults would OOM.

**How (cheapest first).**

- Recreate ORT sessions periodically. Each session's activation arena
  grows monotonically. Drop and rebuild the session every N groups.
- Probe `MemAvailable` from `/proc/meminfo` instead of `MemTotal`.
- Dynamic batch-size selection: halve `batch_size` on slow flushes.
- Read `/sys/fs/cgroup/cpu.max` to size workers on constrained VMs.
- Reuse model weights across workers (needs an upstream fastembed PR).

---

## 3. Search quality lever for codebases

**Why.** MiniLM-L6 is trained on English sentence pairs and treats
code like prose. Code-aware models (`jina-code`, CodeSage) do meaningfully
better on identifier / API-shape queries.

**How.**

- Re-benchmark `--model jina-code` with the parallel-session model.
- *Re-ranker* pattern: top-50 via MiniLM (fast), re-rank with jina-code
  on those 50. Lives entirely in `search.rs`. Now trivial because the
  ref-per-model design lets both caches coexist.
- AST-aware chunking via `tree-sitter`, behind a feature flag.
  Bumping the `VGRP` magic to `VGR2` (or the chunker version) triggers
  a rebuild without affecting other model refs.

---

## 4. Cleanups

- `Embedder` keeps `Mutex<TextEmbedding>` per worker only to satisfy
  `Sync`; in practice each worker only ever touches its own session.
  Move to `thread_local!` or rayon-by-index without the Mutex.
- Concurrent-indexer safety: two `git-vector-grep index` runs in the
  same repo race on the ref. Add a `flock` on
  `.git/vector-grep-<short_id>.lock` for the duration of `Store::commit()`.
- `--workers` default uses `available_parallelism()`, which on
  cgroup-constrained VMs returns the wrong answer. Read
  `/sys/fs/cgroup/cpu.max` when present.

---

## 5. Documentation / packaging

- README explaining the notes-ref design and how to inspect/share/merge
  the cache with stock git commands.
- A pre-commit hook example for keeping the cache fresh
  (`git-vector-grep index --quiet` in `.git/hooks/post-commit`).
- A GitHub Actions snippet that runs `git-vector-grep index` and then
  `push` for downstream consumers.
- Single-file install script that grabs the right release asset for
  the platform and drops the binary into `~/.local/bin`.
- Man page (`clap_mangen`).
