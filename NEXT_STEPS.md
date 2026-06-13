# git-vector-grep -- next steps

Ordered by ROI, not by ease.

## 1. Drop the `files/` manifest entirely

**Why.** Every entry in `files/<sha1(path)>.json` is redundant with what
`git ls-files -s` already prints in ~14 ms. The manifest is the source of
branch-switch churn (it has to be rewritten when the path set changes
even if no content moved), the source of two extra subtrees in the
commit, and the only thing standing between us and a cache that's
purely a function of the multiset of source blob SHAs.

**How.**

- Delete `Store::files`, `FileRow`, all `upsert_file` / `delete_files`
  call sites, and the manifest-write loop in `commit()`.
- Replace `Store::load_files()` with `repo::list_tracked_with_modified()`
  -- a single helper that combines `git ls-files -s` + `git diff
  --name-only` and yields `(path, blob_sha)` pairs straight from git.
- Indexer becomes: target = `set(blob_sha for tracked path)`; on-disk =
  `set(git ls-tree -r REF blobs/)`; embed (target - on_disk); commit a
  tree that contains exactly target's blobs. Orphans drop
  automatically.
- Search builds `blob_sha -> Vec<path>` from `git ls-files -s` on every
  invocation (~14 ms on exe.git). One winning chunk → one hit per path
  in the reverse map (or one hit with a `paths: [...]` list -- display
  choice).
- `stats` learns to compute file count by running `git ls-files -s`
  instead of reading the manifest.

**Bonus that falls out.** The cache becomes mergeable as a set union:
two `refs/vector-grep/index` snapshots can be combined by union-ing
their `blobs/` trees, which makes the "two devs push at once" problem
trivial (see item 2).

---

## 2. Linear history + push/pull/gc subcommands

**Why.** Today `Store::commit()` writes an orphan root commit every
time, so every `git push` is non-fast-forward. We also offer no
ergonomic way to share or refresh the cache across machines, even
though that's the whole point of stashing it in a ref.

**How.**

- In `Store::commit()`, change the fast-import stream to emit
  `from refs/vector-grep/index^0` whenever the ref exists. Drop the
  unconditional `deleteall`. The new commit becomes a child of the old.
- Add subcommands:
  - `git-vector-grep push [--remote ORIGIN] [--force-with-lease]`
    → `git push <remote> refs/vector-grep/index:refs/vector-grep/index`.
    Default to `--force-with-lease` because the cache is reproducible
    and last-writer-wins is fine.
  - `git-vector-grep pull [--remote ORIGIN]`
    → `git fetch <remote> '+refs/vector-grep/*:refs/vector-grep/*'`.
  - `git-vector-grep gc`
    → rewrite the ref to a single orphan root commit at HEAD's tree,
      then suggest `git gc --prune=now`. Trims accumulated history.
  - `git-vector-grep config-remote [--remote ORIGIN]`
    → install `+refs/vector-grep/*:refs/vector-grep/*` into the
      remote's fetch refspec so plain `git fetch` picks us up. One-time
      setup.
- Document server-side caveats in the README (branch-protection rules
  that reject non-`refs/heads/*` writes; don't put the cache under LFS).

---

## 3. Speed up search by avoiding `git cat-file --batch` on every query

**Why.** On exe.git, search end-to-end is ~1.5 s; ~1.0 s of that is
streaming all 73k vector payloads out of pack via
`git cat-file --batch`. The actual cosine scan is ~3 ms. We're paying
the full load cost on every invocation even when nothing changed.

**How (pick one).**

a. **Materialize a flat shadow file under `.git/vector-grep/matrix.bin`.**
   On every index commit, after writing the ref, also write a single
   contiguous `[u32 n_chunks][f32 dim*n]` matrix plus a parallel array
   of `(blob_sha, chunk_idx, start_line, end_line)`. Stamp it with the
   ref's commit OID. On search, mmap and dot-product directly. If the
   ref OID matches the shadow's stamp, skip the rebuild; otherwise
   regenerate from `git cat-file`. Expected: ~50 ms cold search. **This
   is the right answer.**

b. Concatenate all payloads into a single git blob `matrix.bin`
   alongside `blobs/`, indexed by an offset table. One cat-file call
   loads everything as a contiguous slab. Cleaner than (a) because the
   shadow is also shareable, but loses per-blob dedup in the pack.

c. Long-running daemon. (User has explicitly nixed this.)

Prefer (a). It's a couple hundred lines and turns search into a
`memmap2::Mmap` + a dot product.

---

## 4. Bound cold-index memory more tightly

**Why.** We're sized against "big enough VM to spawn N independent ORT
sessions," which is wasteful. On the 8-CPU / 8 GB box we get 8 workers
but have to run at `batch_size=8` to fit, and even then peak RSS is
5.7 GB. On a 4 GB box the current defaults would OOM.

**How (cheapest first).**

- **Recreate ORT sessions periodically.** Each session's activation
  arena grows monotonically as it sees longer sequences. Drop and
  rebuild the session every N groups (e.g. N=20) to release memory.
  ORT session creation takes ~200 ms; amortizes fine across thousands
  of chunks.
- **Probe available RAM, not total RAM.** `MemAvailable` from
  `/proc/meminfo` is what we should be dividing by, not `MemTotal`.
  Currently we leave 1.5 GB headroom which is too aggressive on big
  boxes and not aggressive enough on shared ones.
- **Dynamic batch-size selection.** Start at the default; if a flush
  takes >2× the moving average, halve `batch_size`. Conservative
  upgrade if the average sequence length is short.
- **Reuse model weights across workers.** All sessions load the same
  ~85 MB of MiniLM weights independently. `ort` 2.x supports shared
  prepacked weights via `SessionOptions::add_external_initializer_*`;
  not exposed by `fastembed` today, but worth a small upstream PR.

---

## 5. Search quality lever for codebases

**Why.** MiniLM-L6 is trained on English sentence pairs and treats
code like prose. Code-aware models (`jina-code`, CodeSage) do meaningfully
better on identifier / API-shape queries.

**How.**

- Today `--model jina-code` exists but is too slow on small VMs. With
  the parallel-worker model from the latest changes, jina-code at
  workers=2 on Zen 4 may already be tolerable -- benchmark.
- Consider a *re-ranker* pattern: top-50 via MiniLM (fast), re-rank
  with jina-code only on those 50. Best of both. Implementation lives
  entirely in `search.rs`; doesn't touch storage.
- Eventually: AST-aware chunking via `tree-sitter` so chunks respect
  function boundaries. Easy to add behind a feature flag; the cache
  key doesn't change because we already hash by source blob SHA, but
  the chunker version should be stamped into `meta.json` so a
  chunker upgrade triggers a rebuild.

---

## 6. Cleanups

- The `Embedder` struct still has a `Mutex<TextEmbedding>` per worker.
  Each worker only ever touches its own mutex (rayon's scheduling
  guarantees no aliasing under our shard scheme), so we never contend,
  but the Mutex is there to satisfy `Sync`. Could be cleaner with
  `thread_local!` sessions or by handing each rayon worker its session
  by index without the Mutex.
- `Hit::blob_sha` and `Hit::chunk_idx` are unused; either expose them
  in `--json` output or drop them. (Tied to item 1: blob_sha becomes
  the primary key in the path-less design.)
- `embedder::DEFAULT_MODEL_ID` and `DEFAULT_DIM` constants are dead;
  delete them (clap defaults live in `main.rs`).
- `unused_imports` warnings on `Path` and `Child` -- trivial.
- `--workers` default uses `available_parallelism()`, which on
  cgroup-constrained VMs returns the wrong answer if `cpu.max` is
  configured. Read `/sys/fs/cgroup/cpu.max` when present.
- Concurrent-indexer safety: two `git-vector-grep index` runs in the
  same repo race on the ref. Add a `flock` on
  `.git/vector-grep.lock` for the duration of `Store::commit()`.

---

## 7. Documentation / packaging

- A pre-commit hook example for keeping the cache fresh
  (`git-vector-grep index --quiet` in `.git/hooks/post-commit`).
- A GitHub Actions snippet that runs `git-vector-grep index` and then
  `git push` to a sibling branch for downstream consumers.
- Single-file install script that grabs the right release asset for
  the platform and drops the binary into `~/.local/bin`.
- Man page (`clap_mangen`).
