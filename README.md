# git-vector-grep

A fast, cache-friendly **semantic grep** for any git repository. Single
static-ish Rust binary; CPU-only; no GPU, no daemon, no external service.

It indexes the current state of a git repo into a **hidden git ref**
(`refs/vector-grep/index`), embeds chunks with an ONNX text-embedding
model via [`fastembed-rs`](https://github.com/Anush008/fastembed-rs), and
performs cosine top-k search against an in-memory float32 matrix.

No SQLite, no daemon, no scratch directory. The cache *is* a git tree:

```
refs/vector-grep/index
├─ meta.json                 {"model_id":"...","dim":N,"schema":1}
├─ files/<sha1(path)>.json   {"path":..., "blob_sha":..., "n_chunks":...}
└─ blobs/<2hex>/<rest>.bin   packed f32 vectors, keyed by git blob SHA
```

Because the per-content payload is stored under git's blob SHA, you get:

- **Free renames**: same content → same path under `blobs/`.
- **Free dedup**: identical files share one payload, packed once.
- **Free sharing**: `git push origin refs/vector-grep/index` ships embeddings
  to teammates; they pay zero cold-index cost.
- **Free GC**: `git gc` packs vectors with delta + zlib like any other blob.
- **Free invalidation**: change models → bump the schema/model in `meta.json`
  → the ref auto-rebuilds on the next run.

## Backend

Local-only, ONNX via `fastembed-rs`. The default model is
`sentence-transformers/all-MiniLM-L6-v2` (384-D, 22M params, 6 transformer
layers, ~85 MB on disk). 6 layers turned out to win on CPU even against
static-INT8 `BGESmallENV15Q` on Zen 4: fewer layers > VNNI uplift.

Override with `--model {minilm,bge-small,bge-small-q,bge-base,jina-code}`.

## CPU parallelism

For tiny transformer encoders, one ORT session with `intra_threads=8`
scales sub-linearly past ~2 threads. We instead spawn **one ORT session
per worker thread, each pinned to `intra_threads=1`**, and shard work
across workers with rayon. On an 8-core box this is roughly 3-4× the
throughput of the single-session approach.

Worker count defaults to roughly `min(cpus, (RAM_GB - 1.5) / 0.75)` --
model weights and activation arenas grow per session. Override with
`--workers N`. Lower `--batch-size` if you OOM (default 16).

## Why it's fast

- **The cache key is the git blob SHA** that `git ls-files -s` already prints.
  For an unmodified working tree, indexing reads zero file bytes and computes
  zero hashes -- it's basically just SQLite lookups.
- For modified files we recompute the SHA-1 ourselves (it matches what git
  would compute), and **renames + duplicate-content files reuse embeddings
  for free** because the cache is keyed by content, not path.
- The embedding model is downloaded once into `~/.cache/git-vector-grep/`
  and reloaded from disk on each call.
- Top-k cosine is a NumPy-style dense scan parallelized with `rayon`; at
  ~3.4k chunks of dim 384 it returns top-10 in **~0.5 ms**.

## Build

```
cargo build --release
```

The resulting `target/release/git-vector-grep` is a single binary with
ONNX Runtime statically linked.

## Use

```
git-vector-grep index               # bring the cache up to date
git-vector-grep search QUERY...     # auto-reindexes, then searches
git-vector-grep search --no-auto-index QUERY...
git-vector-grep stats
git-vector-grep clear
```

Flags:

- `--model` -- `bge-small` (default), `bge-base`, `jina-code`, `jina-en`,
  `minilm`, `minilm-q`, `bge-small-q`.
- `--threads N` -- ONNX intra-op threads (defaults to all cores).
- `--path PREFIX` -- restrict matches to a subtree.
- `-k N` -- top-k (default 10).
- `--show` -- print the matching chunk text.
- `--json` -- machine-readable output.

The cache **self-invalidates** if the model id or dim changes.

## Design notes

- **Chunker**: 40-line windows, 8-line overlap, max 4 KB / chunk. Plain text;
  binary and oversized files are skipped. No tree-sitter (keeps the binary
  small); the chunker is the obvious upgrade path for code-aware splitting.
- **Schema**: `files(path PK, blob_sha, mtime_ns, size, n_chunks)` and
  `chunks(blob_sha, idx, start_line, end_line, vec BLOB)` with PK
  `(blob_sha, idx)`. Storing vectors under the blob SHA decouples them from
  paths, which is what makes rename detection free.
- **Embeddings** are L2-normalized at write time so search is a plain dot
  product.

## Benchmarks (local ONNX, MiniLM-L6-v2)

**arcaneum** (272 files, 6.6k chunks):

| Phase | 2 CPUs | 8 CPUs (Zen 4) |
|---|---:|---:|
| Cold index | 6 min 12 s | **2 min 20 s** (~2.7×) |
| Incremental | 0.2 s | 0.2 s |
| Search | 0.2 s | 0.2 s |

**exe.git** (4262 textual files, 69k unique chunks, 4169 unique blobs):

| Phase | 2 CPUs | 8 CPUs (Zen 4) |
|---|---:|---:|
| Cold index | 65 min | **26 min 36 s** (~2.5×) |
| Peak RSS during index | 2.5 GB | 5.7 GB |
| Incremental, clean tree | 0.6 s | 1.1 s |
| Search end-to-end | 1.2 s | 1.7 s |
| Pack size added to `.git` | ~114 MB | ~114 MB |

Cold-index speedup is sub-linear vs. CPU count because each ORT session
has its own model weights + activation arena; you can't run as many
workers in parallel as you have cores without OOMing. The current sweet
spot on the 8 GB / 8-CPU box is **8 workers × batch_size=8**.

The ~1 s of "search end-to-end" is dominated by streaming vectors out of
the pack via `git cat-file --batch`; the in-memory dot-product over 73k
vectors is ~3 ms.

## Inspiration

Model choice and overall pipeline ideas borrowed from
[arcaneum](https://github.com/cwensel/arcaneum), which surveys current
code-embedding models thoughtfully. Differences:

- arcaneum uses Qdrant; we use SQLite (single binary, no daemon).
- arcaneum's code cache is at *commit* granularity; ours is at *content*
  granularity, which is much better for interactive use as the working tree
  changes.
- arcaneum reads files to hash them; we lean on `git ls-files -s` so an
  unchanged repo costs zero file I/O.
