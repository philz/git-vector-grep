# git-vector-grep

A fast, cache-friendly **semantic grep** for any git repository. Single
static-ish Rust binary; CPU-only; no GPU, no daemon, no external service.

It indexes the current state of a git repo into a per-repo SQLite cache
(`.git/vector-grep/index.sqlite`), embeds chunks with an ONNX text-embedding
model via [`fastembed-rs`](https://github.com/Anush008/fastembed-rs), and
performs cosine top-k search against an in-memory float32 matrix.

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
