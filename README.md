# git-vector-grep

Semantic grep over a git repo. Single Rust binary; CPU-only; no daemon, no
sidecar database, no external service. The append-only embedding cache lives
inside your repo as a **git notes ref**, so it ships over `git push` and dedups
across branches, renames, and machines for free.

> ⚠️ **Caveat emptor: this is vibe-coded.** Built end-to-end in a few
> sessions with an LLM driving. It works on the repos I've thrown at it
> (a 4.2k-file Go codebase, a 280-file Python one) but it is not battle
> tested. Read the source before pointing it at anything precious.

## What it does

```
$ git-vector-grep "how do we chunk PDFs" -k 5   # search is the default action
                                                # (indexes on first run)
0.6122  src/arcaneum/indexing/pdf/chunker.py:49-68
0.6061  docs/rdr/RDR-014-markdown-indexing.md:129-148
...
```

## Where the cache lives

Under `refs/notes/vector-grep/<model-short-id>`, with the standard
git-notes 2/38 fanout tree:

```
refs/notes/vector-grep/minilm
├── 73/3d936e948d3d022e94ee4c2172e15fd83e934d    ← payload, attached to
├── fc/7d1ab1d12d9a8fbb5d29a911d2cc47ac385b23       source blob SHA
└── ...                                              (one note per blob)
```

Each note is a packed binary: `VGRP` magic, dim, n_chunks, line ranges,
f32 vectors. The cache key for a payload is **git's own blob SHA1** of
the source file — git already computed it; we just look it up via
`git ls-files -s`.

Consequences:

- **Renames are free.** Same content → same SHA → cache hit.
- **Dedup is free.** Two identical files share one note.
- **Branch switches are free.** The cache retains blob versions seen on other
  branches, while search only loads blobs tracked by the current worktree.
- **Interrupted indexing resumes.** Each bounded embedding group is committed,
  so a later invocation starts from the last completed checkpoint.
- **Sharing is free.** `git push` ships the cache; `git pull` receives it.
- **Inspectable with stock git.** `git notes --ref=refs/notes/vector-grep/minilm list` etc.
- **Mergeable with stock git.** `git notes merge --strategy=union` resolves
  the two-clients-pushing-disjoint-blobs case with zero custom code,
  because embeddings are deterministic per `(model, blob_sha)`.
- **Multi-model coexistence.** Each model gets its own notes ref. Index
  with two models and they live side by side.

## Usage

```
git-vector-grep QUERY -k 10                  # top-k cosine search (default action)
git-vector-grep QUERY --show                 # with snippet text
git-vector-grep QUERY --path src/            # restrict by prefix
git-vector-grep QUERY -q                      # suppress indexing progress
git-vector-grep search QUERY                 # explicit search subcommand (same thing)
git-vector-grep index                        # just (re)build the cache
git-vector-grep stats                        # what's cached

git-vector-grep config-remote --remote origin   # one-time setup
git-vector-grep push --remote origin            # share cache
git-vector-grep pull --remote origin            # receive cache

git-vector-grep gc                              # collapse cache history
git-vector-grep clear                           # drop current model's cache
```

Global flags:

- `--model {minilm,bge-small,bge-small-q,bge-base,jina-code}` — default
  `minilm` (`sentence-transformers/all-MiniLM-L6-v2`, 384-D, ~85 MB ONNX).
- `--workers N` — override CPU/ONNX parallelism. MiniLM and BGE-small variants
  auto-size up to 8 sessions from CPU count and the lower of host or cgroup RAM,
  reserving at least 1.5 GiB and 25% of RAM. Larger `bge-base` and `jina-code`
  models retain the safe one-session default unless explicitly overridden.
  Only one session is loaded for cached searches; additional sessions initialize
  lazily after the cache scan finds chunks that actually need embedding.
- `--batch-size N` — override the embedding batch. Defaults to 1 on CPU/ONNX
  (avoids padding variable-length code chunks) and 16 on MLX.
- `--repo PATH` — operate on a repo other than `$PWD`.

First run of a given model downloads the ONNX weights from HuggingFace
into `~/.cache/git-vector-grep/models/`.

## Build

Rust 1.75+:

```
cargo build --release
# -> ./target/release/git-vector-grep
```

On **non-Apple-Silicon** (Linux, Intel Mac) this is a pure CPU/ONNX binary with
no extra dependencies. Prebuilt Linux x86_64 binaries are attached to each
GitHub release.

### Apple-GPU backend (automatic on Apple Silicon)

On an Apple-Silicon Mac, `cargo build` **automatically** compiles in the GPU
embedding backend ([mlx-rs](https://github.com/oxideai/mlx-rs)) and uses it by
default at runtime — same single Rust binary, faster cold index at full context
in a fraction of the RAM. The one build-time requirement is the Metal Toolchain
(mlx-sys compiles MLX + its metallib):

```
xcodebuild -downloadComponent MetalToolchain   # one-time, ~688 MB
cargo build --release                          # GPU backend included automatically
```

Then it's automatic at runtime:

```
git-vector-grep "how do we chunk PDFs"     # --backend auto (default): uses the GPU
git-vector-grep --backend cpu QUERY        # force CPU/ONNX (no GPU needed)
git-vector-grep --backend mlx QUERY        # force GPU
```

`auto` (the default) uses MLX when the model has an MLX variant (`minilm`,
`bge-small`, `bge-base`), else falls back to CPU/ONNX. MLX caches live under
their own `mlx-*` notes refs, so GPU and CPU indexes coexist and never mix. Each
index prints the active backend and its caps, e.g.
`[embed] mlx/apple-gpu · mlx-minilm · GPU buffer-cache cap 512 MB · ~0.4 GB peak`.
See `bench/RESULTS.md` for the full backend comparison.

## Performance

CPU embedding microbenchmark, August 30, 2026: 8-vCPU AMD EPYC 9554P,
16 GiB RAM, MiniLM, 4,096 production chunks from `exe`, three warmed runs.
The benchmark caps each fastembed call to the production-equivalent 512 total
chunks and reports median wall time.

| CPU configuration | batch | chunks/s | peak RSS |
|-------------------|------:|---------:|---------:|
| old default: 1 session × 8 intra-threads | 16 | 33 | 1.20 GB |
| new default: 8 sessions × 1 intra-thread | 1 | **74** | 1.40 GB |

That is **2.26× throughput** for about 0.20 GB additional peak RSS. A
256-vector comparison was byte-identical between configurations. Every tested
larger CPU batch was slower and consumed more memory on this variable-length
workload. An additional three-run, baseline/candidate-interleaved check while
three unrelated VM processes each consumed about one CPU core still measured a
**2.21× median speedup** (35→77 chunks/s), with median peak RSS of 1.19→1.37 GB.
A post-lazy-loading end-to-end index of 1,024 corpus-derived files measured
30.46s→13.86s (**2.20×**) with median peak RSS of 1.18→1.40 GB, including lazy
worker creation and git-note commits. See `bench/README.md` for reproduction
commands.

Cached-search validation on August 31, 2026 used a fully indexed 128-file repo
with normal auto-index enabled. Loading only the eager session reduced median
startup from 0.50s to 0.22s and median peak RSS from 0.97 GB to 0.17 GB. On the
large `exe` cache with `--no-auto-index`, median wall time fell from 2.69s to
2.38s and median peak RSS from 1.30 GB to 0.45 GB.

Older end-to-end reference, 8-CPU / 8 GB Zen 4 VM, MiniLM, CPU only:

|                           | files | chunks | cold index | incremental | search |
|---------------------------|-------|--------|-----------:|------------:|-------:|
| arcaneum (Python)         |   280 |  6 701 |   2m 20s   |    0.3s     |  0.2s  |
| exe (Go)                  | 4 262 | 69 373 |  26m 36s   |    1.1s     |  1.6s  |

Incremental reindex on a clean repo is dominated by `git ls-files`.
Search is dominated by streaming the vector matrix out of pack via
`git cat-file --batch` (~1.0s of the 1.6s on exe.git); that's the next
thing to fix, see `NEXT_STEPS.md`.

## Design choices the code is opinionated about

- **No daemon.** Each invocation is short-lived.
- **No SQLite, no sidecar files.** Everything lives in git objects.
- **No network at runtime.** All embedding is local via bundled ONNX
  Runtime. The only network call is the one-time model download from
  HuggingFace on first use of each model.
- **No path table.** Paths come from `git ls-files -s` on every search
  (~14 ms on a 5k-file repo). The cache stores the append-only union of
  `(blob_sha → vectors)` entries seen across worktrees, but search reads only
  entries referenced by the current worktree.
- **Stock git everything.** push/pull are `git push`/`git fetch`; gc is
  collapse-then-`git gc`; merge is `git notes merge --strategy=union`.
  We're a thin layer over git's existing notes machinery.

## License

MIT OR Apache-2.0.
