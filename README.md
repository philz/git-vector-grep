# git-vector-grep

Semantic grep over a git repo. Single Rust binary; CPU-only; no daemon, no
sidecar database, no external service. The embedding cache lives inside
your repo as a **git notes ref**, so it ships over `git push` and dedups
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
- **Branch switches are free.** Indexing a feature branch off main only
  embeds the blobs unique to the branch.
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
- `--workers N` — parallel ONNX sessions (default: auto-sized by RAM).
- `--repo PATH` — operate on a repo other than `$PWD`.

First run of a given model downloads the ONNX weights from HuggingFace
into `~/.cache/git-vector-grep/models/`.

## Build

Rust 1.75+:

```
cargo build --release
# -> ./target/release/git-vector-grep (~28 MB; bundles ONNX Runtime)
```

Prebuilt Linux x86_64 binaries are attached to each GitHub release.

## Performance

8-CPU / 8 GB Zen 4 VM, MiniLM, CPU only:

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
  (~14 ms on a 5k-file repo). The cache only stores `(blob_sha → vectors)`.
- **Stock git everything.** push/pull are `git push`/`git fetch`; gc is
  collapse-then-`git gc`; merge is `git notes merge --strategy=union`.
  We're a thin layer over git's existing notes machinery.

## License

MIT OR Apache-2.0.
