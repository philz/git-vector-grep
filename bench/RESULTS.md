# Embedding backend benchmark — results

**Question.** Embedding `~/src/exe` is slow. Can we go faster — ideally by using
the M4 Max GPU or Neural Engine — within a **16 GB** memory budget?

**Short answer.** Two production-safe wins, depending on platform:
1. **GPU (MLX) is the fastest on Apple Silicon** — MiniLM/bge on the Apple GPU
   reaches **~405–500 chunks/s in 0.27 GB at full ≤512-token context**.
2. **CPU/ONNX should not batch variable-length code chunks together.** On the
   target 8-vCPU AMD EPYC VM, batch 1 across eight single-thread sessions is
   **2.26× faster** than one eight-thread session at batch 16, with only
   0.20 GB additional peak RSS and byte-identical embeddings.

A direct fixed-sequence ONNX path remains an interesting Apple CPU experiment,
but shortening `seq_len` truncates inputs and therefore was not shipped.

The dead ends (all verified): the ONNX **CoreML EP partitions BERT and falls
back to CPU** (ANE=GPU=CPU); **candle Metal** is too unoptimized (~30/s); and
Apple's **`afm` server serializes** (~57/s) — though its underlying
`NLContextualEmbedding` parallelizes to ~400/s when you skip the actor + HTTP.
The GPU only wins with an *optimized* runtime (MLX) + large batches.

## Linux CPU follow-up — August 30, 2026

Machine: 8-vCPU AMD EPYC 9554P, 16 GiB RAM. Workload: a deterministic,
SHA-ordered corpus extracted from the current `exe` checkout with the production
chunker (131,715 chunks total); each run uses the first 4,096 chunks, sorted by
length, after a 256-chunk warmup. Values are medians of three runs.

| configuration | batch | fastembed group/session | chunks/s | peak RSS |
|---|---:|---:|---:|---:|
| previous default: 1 session × 8 threads | 16 | 512 | 33 | 1.20 GB |
| new default: 8 sessions × 1 thread | 1 | 64 | **74** | 1.40 GB |

The new configuration takes 55.39 s versus 124.91 s: **2.26× throughput**.
A one-session screen found batch 1 faster than 4–128, while eight-session tests
found batches 2, 3, 4, 8, and 16 slower than 1; batch 256 crossed the 8 GB
watchdog limit. A 256-vector dump from the two configurations compared at mean
and worst cosine 1.00000 and was byte-for-byte identical.

A second validation deliberately ran under active CPU contention: three
unrelated `cloud-hypervisor` processes each held about one CPU core, and load
average rose from 5.29 to 14.44 during the sequence. Three 2,048-chunk runs per
configuration were alternated baseline/candidate to avoid ordering bias:

| configuration | run times (s) | median chunks/s | median peak RSS |
|---|---|---:|---:|
| 1 session × 8 threads, batch 16 | 58.75, 58.85, 58.83 | 35 | 1.19 GB |
| 8 sessions × 1 thread, batch 1 | 26.62, 28.03, 25.28 | **77** | 1.37 GB |

The contended median remained a **2.21× speedup**, with 0.18 GB additional
median peak RSS.

## Final review validation — August 31, 2026

### End-to-end indexing after lazy initialization

The final shipping binary was run against 1,024 files made from the deterministic
production corpus. The cache ref was deleted before every run, and three
baseline/candidate runs were interleaved under active CPU contention. These
numbers include model/session initialization, embedding, and git-note commits:

| configuration | run times (s) | median chunks/s | median peak RSS |
|---|---|---:|---:|
| 1 session × 8 threads, batch 16 | 29.55, 30.46, 30.66 | 34 | 1.18 GB |
| lazy 8 sessions × 1 thread, batch 1 | 13.86, 14.00, 13.73 | **74** | 1.40 GB |

The complete indexing path retains a **2.20× median speedup**, including lazy
worker creation. A separate 128-file run with an empty `XDG_CACHE_HOME`
successfully downloaded the model through the eager session before parallel
workers initialized, then completed indexing without Hugging Face lock errors.

### Cached-search session loading

The shipping CPU embedder now creates one session eagerly and initializes the
remaining planned workers only when `embed_flat` is reached. Since `index_repo`
checks the cache before calling `embed_flat`, a cached auto-index search never
loads the extra sessions. Three eager/lazy runs were interleaved:

| workload | eager-all median | lazy median | peak RSS, eager → lazy |
|---|---:|---:|---:|
| cached 128-file repo, auto-index enabled | 0.50s | **0.22s** | 0.97 → **0.17 GB** |
| large `exe` cache, `--no-auto-index` | 2.69s | **2.38s** | 1.30 → **0.45 GB** |

### Larger-model batch check

BGE-base and Jina remain capped at one automatic worker. Three interleaved
128-chunk runs under the same CPU contention found no batch-1 regression, so
all CPU models retain the semantics-preserving batch-1 default:

| model | batch 1 median | batch 16 median | peak RSS, batch 1 → 16 |
|---|---:|---:|---:|
| BGE-base | **20.12s** (6.36/s) | 21.87s (5.85/s) | 0.64 → 1.15 GB |
| Jina-code | **24.06s** (5.32/s) | 24.96s (5.13/s) | 0.82 → 1.42 GB |

The first ONNX session is also initialized synchronously before any lazy worker
can initialize in parallel. This completes fresh-cache Hugging Face downloads
before parallel access and avoids model-cache lock races.

```sh
# baseline
./target/release/bench --corpus bench/corpus/exe.bin --backend onnx-cpu \
  --model minilm --sessions 1 --threads 8 --batch 16 --group 512 \
  --sort --limit 4096 --warmup 256

# candidate (512 total retained texts / 8 sessions = group 64)
./target/release/bench --corpus bench/corpus/exe.bin --backend onnx-cpu \
  --model minilm --sessions 8 --threads 1 --batch 1 --group 64 \
  --sort --limit 4096 --warmup 256
```

## Rig (pure Rust, in `bench/`)

- **Workload** `bench/corpus/exe.bin`: the 85,798 chunk texts that
  `git-vector-grep index` would embed for `~/src/exe` (10,830 textual files,
  7,159 unique blobs, 49.7 MB), extracted with the production chunker.
  Build with `cargo run -p vgg-bench --release --bin corpus -- --repo ~/src/exe --out bench/corpus/exe.bin`.
- **Runner** `bench/src/bin/bench.rs`: one backend per process under a
  Linux/macOS **memory watchdog** (`--budget-gb`, default 16) that aborts before
  RSS can exhaust RAM. Reports chunks/s, peak RSS, init time;
  `--dump`/`--compare` check two backends agree (cosine).
- **Backends** (`bench/src/backends/`):
  - `onnx-cpu` — fastembed, N sessions × intra-threads.
  - `onnx-coreml` — fastembed + CoreML EP (ANE/GPU), dynamic shapes.
  - `onnx-direct` — direct `ort`, **fixed `[batch, seq_len]`** shapes, CPU or
    CoreML (ANE/GPU). Mean-pooled.
  - `candle-metal` — candle on the Apple GPU (jina-bert), F32/F16. (`--features candle`)
- Machine: M4 Max, 16 cores (12P+4E), 64 GB. `cargo build --release`.

## Headline numbers (minilm, 20k-chunk subset, warmed up)

| backend | mechanism | config | chunks/s | peak RSS |
|---|---|---|---:|---:|
| onnx-cpu (fastembed) | CPU | 8 sessions ×1 thread | 248 | **12.8 GB** |
| onnx-cpu (fastembed) | CPU | 1 session ×8 threads | 82 | 1.8 GB |
| onnx-direct | CPU | 1 session, seq 256 | 167 | 0.64 GB |
| onnx-direct | CPU | 1 session, **seq 128** | **410** | 0.41 GB |
| onnx-direct | CPU | 1 session, **seq 64** | **779** | 0.35 GB |
| onnx-direct | CoreML **ANE** | seq 256 | 167 | 0.65 GB |
| onnx-direct | CoreML **GPU** | seq 256 | 167 | 0.65 GB |
| onnx-direct | CoreML **ALL** | seq 256 | 166 | 0.65 GB |
| onnx-coreml (fastembed) | CoreML, dynamic | — | ~19 | >14 GB |
| candle-metal | GPU (Metal) | jina-en, F32 | 30 | 7.3 GB |
| afm-http | Apple NL (server) | apple-nl-contextual-en, 512d | 58 | 0.16 GB (server) |
| afm-native | Apple NL (parallel) | apple-nl-contextual-en, 512d, 16 threads | **355–406** | 1–3 GB |
| mlx-metal | Apple GPU (MLX) | minilm-l6 / bge-small, batch 128 | **405–500** | **0.27 GB** | full ≤512 context; 8/8 retrieval |

### jina-code (768-dim, the code-aware target model; 6k-chunk subset, seq 256)

| mechanism | chunks/s | peak RSS | CPU during run |
|---|---:|---:|---|
| CPU (1 session) | 27 | 1.5 GB | ~5 cores |
| CoreML **ANE** | 27 | 1.5 GB | **~5 cores** (i.e. CPU fallback) |

The ANE run burns 5 CPU cores and returns the *identical* 27 chunks/s — proof
the Neural Engine never runs the graph; the CoreML EP falls back to CPU. Bigger
model, same outcome.

### Apple's `afm embed` server (apple-nl-contextual-en) — the "proper" ANE path
Apple's own OpenAI-compatible embeddings server (`afm embed`, Apple
NaturalLanguage contextual embeddings, 512-dim) is the ANE path done *right* —
Apple's optimized model, not the partition-and-fall-back ONNX CoreML EP. It is
still the **slowest bulk option**:
- **~57–58 chunks/s, flat across batch size (16→128) and concurrency (1→16).**
  The server **serializes**: 4 parallel `curl`s take 4× the wall time of 1
  (2.9 s vs 0.75 s), so neither batching nor in-flight concurrency helps.
- Server RAM is tiny (~165 MB), client negligible. Zero setup beyond the daemon.

Full exe (85,798 chunks) projected: **~25 min** via afm vs ~3.5 min for
single-session CPU at seq 128.

**But the serialization is in the server, not the model.** MacLocalAPI's
`NLContextualEmbeddingBackend` is a Swift `actor` (serial executor) that also
loops over a batch one text at a time. Calling `NLContextualEmbedding` directly
with **one instance per worker thread** (no actor, no HTTP) parallelizes well
— `bench/swift/afm_native.swift`:

| parallel | chunks/s | peak RSS |
|---:|---:|---:|
| 1 | 59 | 0.2 GB |
| 4 | 146 | 0.4 GB |
| 8 | 285 | 0.9 GB |
| **16** | **355–406** | 1–3 GB |
| 24–32 | ~377 (worse) | 1.2 GB |

So native parallel Apple NL embeddings hit **~355–406 chunks/s** — a **6–7×
speedup over the serialized server**, and on par with the best CPU ONNX path.
Peak at parallel = core count (16); oversubscribing doesn't help, so it's
CPU-bound across cores (NLContextualEmbedding runs on CPU, not a magic ANE win).
Full exe ≈ **4 min** (vs 25 via the server). This makes Apple's 512-dim
contextual model a viable option if we want it — via a small native Swift
helper, not the HTTP server.

### MLX on the Apple GPU — the winner (fast AND correct)
MLXEmbedders (`bench/swift-mlx/`) runs MiniLM-L6 / bge-small on the GPU with
large batches — the one path that beats CPU on throughput *at full context*:
- **405 chunks/s full corpus / ~500 on a 20k subset, in 0.27 GB**, at the full
  ≤512-token context (the CPU's 410/s ran at 128-token truncation). At equal
  context MLX is several× the CPU. Peaks at batch 32–128; bigger batches lose to
  padding. Leaves all 16 CPU cores free → ~2.9–3.5 min for the full exe index.
- **Embeddings are correct.** Retrieval test (8 sentences, 4 topic pairs):
  MLX-bge gets **8/8** nearest-neighbor pairs right — same as fastembed-bge, and
  the same similarity structure (cat~kitten 0.73 vs 0.74). MLX vectors live in a
  different *basis* than fastembed's (near-zero raw cross-cosine on identical
  weights — likely a normalization/export quirk), so you can't mix an MLX cache
  with a fastembed one, but a self-consistent MLX index retrieves correctly.
- **One-time setup (Metal Toolchain).** `swift build` does NOT compile Metal, so
  no metallib is produced and MLX won't start. Install the toolchain once
  (`xcodebuild -downloadComponent MetalToolchain`, ~688 MB) and run
  `bench/swift-mlx/build-metallib.sh` — it compiles MLX's JIT-on minimal kernel
  set from the *resolved* mlx-swift source and drops `mlx.metallib` next to the
  binary. (Borrowing a metallib from a different mlx-swift version still runs and
  — surprisingly — still retrieves correctly here, but version-match removes all
  doubt.)

### Correctness (CPU/ONNX)
`onnx-direct` vs fastembed (minilm, 256 chunks): **mean cosine 0.993**, worst
0.862 (the worst cases are chunks longer than `seq_len`, which `onnx-direct`
truncates). ANE output is bit-identical to CPU (0.99344 either way) — same math,
same cores.

## What we learned

### The accelerators don't engage for BERT embeddings
- **CoreML ANE = GPU = CPU = 167 chunks/s** (minilm, fixed shapes). The CoreML
  execution provider splits the BERT graph into many partitions and runs the
  glue on CPU; the Neural Engine never meaningfully kicks in. Identical results
  for `CPUAndNeuralEngine`, `CPUAndGPU`, and `ALL`.
- **candle + Metal** reached only ~30 chunks/s for jina (F32); F16 (which the
  Apple GPU is much faster at) **crashes inside candle's `jina_bert`** — its
  ALiBi bias is F32 and mismatches F16 attention scores. candle's Metal backend
  isn't optimized enough to beat CPU for these small models.
- **MLX** has no mature Rust binding; ruled out under the Rust-only constraint.

### Two CoreML traps (both real, both caught by the rig)
1. **Dynamic shapes are fatal.** The CoreML EP compiles+caches a separate model
   per unique input shape. fastembed pads `BatchLongest`, so nearly every batch
   is a new shape → constant recompilation: a 512-text warmup took **26.5 s**
   and RSS blew past **14 GB**. With MLProgram + GPU/ANE + dynamic shapes it
   **SIGSEGVs**. → You must feed *fixed* `[batch, seq_len]` shapes (what
   `onnx-direct` does); then it's stable and warms up in ~1.5 s.
2. **fastembed retains all hidden states per `embed()` call** (every batch's
   `[batch, seq, hidden]` tensor is held until the call returns). Embedding all
   85.8k chunks at once needs tens of GB; the watchdog aborted it at 16 GB. Both
   the shipping indexer and the rig bound this by calling `embed()` in groups.

### Platform-specific CPU findings
- On the **AMD EPYC Linux target**, fastembed's dynamic `BatchLongest` padding
  makes batch 1 fastest for mixed-length code. Eight single-thread sessions
  keep all eight cores busy at 74 chunks/s and 1.40 GB peak RSS.
- On the **M4 Max fixed-shape experiment**, one multi-threaded direct-ORT
  session was memory-efficient, and shorter fixed sequence lengths were fast.
  That path was not shipped because truncating to 64/128/256 tokens can change
  embeddings for longer chunks.

## Recommendation

For the shipping CPU backend, retain fastembed's full model context and use:
1. **batch 1** by default, eliminating padding between unrelated code chunks;
2. up to **eight RAM-capped sessions** for MiniLM/BGE-small, with intra-op
   threads divided among them; larger BGE-base/Jina models retain the previous
   one-session default unless `--workers N` explicitly opts in;
3. bounded per-session calls so each index checkpoint retains about 512 texts
   total.

The automatic worker policy is deliberately conservative outside the measured
small-model path: it reserves at least 1.5 GiB and 25% of the lower of host or
cgroup memory (`memory.max` on v2 and common `memory.limit_in_bytes` v1
layouts), caps MiniLM/BGE-small at eight workers, falls back to one worker when
RAM cannot be detected, and leaves BGE-base/Jina at one worker by default.
Only the first session is eager; additional workers are initialized after the
cache scan proves embedding work exists.

Keep CoreML/candle and sequence-length truncation out of the CPU hot path: none
provided a semantics-preserving win on this target.
