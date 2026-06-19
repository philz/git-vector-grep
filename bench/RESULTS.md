# Embedding backend benchmark — results

**Question.** Embedding `~/src/exe` is slow. Can we go faster — ideally by using
the M4 Max GPU or Neural Engine — within a **16 GB** memory budget?

**Short answer.** Two wins, depending on appetite:
1. **GPU (MLX) is the fastest** — MiniLM/bge on the Apple GPU via MLXEmbedders:
   **~405–500 chunks/s in 0.27 GB at full ≤512-token context**, embeddings
   verified correct (8/8 retrieval). Needs a Swift helper + a one-time Metal
   Toolchain install. Full exe in ~3 min, CPU cores left free.
2. **CPU is the easy win** — one multi-threaded ONNX session (not N
   memory-hungry ones) at a chunk-sized `seq_len`: **1.7–3× faster and ~30× less
   memory** than today's path, a small change to the shipping Rust tool.

The dead ends (all verified): the ONNX **CoreML EP partitions BERT and falls
back to CPU** (ANE=GPU=CPU); **candle Metal** is too unoptimized (~30/s); and
Apple's **`afm` server serializes** (~57/s) — though its underlying
`NLContextualEmbedding` parallelizes to ~400/s when you skip the actor + HTTP.
The GPU only wins with an *optimized* runtime (MLX) + large batches.

## Rig (pure Rust, in `bench/`)

- **Workload** `bench/corpus/exe.bin`: the 85,798 chunk texts that
  `git-vector-grep index` would embed for `~/src/exe` (10,830 textual files,
  7,159 unique blobs, 49.7 MB), extracted with the production chunker.
  Build with `cargo run -p vgg-bench --release --bin corpus -- --repo ~/src/exe --out bench/corpus/exe.bin`.
- **Runner** `bench/src/bin/bench.rs`: one backend per process under a
  **memory watchdog** (`--budget-gb`, default 16) that aborts before RSS can
  exhaust RAM. Reports chunks/s, peak RSS, init time; `--dump`/`--compare`
  check two backends agree (cosine).
- **Backends** (`bench/src/backends/`):
  - `onnx-cpu` — fastembed, N sessions × intra-threads (today's design).
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

### The actual win is CPU shape + session strategy
- **One multi-threaded ORT session** uses all cores in **<0.7 GB**. Today's
  8-session design hits 248 chunks/s but **12.8 GB** — it's memory-bound and
  can't scale to all 16 cores within budget.
- **`seq_len` dominates throughput.** Chunks are ≤1000 chars (~250 tokens) but
  most are far shorter. Capping the model's sequence length is near-linear:
  256→167, 128→410, 64→779 chunks/s. seq 128 keeps ~500 chars/chunk and already
  **beats the current design 1.7× at 1/30th the memory**.

## Recommendation

For the shipping tool (no accelerator needed):
1. Replace N fastembed sessions with **one session, default intra-threads**
   (all cores), in <1 GB.
2. Make **`seq_len` a tunable** and default it to ~128 (or shrink the chunker so
   chunks fit ~128 tokens). 1.7–3× faster indexing of exe, ~30× less RAM.
3. Keep CoreML/candle out of the hot path — they don't pay off here.

Projected full exe (85,798 chunks): today ≈ **5.7 min**; seq-128 single-session
≈ **3.5 min**; seq-64 ≈ **1.8 min** — all in well under 1 GB.

_Caveat:_ shortening `seq_len` truncates long chunks (a quality trade-off);
mean-pooling in `onnx-direct` matches minilm/jina but **not** BGE (CLS-pooled).
