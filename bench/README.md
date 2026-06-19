# vgg-bench — embedding backend benchmark rig

Compares embedding **mechanisms** (CPU vs CoreML ANE/GPU vs candle Metal) and
**models** on one fixed workload, on Apple silicon, within a memory budget.
See [RESULTS.md](RESULTS.md) for findings.

## Build

```sh
cargo build --release -p vgg-bench                  # onnx-cpu, onnx-coreml, onnx-direct
cargo build --release -p vgg-bench --features candle # adds candle-metal (heavy)
```

## 1. Make the corpus (once)

Extracts a repo's chunk texts with the production chunker into a stable file:

```sh
./target/release/corpus --repo ~/src/exe --out bench/corpus/exe.bin
```

## 2. Run a backend

```sh
# CPU baseline (today's design): N fastembed sessions
./target/release/bench --corpus bench/corpus/exe.bin --backend onnx-cpu \
    --model minilm --sessions 8 --batch 16 --sort

# Fixed-shape direct ORT on CPU — the fast, low-memory path
./target/release/bench --corpus bench/corpus/exe.bin --backend onnx-direct \
    --model minilm --seq-len 128 --batch 16 --coreml-units none

# Fixed-shape on the CoreML ANE / GPU
./target/release/bench --corpus bench/corpus/exe.bin --backend onnx-direct \
    --model minilm --seq-len 256 --coreml-units ane --warmup 256

# candle on the Apple GPU (needs --features candle)
./target/release/bench --corpus bench/corpus/exe.bin --backend candle-metal \
    --model jina-en --batch 64 --f16
```

Every run is under a **memory watchdog** (`--budget-gb`, default 16) that aborts
the process before RSS can exhaust RAM. Use `--limit N` to run on a subset,
`--warmup N` to exclude one-time compile/cache costs, and `--dump`/`--compare`
to check two backends agree (cosine).

## Backends

| `--backend` | mechanism | notes |
|---|---|---|
| `onnx-cpu` | CPU | fastembed, `--sessions`×`--threads`. Dynamic (BatchLongest) shapes. |
| `onnx-coreml` | CoreML | fastembed + CoreML EP. Dynamic shapes — see RESULTS (unusable). |
| `onnx-direct` | CPU or CoreML | direct `ort`, **fixed** `[batch, seq_len]`. `--coreml-units none\|ane\|gpu\|all\|cpu`. |
| `candle-metal` | Apple GPU | candle jina-bert. `--model jina-en\|jina-code` (code needs a custom model), `--f16`. |
| `afm-http` | Apple NL (server) | POST to `afm embed` `/v1/embeddings`. `--afm-url`, `--sessions` (in-flight requests). |

## Native Apple NL embeddings (Swift)

`bench/swift/afm_native.swift` calls `NLContextualEmbedding` directly — one
instance per worker thread, no actor, no HTTP — to show the server's
serialization, not the model, is the bottleneck:

```sh
swiftc -O bench/swift/afm_native.swift -o target/afm_native
./target/afm_native --corpus bench/corpus/exe.bin --parallel 16 --limit 20000
```

~355–406 chunks/s at 16 threads vs ~57/s for the HTTP server.
