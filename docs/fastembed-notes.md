# fastembed-rs research notes

- Latest: **5.15.0** (published 2026-05-30 by Anush008).
- `rust-version` field: **not set** in Cargo.toml. edition = "2021". Uses ort 2.0.0-rc.12, ndarray 0.17, tokenizers 0.22, hf-hub 0.5. So MSRV is effectively whatever those transitive deps require (modern stable; in practice 1.75+ works).
- `jinaai/jina-embeddings-v2-base-code` IS supported as `EmbeddingModel::JinaEmbeddingsV2BaseCode`, dim 768, model_code `jinaai/jina-embeddings-v2-base-code`, file `onnx/model.onnx`.
- Default model: `EmbeddingModel::BGESmallENV15` (dim 384), via `Default::default()`.
- ONNX runtime: ships via `ort` crate; default features include `ort-download-binaries-native-tls` which downloads a prebuilt ORT static lib at build-time (no system shared lib needed). Alternative: `ort-load-dynamic` to load a system libonnxruntime at runtime.
- Default features: `["ort-download-binaries-native-tls", "hf-hub-native-tls", "image-models"]`.
- Cache: `FASTEMBED_CACHE_DIR` env var, default `./.fastembed_cache` (relative). Override via `InitOptions::with_cache_dir(PathBuf)`.
- Threading: ORT intra-op threads default to `std::thread::available_parallelism()` (all cores). Cap via `InitOptions::with_intra_threads(n)`.
- `model.embed(documents, batch_size: Option<usize>)` returns `Vec<Vec<f32>>`. Default batch size 256.
