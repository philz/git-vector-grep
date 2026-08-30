//! Backend-agnostic embedding interface.
//!
//! The shipping default is `embedder::Embedder` (ONNX/CPU via fastembed). With
//! the `mlx` feature, `mlx_embed::MlxEmbedder` (Apple GPU via mlx-rs) also
//! implements this trait. Everything downstream (the indexer, the store, search)
//! depends only on `short_id` + `dim` + the produced vectors, so backends are
//! interchangeable and their caches coexist (one notes ref per `short_id`).

use anyhow::Result;

pub trait Embed: Send + Sync {
    /// Embed many texts; returns a flat L2-normalized `texts.len() * dim` buffer.
    fn embed_flat(&self, texts: Vec<String>, batch_size: usize) -> Result<Vec<f32>>;
    /// Embed a single query string (L2-normalized).
    fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
    /// Backend-specific default batch size. CPU/ONNX prefers one text per
    /// dynamic-shape batch to avoid padding; accelerator backends may prefer
    /// larger batches.
    fn default_batch_size(&self) -> usize {
        16
    }
    fn dim(&self) -> usize;
    /// Stable slug used as the notes-ref segment (`[a-z0-9-]+`).
    fn short_id(&self) -> &str;
    /// Canonical, human-facing model id (for logs).
    fn model_id(&self) -> &str {
        self.short_id()
    }
    /// One-line summary of the active backend, its resource caps, and how to
    /// tweak them. Printed when an index actually embeds new chunks.
    fn describe(&self) -> String {
        self.short_id().to_string()
    }
}
