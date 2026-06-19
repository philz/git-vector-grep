//! Library surface for git-vector-grep.
//!
//! The binary (`src/main.rs`) is a thin CLI over these modules. They are also
//! consumed by the `bench/` crate so that benchmarks chunk and embed against
//! the *exact* same code paths the shipping tool uses.

pub mod chunker;
pub mod embed;
pub mod embedder;
pub mod indexer;
#[cfg(mlx)]
pub mod mlx_embed;
pub mod repo;
pub mod search;
pub mod store;
