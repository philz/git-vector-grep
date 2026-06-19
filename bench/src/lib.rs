//! Shared plumbing for the embedding benchmark rig.
//!
//! - `corpus`: a stable on-disk workload (the chunk texts of a repo), so every
//!   backend embeds the *exact same* inputs and runs are reproducible without
//!   re-walking a 44 GB working tree.
//! - `mem`: resident-set probing + a watchdog thread that aborts the process if
//!   it crosses a memory budget (we must never run the machine out of RAM).
//! - `backends`: the `Backend` trait and the ONNX (fastembed) implementation,
//!   which can target CPU or the CoreML execution provider (ANE/GPU).

pub mod backends;
pub mod corpus;
pub mod mem;
