//! Enable the MLX (Apple GPU) embedding backend automatically on Apple Silicon.
//!
//! We gate MLX by *platform* rather than a Cargo feature so a plain
//! `cargo build` on an Apple-Silicon Mac includes the GPU backend (used by
//! default at runtime via `--backend auto`), while Linux / Intel builds stay
//! pure CPU/ONNX with no extra dependencies. The trade-off: building on Apple
//! Silicon requires the Metal Toolchain (`xcodebuild -downloadComponent
//! MetalToolchain`), which mlx-sys uses to compile MLX + its metallib.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(mlx)");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if os == "macos" && arch == "aarch64" {
        println!("cargo:rustc-cfg=mlx");
    }
}
