//! Configure build metadata and platform-specific embedding support.
//!
//! The CLI version records the exact git tag/commit when available. We also
//! gate MLX by *platform* rather than a Cargo feature so a plain `cargo build`
//! on an Apple-Silicon Mac includes the GPU backend (used by default at runtime
//! via `--backend auto`), while Linux / Intel builds stay pure CPU/ONNX with no
//! extra dependencies. Building on Apple Silicon requires the Metal Toolchain
//! (`xcodebuild -downloadComponent MetalToolchain`), which mlx-sys uses to
//! compile MLX + its metallib.

use std::path::{Path, PathBuf};
use std::process::Command;

fn git_output(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

fn git_path(root: &Path, spec: &str) -> Option<PathBuf> {
    let path = PathBuf::from(git_output(root, &["rev-parse", "--git-path", spec])?);
    Some(if path.is_absolute() {
        path
    } else {
        root.join(path)
    })
}

fn watch_git_path(root: &Path, spec: &str) {
    if let Some(path) = git_path(root, spec).filter(|path| path.exists()) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn build_version() -> String {
    println!("cargo:rerun-if-env-changed=GIT_VECTOR_GREP_VERSION");
    if let Ok(version) = std::env::var("GIT_VECTOR_GREP_VERSION") {
        let version = version.trim();
        if !version.is_empty() && !version.bytes().any(|byte| matches!(byte, b'\r' | b'\n')) {
            return version.to_owned();
        }
    }

    let package_version = || std::env::var("CARGO_PKG_VERSION").unwrap();
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let Ok(root) = root.canonicalize() else {
        return package_version();
    };
    let Some(git_root) = git_output(&root, &["rev-parse", "--show-toplevel"])
        .and_then(|path| PathBuf::from(path).canonicalize().ok())
    else {
        return package_version();
    };
    if git_root != root {
        return package_version();
    }

    watch_git_path(&root, "HEAD");
    watch_git_path(&root, "logs/HEAD");
    watch_git_path(&root, "packed-refs");
    watch_git_path(&root, "refs/heads");
    watch_git_path(&root, "refs/tags");
    if let Some(head_ref) = git_output(&root, &["symbolic-ref", "-q", "HEAD"]) {
        watch_git_path(&root, &head_ref);
    }

    let has_tags = git_output(
        &root,
        &["describe", "--tags", "--abbrev=0", "--match", "v[0-9]*"],
    )
    .is_some();
    git_output(
        &root,
        &["describe", "--tags", "--always", "--match", "v[0-9]*"],
    )
    .map(|version| {
        if has_tags {
            version.strip_prefix('v').unwrap_or(&version).to_owned()
        } else {
            version
        }
    })
    .unwrap_or_else(package_version)
}

fn main() {
    println!(
        "cargo:rustc-env=GIT_VECTOR_GREP_VERSION={}",
        build_version()
    );
    println!("cargo:rustc-check-cfg=cfg(mlx)");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if os == "macos" && arch == "aarch64" {
        println!("cargo:rustc-cfg=mlx");
    }
}
