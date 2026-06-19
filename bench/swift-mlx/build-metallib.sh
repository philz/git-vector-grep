#!/usr/bin/env bash
# Build the MLX Metal shader library for the resolved mlx-swift version.
#
# `swift build` does NOT compile any Metal — mlx-swift expects a prebuilt
# metallib colocated with the binary (device.cpp searches for `mlx.metallib`
# next to the executable first). Xcode 26 ships the Metal compiler as a separate
# component; install it once with:
#     xcodebuild -downloadComponent MetalToolchain
#
# This compiles the MLX "JIT-on" minimal precompiled kernel set from the
# *resolved* mlx-swift source (so it matches the version you build against) and
# installs it next to the release binary. Everything else MLX JIT-compiles at
# runtime. Re-run after changing the mlx-swift version.
set -euo pipefail
cd "$(dirname "$0")"

MLXROOT="$PWD/.build/checkouts/mlx-swift/Source/Cmlx/mlx"
KDIR="$MLXROOT/mlx/backend/metal/kernels"
OUT="$PWD/.build/release/mlx.metallib"
[ -d "$KDIR" ] || { echo "mlx-swift not resolved yet — run 'swift build -c release' first"; exit 1; }

if ! echo 'kernel void _probe(){}' | xcrun -sdk macosx metal -x metal -c - -o /dev/null 2>/dev/null; then
  echo "Metal toolchain unusable. Install it: xcodebuild -downloadComponent MetalToolchain" >&2
  exit 1
fi

BUILD="$(mktemp -d /tmp/vgg-metallib.XXXXXX)"
FLAGS=(-x metal -Wall -Wextra -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions -mmacosx-version-min=26.0)
# MLX's always-precompiled (JIT-on) translation units.
TUS=(arg_reduce conv gemv layer_norm random rms_norm rope scaled_dot_product_attention steel/attn/kernels/steel_attention)
AIRS=()
for tu in "${TUS[@]}"; do
  air="$BUILD/$(basename "$tu").air"
  xcrun -sdk macosx metal "${FLAGS[@]}" -c "$KDIR/$tu.metal" -I"$MLXROOT" -o "$air"
  AIRS+=("$air")
  echo "  metal -c $tu.metal"
done
xcrun -sdk macosx metal "${AIRS[@]}" -o "$OUT"
echo "installed $OUT ($(stat -f%z "$OUT") bytes)"
