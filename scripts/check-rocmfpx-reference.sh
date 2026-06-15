#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-rocmfpx-reference}"
CC_BIN="${CC:-cc}"

mkdir -p "$BUILD_DIR"

"$CC_BIN" \
    -std=c11 \
    -Wall \
    -Wextra \
    -pedantic \
    -I"$ROOT/ggml/include" \
    -I"$ROOT/ggml/rocmfpx" \
    "$ROOT/ggml/rocmfpx/rocmfpx.c" \
    "$ROOT/ggml/rocmfpx/test_rocmfpx.c" \
    -lm \
    -o "$BUILD_DIR/test-rocmfpx"

"$BUILD_DIR/test-rocmfpx"
