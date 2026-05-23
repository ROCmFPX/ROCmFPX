#!/usr/bin/env bash
# Build the isolated Strix Halo ROCmFP4 + MTP llama.cpp tree.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
JOBS="${JOBS:-$(nproc)}"

cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DGGML_HIP_ROCWMMA_FATTN="${GGML_HIP_ROCWMMA_FATTN:-OFF}" \
    -DGGML_HIP_FORCE_MMQ=ON \
    -DGGML_VULKAN=ON \
    -DGGML_CUDA=OFF \
    -DCMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES:-gfx1151}" \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_WEBUI=OFF \
    -DLLAMA_USE_PREBUILT_WEBUI=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DGGML_BUILD_TESTS=OFF

cmake --build "$BUILD_DIR" -j "$JOBS" --target llama-cli llama-quantize llama-bench

echo "Built:"
echo "  $BUILD_DIR/bin/llama-cli"
echo "  $BUILD_DIR/bin/llama-quantize"
echo "  $BUILD_DIR/bin/llama-bench"
