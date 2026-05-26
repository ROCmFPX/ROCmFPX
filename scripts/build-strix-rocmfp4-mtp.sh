#!/usr/bin/env bash
# Build the isolated Strix Halo ROCmFP4 + MTP llama.cpp tree.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
JOBS="${JOBS:-$(nproc)}"
ROCM_WMMA_INCLUDE="${ROCM_WMMA_INCLUDE:-/home/caf/strix-fp4/third_party/rocWMMA/library/include}"
HIP_EXTRA_FLAGS="${CMAKE_HIP_FLAGS:-}"

if [[ "${GGML_HIP_ROCWMMA_FATTN:-OFF}" == "ON" ]]; then
    if [[ -d "$ROCM_WMMA_INCLUDE/rocwmma/internal" ]]; then
        HIP_EXTRA_FLAGS="-I${ROCM_WMMA_INCLUDE} ${HIP_EXTRA_FLAGS}"
        echo "Using local rocWMMA headers: $ROCM_WMMA_INCLUDE"
    else
        echo "Warning: rocWMMA headers not found at $ROCM_WMMA_INCLUDE" >&2
    fi
fi

cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DGGML_HIP_ROCWMMA_FATTN="${GGML_HIP_ROCWMMA_FATTN:-OFF}" \
    -DGGML_HIP_FORCE_MMQ=ON \
    -DGGML_VULKAN=ON \
    -DGGML_CUDA=OFF \
    -DCMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES:-gfx1151}" \
    -DCMAKE_HIP_FLAGS="$HIP_EXTRA_FLAGS" \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_WEBUI=OFF \
    -DLLAMA_USE_PREBUILT_WEBUI=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DGGML_BUILD_TESTS=OFF

cmake --build "$BUILD_DIR" -j "$JOBS" --target \
    llama-cli \
    llama-server \
    llama-completion \
    llama-quantize \
    llama-bench \
    test-backend-ops \
    test-quantize-fns \
    test-quantize-perf

echo "Built:"
echo "  $BUILD_DIR/bin/llama-cli"
echo "  $BUILD_DIR/bin/llama-server"
echo "  $BUILD_DIR/bin/llama-completion"
echo "  $BUILD_DIR/bin/llama-quantize"
echo "  $BUILD_DIR/bin/llama-bench"
echo "  $BUILD_DIR/bin/test-backend-ops"
echo "  $BUILD_DIR/bin/test-quantize-fns"
echo "  $BUILD_DIR/bin/test-quantize-perf"
