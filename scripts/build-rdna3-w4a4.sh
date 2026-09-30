#!/usr/bin/env bash
# RDNA3 HIP-only build with opt-in ROCmI4 packed W4A4 / IU4 MMQ.
#
# Defaults to gfx1100 (RX 7900 class). Prefer building inside a ROCm 10
# container on hosts without a system ROCm toolchain:
#
#   docker run --rm --device=/dev/kfd --device=/dev/dri \
#     --group-add 44 --group-add 993 \
#     -v "$PWD":/workspace -w /workspace rocm-dev:10.0.0 \
#     bash -lc 'apt-get update -qq && apt-get install -y -qq ninja-build pkg-config libcurl4-openssl-dev && scripts/build-rdna3-w4a4.sh'
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/rocmfpx-hip-arch.sh"

HIP_ARCH="$(rocmfpx_select_hip_arch gfx1100 '^gfx11[0-9a-f]{2}$')"

exec env \
    CMAKE_HIP_ARCHITECTURES="${HIP_ARCH}" \
    BUILD_DIR="${BUILD_DIR:-build-rdna3-w4a4}" \
    GGML_HIP_ROCMI4_W4A4=ON \
    GGML_VULKAN=OFF \
    "${SCRIPT_DIR}/build-rocmfp4.sh" "$@"
