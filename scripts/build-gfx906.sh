#!/usr/bin/env bash
# Experimental Vega 20 / MI50 build target.
#
# This path is provided for community testing on gfx906 hardware. ROCmFP4
# performance tuning remains validated primarily on Strix Halo / RDNA3.5.
#
# An explicit CMAKE_HIP_ARCHITECTURES always wins; gfx906 is the default.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/rocmfpx-hip-arch.sh"

HIP_ARCH="$(rocmfpx_select_hip_arch gfx906 '^gfx906$')"

exec env CMAKE_HIP_ARCHITECTURES="${HIP_ARCH}" BUILD_DIR="${BUILD_DIR:-build-gfx906}" \
    "${SCRIPT_DIR}/build-rocmfp4.sh" "$@"
