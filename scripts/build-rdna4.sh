#!/usr/bin/env bash
# RDNA4 build (RX 9000 class).
#
# RDNA4 ships as two AMDGPU targets: gfx1200 (Navi 44) and gfx1201 (Navi 48, which
# covers the RX 9070 / 9070 XT and the AI PRO R9700). A binary built for the wrong
# one still links and loads the model, then segfaults or hangs (issues #18, #37), so
# build for the installed GPU when it is visible. An explicit CMAKE_HIP_ARCHITECTURES
# always wins, and gfx1200 stays the default when no RDNA4 GPU can be detected.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/rocmfpx-hip-arch.sh"

HIP_ARCH="$(rocmfpx_select_hip_arch gfx1200 '^gfx120[01]$')"

exec env CMAKE_HIP_ARCHITECTURES="${HIP_ARCH}" BUILD_DIR="${BUILD_DIR:-build-rdna4}" \
    "${SCRIPT_DIR}/build-rocmfp4.sh" "$@"
