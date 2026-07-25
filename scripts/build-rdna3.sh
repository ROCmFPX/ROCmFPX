#!/usr/bin/env bash
# RDNA3 build (RX 7000 class).
#
# Defaults to gfx1100, but builds for the installed GPU when it is another RDNA3 /
# RDNA3.5 target (gfx1101, gfx1102, gfx1151, ...), because a binary built for the
# wrong target links and loads and then fails at runtime. An explicit
# CMAKE_HIP_ARCHITECTURES always wins.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/rocmfpx-hip-arch.sh"

HIP_ARCH="$(rocmfpx_select_hip_arch gfx1100 '^gfx11[0-9a-f]{2}$')"

exec env CMAKE_HIP_ARCHITECTURES="${HIP_ARCH}" BUILD_DIR="${BUILD_DIR:-build-rdna3}" \
    "${SCRIPT_DIR}/build-rocmfp4.sh" "$@"
