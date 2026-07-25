#!/usr/bin/env bash
# RDNA2 build (RX 6000 class).
#
# Defaults to gfx1030, but builds for the installed GPU when it is another RDNA2
# target (gfx1031, gfx1032, gfx1034, ...), because a binary built for the wrong
# target links and loads and then fails at runtime. An explicit
# CMAKE_HIP_ARCHITECTURES always wins.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/rocmfpx-hip-arch.sh"

HIP_ARCH="$(rocmfpx_select_hip_arch gfx1030 '^gfx103[0-9a-f]$')"

exec env CMAKE_HIP_ARCHITECTURES="${HIP_ARCH}" BUILD_DIR="${BUILD_DIR:-build-rdna2}" \
    "${SCRIPT_DIR}/build-rocmfp4.sh" "$@"
