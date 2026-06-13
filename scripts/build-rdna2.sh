#!/usr/bin/env bash
# RDNA2 build (RX 6000 class) — targets gfx1030
exec env CMAKE_HIP_ARCHITECTURES=gfx1030 BUILD_DIR="${BUILD_DIR:-build-rdna2}" \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build-rocmfp4.sh" "$@"
