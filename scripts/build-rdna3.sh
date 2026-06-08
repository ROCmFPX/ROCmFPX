#!/usr/bin/env bash
# RDNA3 build (RX 7000 class) — targets gfx1100
exec env CMAKE_HIP_ARCHITECTURES=gfx1100 BUILD_DIR="${BUILD_DIR:-build-rdna3}" \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build-rocmfp4.sh" "$@"
