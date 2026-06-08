#!/usr/bin/env bash
# RDNA4 build (RX 9000 class) — targets gfx1200
exec env CMAKE_HIP_ARCHITECTURES=gfx1200 BUILD_DIR="${BUILD_DIR:-build-rdna4}" \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build-rocmfp4.sh" "$@"
