#!/usr/bin/env bash
# Experimental Vega 20 / MI50 build target.
#
# This path is provided for community testing on gfx906 hardware. ROCmFP4
# performance tuning remains validated primarily on Strix Halo / RDNA3.5.
exec env CMAKE_HIP_ARCHITECTURES=gfx906 BUILD_DIR="${BUILD_DIR:-build-gfx906}" \
    "$(dirname "$0")/build-rocmfp4.sh"
