#!/usr/bin/env python3
"""PP-0 ablation harness for the W4A4 MMQ dot-product epilogue.

Produces two deliberately-incorrect builds of
ggml_cuda_mmq_vec_dot_rocmi4_w4a4_wmma to attribute kernel time between the
MMA/operand-staging path, the per-element scale loads, and the per-element
scale arithmetic. See docs/rocmfpx/plans/rdna3-gfx1100-tg-pp-acceleration.md.

Measured on gfx1100, m=4096 n=512 k=14336, via
  test-backend-ops perf -o MUL_MAT -p type_a=q4_0_rocmi4

    baseline (full epilogue)   758.67 - 763.30 us   78.78 - 79.26 TFLOPS
    B: loads kept, math gone   714.86 us            84.11 TFLOPS
    A: no loads, no math       592.14 us           101.55 TFLOPS

Usage:
    python3 ablate.py save      # back up the source (do this first)
    python3 ablate.py A|B       # apply a variant
    ... rebuild, measure ...
    python3 ablate.py restore   # put the source back

NEVER merge an ablated build. Both variants are numerically wrong; they exist
only to time the kernel without the epilogue.
"""

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.normpath(os.path.join(HERE, "..", "..", "src", "ggml-cuda", "mmq-vec-dot.cuh"))
BK = F + ".orig"

ORIG = """                    const int i = i0 + n*tile_A::I + tile_C::get_i(l);
                    const float dA = x_df[i*sram_stride + k0/QI8_0];
                    const int acc = C.x[l]*16;
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += acc*dA*dB;
"""

# A: no scale loads, no scale arithmetic -> isolates MMA + operand staging.
A = """                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += (float)C.x[l];
"""

# B: both scale loads kept alive, multiply chain removed -> isolates the ALU cost.
# The `+ dA + dB` terms are cheap and keep both loads from being eliminated.
B = """                    const int i = i0 + n*tile_A::I + tile_C::get_i(l);
                    const float dA = x_df[i*sram_stride + k0/QI8_0];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += (float)C.x[l] + dA + dB;
"""


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "save":
        shutil.copyfile(F, BK)
        print("backup saved")
        return
    if mode == "restore":
        shutil.copyfile(BK, F)
        os.remove(BK)
        print("restored")
        return
    if mode not in ("A", "B"):
        print(__doc__)
        sys.exit(1)
    with open(F) as fh:
        src = fh.read()
    if ORIG not in src:
        print("ERROR: epilogue anchor not found (already patched?)")
        sys.exit(1)
    with open(F, "w") as fh:
        fh.write(src.replace(ORIG, A if mode == "A" else B))
    print("applied variant " + mode)


if __name__ == "__main__":
    main()
