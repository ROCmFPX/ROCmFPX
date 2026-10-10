# Plan: Expand packed W4A4 IU4 MMQ to discrete RDNA3

Branch: `feature/rdna3-w4a4-iu4-mmq`  
Target host: RX 7900 XTX (`gfx1100`) / ROCm 10 container  
Status: WP0–WP5 green on RX 7900 XTX / gfx1100 (ROCm 10)

## Goal

Enable the existing opt-in ROCmI4 **packed W4A4 IU4 MMQ** path on discrete
RDNA3 (`gfx1100`/`gfx1101`/`gfx1102`) as well as the already-qualified RDNA3.5
(`gfx115x`) devices, without changing the on-disk `Q4_0_ROCMI4` format.

## Decisions already taken

| Decision | Choice |
|---|---|
| Build switch | Keep `GGML_HIP_ROCMI4_W4A4` **opt-in, default OFF** |
| Device gating style | Broader **RDNA3 family macros** (`RDNA3` / `GGML_CUDA_CC_IS_RDNA3`), not gfx1151-only IDs |
| Done bar | Kernel correctness + short model smoke on gfx1100 |

## Why this is plausible

1. **ISA**: RDNA3 documents `V_WMMA_I32_16X16X16_IU4` (ISA §7.9 / opcode 69).
2. **Toolchain**: On `rocm-dev:10.0.0` (HIP 7.15),  
   `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32` **compiles for both `gfx1100` and `gfx1151`**.
3. **Software, not hardware, is the blocker today**: every live path hard-codes
   `__gfx1151__` or `GGML_CUDA_CC_IS_GFX1151`.

## Current architecture (as of branch point)

```text
GGML_HIP_ROCMI4_W4A4=ON
        │
        ▼
  GGML_ROCMI4_W4A4=1   (ggml/src/ggml-hip/CMakeLists.txt)
        │
        ├─ host: quantize_mmq_q8_1_cuda  → IU4 activation grid  (only if GFX1151)
        ├─ host: backend feature ROCMI4_W4A4=1               (only if GFX1151)
        ├─ device dispatch mmq.cuh Q4_0_ROCMI4               (only if __gfx1151__)
        │     ├─ load_tiles_rocmi4_w4a4   (packed nibbles, byte-interleaved)
        │     └─ vec_dot_rocmi4_w4a4_wmma → mma_iu4           (only if __gfx1151__)
        └─ tile configs
              ├─ mmq-config-rdna3-5.cuh : ROCMI4 SRAM layout when W4A4
              └─ rocmfpx_mmq_rdna3.cuh  : always Q8_0 layout (exact int8 path)
```

Exact fallback (default / non-W4A4 / non-1151):

- Weights expanded to int8 staging
- Activations stay Q8_1 MMQ
- Dot uses `ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma` (IU8 WMMA)

Important invariants that must not change:

- Byte-interleaved nibble order `[b+0,b+4,b+1,b+5,...]` on **both** weight and activation packers
- Accumulator scale `C.x[l] * 16` before multiplying by `dA*dB`
- Lossy activation quant to signed `[-8,+7]` only on the W4A4 path
- Backend feature `ROCMI4_W4A4` must be advertised iff the approximate path can actually run

## Work packages

### WP0 — Capability helpers (small, first)

Add one compile-time and one runtime helper so gates stay consistent:

```cpp
// device / compile time (near AMD_WMMA_AVAILABLE)
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
#  define GGML_HIP_WMMA_IU4_AVAILABLE 1
#else
#  define GGML_HIP_WMMA_IU4_AVAILABLE 0
#endif

// host / runtime
static inline bool ggml_cuda_amd_wmma_iu4_available(int cc) {
    return amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA3(cc);
}
```

Notes:

- `RDNA3` is defined for all `__GFX11__`, which includes RDNA3.0 and RDNA3.5.
- RDNA4 (`__GFX12__`) is **excluded** — correct for this plan.
- Prefer these helpers over scattering `__gfx1151__` / `IS_GFX1151` checks.

### WP1 — Unlock device kernels

Replace `__gfx1151__` guards with `GGML_HIP_WMMA_IU4_AVAILABLE` in:

| File | Symbol / site |
|---|---|
| `ggml/src/ggml-cuda/mma.cuh` | `mma_iu4` |
| `ggml/src/ggml-cuda/mmq-vec-dot.cuh` | `ggml_cuda_mmq_vec_dot_rocmi4_w4a4_wmma` |
| `ggml/src/ggml-cuda/mmq.cuh` | `GGML_TYPE_Q4_0_ROCMI4` util-func dispatch |

No algorithm change in this WP — only gate widening.

### WP2 — Unlock host activation path + feature flag

| File | Change |
|---|---|
| `ggml/src/ggml-cuda/quantize.cu` | `GGML_CUDA_CC_IS_GFX1151` → `ggml_cuda_amd_wmma_iu4_available(cc)` for ROCmI4 IU4 activation quant (normal + scatter if present) |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Advertise `ROCMI4_W4A4=1` when any device passes the same runtime helper |

### WP3 — RDNA3.0 MMQ tile configs

`rocmfpx_mmq_rdna3.cuh` currently forces `Q8_0` staging for `Q4_0_ROCMI4`.
For W4A4 it must select `GGML_CUDA_MMQ_SRAM_LAYOUT_ROCMI4`, mirroring
`mmq-config-rdna3-5.cuh`.

Approach:

1. Keep exact-path tiles as today when `GGML_ROCMI4_W4A4 == 0`.
2. When W4A4 is on, switch ROCMI4 cases to `SRAM_LAYOUT_ROCMI4`.
3. Start from the existing RDNA3 tile shape (`128` threads, `64` rows, `J` multiples of 16, LDS < 64 KiB).
4. Only retune occupancy / `J` after WP5 shows a regression vs exact int8.

Also audit comments that say "gfx1151 only" in:

- `mmq.cuh` (`SRAM_LAYOUT_ROCMI4`)
- `mmq-vec-dot.cuh`
- `mma.cuh`
- `ggml/CMakeLists.txt` option text
- `CMakePresets.json` description

### WP4 — Build / preset plumbing

- Keep default OFF.
- Add or extend a HIP-only preset for discrete RDNA3 W4A4, e.g.  
  `rdna3-rocmfpx-w4a4` with:
  - `GGML_HIP=ON`
  - `GGML_VULKAN=OFF` (ROCmI4 has no Vulkan shader; mixed schedulers are unsafe)
  - `CMAKE_HIP_ARCHITECTURES=gfx1100` (or detected gfx11xx)
  - `GGML_HIP_ROCMI4_W4A4=ON`
  - `GGML_HIP_FORCE_MMQ=ON`
- Optional helper: `scripts/build-rdna3-w4a4.sh` wrapping `build-rdna3.sh` / container flow.
- Document container build against `rocm-dev:10.0.0` (this machine has no host ROCm).

### WP5 — Validation (done bar)

Build A (exact) and Build B (W4A4) for `gfx1100`, Vulkan off.

1. **Compile probe** (already green on ROCm 10): IU4 builtin for `gfx1100`.
2. **Backend ops**  
   `test-backend-ops` `MUL_MAT` + `MUL_MAT_ID` for `Q4_0_ROCMI4`  
   - Exact build: NMSE ≤ `5e-4`  
   - W4A4 build: feature `ROCMI4_W4A4` present; NMSE ≤ `1e-2`
3. **Fixed-shape microbench**  
   `llama-bench -m ...Q4_0_ROCMI4.gguf -ngl 999 -fa 1 -p 512 -n 128`  
   Compare exact vs W4A4 `pp512` and non-spec `tg128`.
4. **Short model smoke** on the live Qwen3.8-27B ROCmI4 GGUF:
   - Prompt processing improvement expected (MMQ path)
   - Non-speculative tg may be ~flat (still MMVQ / bandwidth)
   - Optional: one MTP run later; **not** required for WP5 pass

Pass criteria for calling RDNA3 support real:

- No wrong-kernel / `NO_DEVICE_CODE` crashes
- Backend-ops within declared tolerances
- W4A4 `pp512` ≥ exact (or documented why not, with profiles)
- Short generation remains coherent

### WP6 — Docs / support tier

Update after WP5 passes:

- `ggml/rocmfpx/ROCMI4.md` — RDNA3 discrete + RDNA3.5; note ROCm 10 validation host
- `docs/rocmfpx/SUPPORT.md` — W4A4 no longer gfx1151-only experiment wording
- Model card / serve notes if we republish launcher examples for 7900 XTX

Do **not** claim HumanEval parity with the Strix numbers until a follow-up suite is run.

## Explicit non-goals (this branch)

- Changing GGUF / `block_rocmi4` layout
- Making W4A4 default-on
- Runtime switch between exact and W4A4 in one binary (possible later)
- RDNA2 / RDNA4 IU4 enablement
- Vulkan ROCmI4
- Guaranteeing the Strix ~18% MTP gain on discrete cards

## Risk register

| Risk | Mitigation |
|---|---|
| IU4 WMMA thruput/latency differs on Navi31 vs Strix | Microbench before claiming win; keep exact fallback |
| VGPR / LDS pressure worse on discrete configs | Keep RDNA3 tile sizes; profile before widening tiles |
| Accidental enable on non-IU4 targets | Gate via `RDNA3`+`AMD_WMMA` helpers only; keep CMake OFF |
| Mixed HIP+Vulkan build routes ROCmI4 wrong | W4A4 preset forces `GGML_VULKAN=OFF` |
| Accuracy cliff on some prompts | Retain `ROCMI4_W4A4` feature + looser NMSE only for that feature |
| Compiler accepts builtin but hw misbehaves | Backend-ops + smoke before docs |

## Suggested implementation order

1. WP0 helpers  
2. WP1 device gates  
3. WP3 RDNA3 tile layout for W4A4  
4. WP2 host quant + feature flag  
5. WP4 preset/script  
6. Rebuild in ROCm 10 container for `gfx1100`  
7. WP5 validation matrix  
8. WP6 docs  

## First concrete patch sketch (WP0–WP2)

Touch list:

1. `ggml/src/ggml-cuda/common.cuh` or `vendors/hip.h` — `GGML_HIP_WMMA_IU4_AVAILABLE` + host helper  
2. `mma.cuh` — widen `mma_iu4`  
3. `mmq-vec-dot.cuh` — widen W4A4 WMMA dot  
4. `mmq.cuh` — widen ROCMI4 dispatch; comment fix  
5. `quantize.cu` — runtime RDNA3 IU4 activation quant  
6. `ggml-cuda.cu` — feature advertisement  
7. `ggml/rocmfpx/rocmfpx_mmq_rdna3.cuh` — `SRAM_LAYOUT_ROCMI4` when W4A4  
8. `ggml/CMakeLists.txt` / preset text — “RDNA3 family”, not “gfx1151 only”

## Open follow-ups (after smoke)

- Occupancy / `nwarps` retune specific to 7900 XTX
- MTP-16 acceptance A/B on Qwen3.8-27B
- Whether to allow multi-arch fat binaries (`gfx1100;gfx1151`) with both exact and IU4 code objects
- Optional later: runtime env to force exact path inside a W4A4 build

## References

- ISA: `rdna3-shader-instruction-set-architecture-feb-2023_0.md` §7.9 WMMA IU4
- Current docs: `ggml/rocmfpx/ROCMI4.md`
- Support tiers: `docs/rocmfpx/SUPPORT.md`


## Implementation progress

Landed and validated on `feature/rdna3-w4a4-iu4-mmq`:

- WP0–WP3: RDNA3-family IU4 helpers/gates + RDNA3.0 ROCMI4 SRAM layout when W4A4
- Build: `build-rdna3-w4a4/` via `scripts/build-rdna3-w4a4.sh` / ROCm 10 container
- WP5 on RX 7900 XTX:
  - `test-backend-ops` MUL_MAT `q4_0_rocmi4`: 12/12 OK
  - `test-backend-ops` MUL_MAT_ID `q4_0_rocmi4`: 1/1 OK
  - Short smoke: coherent `READY`
  - `llama-bench` exact vs W4A4 (ngl 999, fa 1, r 2):
    - pp512: exact **1015.88 ± 137.45** → W4A4 **1284.02 ± 212.74** (~**+26%**)
    - tg128: exact **42.06 ± 0.13** → W4A4 **42.30 ± 0.14** (flat; MMVQ path)

Next optional: MTP A/B, docs (`ROCMI4.md` / SUPPORT), commit.

## W4A4-MMVQ (decode)

Status: implemented and validated on RX 7900 XTX / gfx1100 (ROCm 10).

Decode stays on MMVQ. Exact MMVQ unpacks weight nibbles and DP4As against Q8
activations (`V_DOT4_I32_IU8`). W4A4-MMVQ instead:

1. Quantizes MMVQ activations onto the signed IU4 grid (`amax/7`, clamp `[-8,7]`)
   packed in the GGUF Q4_0 nibble layout (`lo = elem j`, `hi = elem j+16`).
2. Dots packed weight dwords against packed activation dwords with
   `V_DOT8_I32_IU4` (`ggml_cuda_dot8_iu4`).
3. Scales with the true integer sum: `d_weight * d_act * sumi` (no WMMA `*16`).

Gating matches MMQ: compile-time `GGML_HIP_ROCMI4_W4A4`, runtime
`amd_wmma_iu4_available`. Exact MMVQ remains the default non-W4A4 build.

Host oracle: `tests/test-rocmi4-iu4-dot.cpp` (packed IU4×IU4 DOT8 == unpack+DP4A
against Q8-stored IU4, bit-identical integer sums).

gfx1100 A/B (`llama-bench`, ngl 999, fa 1, r 2):

| build | pp512 t/s | tg128 t/s |
|---|---:|---:|
| exact MMVQ (unpack + DOT4 vs Q8) | 1018.28 ± 136.80 | 42.26 ± 0.10 |
| W4A4-MMQ + W4A4-MMVQ (DOT8) | 1277.66 ± 213.32 | 42.42 ± 0.15 |

Prefill stays the W4A4 MMQ win (~+25%). Decode is still HBM-bandwidth bound:
DOT8 removes the unpack+two-DP4A compute from MMVQ but does not move tg.
Correctness on this gfx1100 run: host oracle `test-rocmi4-iu4-dot` OK;
`test-backend-ops` MUL_MAT `q4_0_rocmi4` 12/12 (including `n=1` MMVQ) and
MUL_MAT_ID 1/1 under `ROCMI4_W4A4` NMSE `1e-2`. HIP 7.15 has no
`__builtin_amdgcn_sudot8`; `mmvq.cu.o` gfx1100 code object contains
`v_dot8_i32_i4` (LLVM mnemonic for `V_DOT8_I32_IU4`) from inline asm
`v_dot8_i32_iu4 ... neg_lo:[1,1,0]` (both operands signed). RDNA3.0 ROCMI4
MMVQ stays at `nwarps=1`.

## AR occupancy (no spec)

Tried on gfx1100 W4A4, no MTP, `llama-bench -ngl 999 -fa 1` against
Qwen3.8-27B Q4_0_ROCMI4. Sweep variables: MMVQ warp count at `ncols=1` and
the Q8_1 MMVQ vector ratio.

| config | pp512 t/s | tg128 t/s | notes |
|---|---:|---:|---|
| nwarps=1, VDR=2 (baseline) | 1277.66 ± 213.32 | 42.42 ± 0.15 | r=2; keep |
| nwarps=8, VDR=2 | 1330.62 ± 172.82 | **40.18 ± 0.11** | r=3; AR regression |
| nwarps=4, VDR=2 | 1329.75 ± 175.75 | 42.42 ± 0.11 | r=3; tg tie |
| nwarps=1, VDR=4 | n/a | n/a | `n=1,k=4096` NMSE 0.0164 > 0.01; reject |

Defaults stay `nwarps=1`, `VDR=2`. Extra warps do not raise non-spec decode;
8 warps slow it. VDR=4 is not a legal ROCmI4 MMVQ ratio on this layout.
Plain AR remains HBM-bound (~42 t/s). No speculative decoding in this sweep.

## Next: measured gfx1100 pp/tg acceleration plan

The occupancy sweep above is exhausted. The follow-on work, with measured
hardware ceilings, a kernel-time decomposition, and PP/TG work packages, lives
in [rdna3-gfx1100-tg-pp-acceleration.md](rdna3-gfx1100-tg-pp-acceleration.md).

Its two load-bearing findings for this branch:

- The IU8-vs-IU4 comparison (pp2048 1090.75 vs 1380.09 t/s, +26.5%) against a
  2.0x IU4/IU8 MMA-issue ceiling ratio shows the W4A4 MMQ kernel spends ~74%
  of its time on non-MMA work, not on IU4 WMMA issue. Prefill tuning belongs
  in operand staging and the fp32 block-scale epilogue.
- Decode is at 792 GB/s of the measured 923.5 GB/s streaming ceiling, so
  `V_DOT8_I32_IU4` cannot move tg further. The remaining decode levers are
  dispatch count and KV-cache bytes.
