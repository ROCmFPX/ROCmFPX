# IU4 tensor-core path — handoff

Status as of 2026-08-20. Written for another agent (Codex / Grok / Gemini 3.7)
picking this up cold. Everything below was measured on this machine, not
inferred.

---

## 1. The one-paragraph version

gfx1151 has a **4-bit matrix instruction that runs at twice the rate of the
8-bit one**: `v_wmma_i32_16x16x16_iu4` retires in half the cycles of
`v_wmma_i32_16x16x16_iu8` at the *same* 16x16x16 shape. llama.cpp/ROCmFPX never
used it. This work wires three quant formats onto it — `Q4_0_ROCMI4`,
`Q2_0_ROCMFPX`, `Q3_0_ROCMFPX` — for **+10% to +21% prefill** at a small,
measured accuracy cost. All of it is **off by default** behind compile flags.

---

## 2. Hardware facts (verified by compiling, not assumed)

Measured throughput on gfx1151, 40 CU @ 2.9 GHz, identical instruction counts
and occupancy per kernel:

| instruction | throughput |
|---|---:|
| `v_dot4_i32_iu8` (dp4a) | 29.5 TOPS |
| `v_dot8_i32_iu4` | 58.9 TOPS |
| `v_wmma_i32_16x16x16_iu8` | 57.1 TOPS |
| **`v_wmma_i32_16x16x16_iu4`** | **110.6 TOPS** |
| `v_wmma_f32_16x16x16_f16` | 57.1 TFLOPS |

93-96% of AMD's published RDNA3 rates (IU8 512, IU4 1024 ops/clock/CU).
Achievable memory bandwidth for roofline work: **231.5 GB/s**.

Instruction availability, confirmed by compiling for each target:

| arch | instruction | K/instr | supported here |
|---|---|---:|---|
| RDNA3 `gfx1100/1101/1102` | `v_wmma_i32_16x16x16_iu4` | 16 | yes (untested, no HW) |
| RDNA3.5 `gfx1150/1151/1152/1153` | `v_wmma_i32_16x16x16_iu4` | 16 | **yes, tested on gfx1151** |
| RDNA4 `gfx1200/1201` | `v_wmma_i32_16x16x**32**_iu4` | 32 | **no** — different builtin, non-mirrored fragment layout; gated out, falls back to int8 WMMA |
| CDNA | — | — | no, uses MFMA |

**The GPU architecture gates this, not the ROCm version.** ROCm 7.14 is only the
toolchain used here.

---

## 3. Call chain

```
mul_mat_q<type, mmq_x, need_check>          ggml/src/ggml-cuda/mmq.cuh
  |- vec_dot_rocmi4_iu4_wmma                 ROCMI4  (scales per 32)
  |- vec_dot_rocmfpx_iu4_mma                 FP2/FP3 (scales per 16)
       |- mma_iu4 / mma_iu4_k16              ggml/src/ggml-cuda/mma.cuh
            |- __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32
                 |- v_wmma_i32_16x16x16_iu4
```

Weights staged by `load_tiles_rocmi4_iu4` / `load_tiles_rocmfpx_fp2_iu4` /
`load_tiles_rocmfpx_fp3_iu4`. Activations by
`quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, i4_grid=true>` in `quantize.cu`.

---

## 4. Build flags (all default 0)

```
-DGGML_HIP_ROCMI4_IU4_MMQ=ON     # Q4_0_ROCMI4 on IU4
-DGGML_HIP_ROCMFPX_IU4_MMQ=ON    # Q2_0_ROCMFPX and Q3_0_ROCMFPX on IU4
-DGGML_HIP_ROCMI4_IU4_EXACT=ON   # bit-exact 2-plane variant; also requires ROCMI4_IU4_MMQ
```

Defaults off were verified to reproduce baseline numbers exactly.

---

## 5. Measured results

### Q4_0_ROCMI4 — Qwen3.8-27B dense

| batch | baseline | IU4 | ratio |
|---:|---:|---:|---:|
| 32 | 147 | 239 | 1.62x |
| 64 | 341 | 375 | 1.10x |
| 128 | 431 | 527 | 1.22x |
| 512 | 462 | **559** | **1.21x** |
| 2048 | 455 | 552 | 1.21x |
| tg128 | 13.78 | 13.81 | 1.00x |

PPL 27B 6.0183 -> 6.3411 (+0.32). PPL 0.6B 24.2381 -> 34.1116.

### Q4_0_ROCMI4 — Qwen3.6-35B-A3B MoE (requantised from Q4_K_M, see caveat)

pp512 1557 -> 1774 (1.14x). tg128 68.2 -> 68.1 (flat).
Decode with speculation: MTP peak 80.7 -> 85.3; **DFlash n_max=4: 74.8 -> 85.9 (+14.8%)**.

### Q2_0_ROCMFPX — Qwen3.8-27B imatrix

pp512 405.7 -> **445.3** (+9.8%); pp2048 397.6 -> 439.2 (+10.5%); tg128 flat.
PPL 7.5042 -> 7.8816 (+0.377, +5.0%).

### Q2_0_ROCMFPX — Qwen3.8-27B Escha-W2 conversion

pp512 387.3 -> **440.4** (+13.7%); pp2048 379.8 -> 433.0 (+14.0%).
PPL 17.3479 -> 20.5478 (+18.4%). **Worse accuracy trade** — this conversion's
baseline PPL of 17.35 is already poor for a 27B, and fragile models degrade more
under W4A4. Prefer the imatrix FP2 model.

### Q3_0_ROCMFPX — Qwen3-0.6B (only test model available)

PPL 35.6635 -> 39.6859. pp2048 10289 -> 10611 (+3.1%, but a 0.6B is dominated by
non-matmul work — needs a larger FP3 model for a real speed figure). **Gap.**

### Downstream capability (BFCL v4, 100-case subset, Qwen3.6-35B-A3B MoE)

| arm | score |
|---|---:|
| ROCm, IU4 **on**, DFlash n=4 | 89/100 |
| ROCm, IU4 **off**, DFlash n=4 | 90/100 |

**W4A4 costs 1 case in 100 — inside binomial noise (SE ~3%).** IU4-on scored
*higher* on `simple_python` (90 vs 80). The PPL delta does not show up as
agentic capability loss. Full methodology in
`/home/caf/evals/bfcl-run-20260820-rocmi4-moe/README.md`.

---

## 6. Why it works, and the constraints

**IU4 needs BOTH operands at 4 bits.** Weights already are. Activations are
`q8_1`, so they are re-quantised onto the 4-bit grid at *quantize* time
(`i4_grid=true`), storing `c*16` so the kernel's per-byte `>>4` is lossless and
free. That is W4A4, and it is where the entire accuracy cost lives.

**Formats that fit, and why:**

| format | decoded values | fits signed nibble? |
|---|---|---|
| `Q4_0_ROCMI4` | two's-complement `[-8,+7]` | yes, natively |
| `Q2_0_ROCMFPX` | `{-4,-1,1,4}` | **yes — keeps its codebook** |
| `Q3_0_ROCMFPX` | `{0,+-1,+-2,+-4}` | **yes — keeps its codebook** |
| `Q6_0_ROCMFPX` | `[-32,31]` | no, needs 8-bit |
| `Q8_0_ROCMFPX` | `int8` | no, needs 8-bit |

FP2/FP3 give up nothing on the weight side — the loader widens each 2/3-bit code
to a nibble instead of an int8. FP6/FP8 are capped at the IU8 rate they already
achieve; **there is no IU4 win available for them.**

FP2/FP3 scale every 16 elements (`e[2]` per 32-value block), so they use
`mma_iu4_k16` (one WMMA per scale group). ROCMI4 scales every 32 and uses
`mma_iu4` (two K=16 WMMA into one accumulator).

**K-order:** a dot product is invariant under any permutation of K applied to
*both* operands. Both sides emit the byte-interleaved order
`[b+0,b+4,b+1,b+5,b+2,b+6,b+3,b+7]`, which costs ~5 ALU ops instead of ~22 for
true nibble compaction. If you touch one packer you must touch the other.

---

## 7. Pitfalls already paid for — do not repeat

1. **Do the activation rounding at quantize time, never in the kernel.**
   Truncating toward -inf in-kernel is a systematic -7.5/16 LSB bias: **PPL 1988**.
   Rounding in-kernel fixes accuracy but costs 540 -> 448 t/s.
2. **`mul_mat_q` is compiled once per `mmq_x`, and `mmq_x` equals the batch
   width.** A spill in one instantiation collapses that batch size *only*.
   Building `tile_B` in-kernel from an 8-int tile spilled 136 words at
   `mmq_x=64` and cost 2.5x *there* while pp512 looked fine. Always dump
   `vgpr_spill_count` per instantiation.
3. **Do NOT fix spills by disabling the `j0` unroll.** It zeroes every spill and
   collapses pp512 from 552 to 233. The unroll is load-bearing.
4. **Do NOT add `#pragma unroll` to the `k01` loop** (the int8 path omits it
   deliberately) — spills 14 -> 412, 3x slowdown.
5. **Occupancy is not the limiter.** LDS at `mmq_x=128` is ~30KB = 2
   workgroups/CU; capping `mmq_x` to 64 (~21KB, 3 workgroups/CU) measured
   *identical* (560.3 vs 561.0). Halving the y-tile would buy nothing — and
   `MMQ_TILE_Y_K` has 79 references. **Ruled out, do not attempt.**
6. **Amdahl.** `rocprofv3` on pp512: `mul_mat_q` is 70-76% of GPU kernel time,
   but only ~1/3 of that kernel is WMMA math; inside it we reach ~43 of 110.6
   TOPS (39%). The rest is LDS/staging traffic.
7. **Beware blind `sed`/`replace` on shared code.** A first-occurrence replace
   once landed in `vec_dot_q8_0_q8_1_mma`, which every Q8_0/ROCMFP4_FAST model
   uses. Patch by line number and re-verify the shared path.
8. **Pre-existing bug, unrelated to IU4:** the int8 kernel spills 46 words at
   `mmq_x=32`, so baseline pp32 (147 t/s) is *below* pp48 (281 t/s). Affects
   **every** quant type on this arch. Worth ~1.9x at batch 32. Not fixed.

---

## 8. Known gaps / next steps, roughly in value order

0. ~~ROCMI4 ignores imatrix~~ **DONE 2026-08-21** -- weighted scale search added
   (`rocmfpx_choose_scale_i4_weighted_mse`), **-10.1% PPL** on Qwen3-0.6B from
   BF16 (24.39 -> 21.93). Quantise from BF16/F16; requantising ROCMI4->ROCMI4
   is a no-op because same-type tensors are copied.
1. **Q4_K and Q4_0 support.** Highest reach by far — Q4_K_M is what most people
   run. `q4_K` nibbles are unsigned `0..15`, a **native unsigned iu4 operand with
   zero conversion**; `q4_0`'s -8 bias is just `n ^ 8`. Same W4A4 trade.
2. **A real FP3 speed number.** Only a 0.6B FP3 model exists here.
3. **QuaRot-style online Hadamard rotation** before activation quantisation, to
   reclaim the W4A4 PPL. QuaRot reports <=0.47 on Llama2-70B W4A4; ROCMI4 is at
   0.32 with no rotation. Hadamard machinery exists from the escha/EXL3 work.
4. **RDNA4 branch** for `v_wmma_i32_16x16x32_iu4` (double K, non-mirrored
   layout). Cannot be tested on this machine — do not write it blind.
5. **Vulkan.** ROCMI4 has no Vulkan integer-dot path: measured pp128 48 t/s vs
   ~1770 on ROCm (~37x) and tg32 20 vs 68. Vulkan runs a generic dequant path.
6. **Shrink the FP2/FP3 tile stride.** ROCMI4 went 76 -> 44 ints/row
   (`MMQ_MMA_TILE_X_K_ROCMI4`) for ~1.7% at pp512 and ~9% at batch 64. FP2/FP3
   still use the full `MMQ_MMA_TILE_X_K_Q3_K` (84) with only 32 ints of payload.

---

## 9. Reproducing

Tree: `/home/caf/ROCMFPX GITHUB/ROCMFPX MAIN CURRENT` (**this is the live tree**;
`/home/caf/ROCmFPXMAIN` is stale). Build dir
`build-rocmi4-rocm714-gfx1151`, ROCm at `/mnt/seconddrive/rocm/7.14-gfx1151`.

```bash
cd "/home/caf/ROCMFPX GITHUB/ROCMFPX MAIN CURRENT/build-rocmi4-rocm714-gfx1151"
ninja llama-bench llama-perplexity
./bin/llama-bench -m <model.gguf> -ngl 999 -dev ROCm0 -fa on -p 512,2048 -n 128 -r 4
./bin/llama-perplexity -m <model.gguf> -f /home/caf/strix-fp4/data/wikitext-2-raw/wiki.test.raw \
    -ngl 999 -dev ROCm0 -fa on -c 512 --chunks 25
```

Confirm the instruction actually landed:

```bash
CMD=$(ninja -t commands ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/template-instances/mmq-instance-q2_0_rocmfpx.cu.o | tail -1)
eval "$(echo "$CMD" | sed 's| -o [^ ]*\.o| -S --cuda-device-only -o /tmp/x.s|; s| -c | |')"
grep -oE 'v_wmma_i32_16x16x16_iu[48]' /tmp/x.s | sort | uniq -c
grep -oE 'vgpr_spill_count: *[0-9]+' /tmp/x.s | grep -oE '[0-9]+$' | sort -n | tail -1
```

**Accuracy is the gate. Always measure perplexity — coherent greedy output proves
nothing** (an earlier `-ffast-math` incident degraded NVFP4 61x while answers
still looked fine).

Test models used here: `Qwen3-0.6B-Q4_0_ROCMI4.gguf`,
`Qwen3.8-27B-Q4_0_ROCMI4.gguf`,
`Qwen3.8-27B-ROCmFP2-VISIONMIX-MTPBLOCKQ4-IMATRIX64.gguf`,
`Qwen3.8-27B-Escha-W2-ROCmFP2.gguf`, and a Qwen3.6-35B-A3B MoE requantised to
ROCMI4 **from Q4_K_M** — that last one is a double quantisation, valid for speed
and for A/B comparison, **not** for absolute quality.

---

## 10. Files touched (all uncommitted)

- `ggml/src/ggml-cuda/mma.cuh` — `mma_iu4`, `mma_iu4_k16`, `tile<16,2,int>` support + `load_ldmatrix`
- `ggml/src/ggml-cuda/mmq.cuh` — flags, IU4 loaders, IU4 vec_dots, traits, `MMQ_MMA_TILE_X_K_ROCMI4`
- `ggml/src/ggml-cuda/quantize.cu` — `i4_grid` activation path + dispatch
- `ggml/rocmfpx/ROCMI4.md` — format + results writeup
- `ggml/rocmfpx/IU4-HANDOFF.md` — this file

ROCMI4 itself (the format) was also never committed — ~23 modified files plus
`mmq-instance-q4_0_rocmi4.cu`. `git status` before assuming anything is upstream.
