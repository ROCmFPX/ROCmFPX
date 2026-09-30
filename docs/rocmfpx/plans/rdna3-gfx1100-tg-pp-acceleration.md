# Plan: RX 7900 XTX (gfx1100) decode/prefill acceleration for Qwen3.8-27B Q4_0_ROCMI4

Branch: off `feature/rdna3-w4a4-iu4-mmq`
Target host: RX 7900 XTX (`gfx1100`), ROCm 10 container (HIP 7.15)
Model: `/home/ghazni/models/rocmfpx/cafonez/Qwen3.8-27B-ROCmI4/Qwen3.8-27B-Q4_0_ROCMI4.gguf`

Status: partially executed. Six work packages measured, one landed. Every
number below was measured on this host, not extrapolated from vendor peaks.

**Revised 2026-09-19** (Revision 2, executed same day). Four packages ran end
to end with the gates below: P-1 refuted the bank-conflict theory and killed
P-3 before any layout code was written; P-2 landed at **+7.2% pp2048**
(1434.20 -> 1537.27 +/- 0.40, token-exact); P-1c partitioned the remaining
kernel into 343/190/137 us; P-8 (k-loop double buffering, built to recover the
staging bucket) measured -11.4% and was reverted with the residency mechanism
identified. P-4 was refuted by byte arithmetic without being built. Current
state of the branch: pp2048 1535-1537, tg128 42.0-42.5, tree contains only the
P-2 kernel change plus this document and the new probe. Remaining open:
P-6 (+8% pp, its own project), P-7 (+2-3% pp), P-5 (+3-6% tg).

**Outcome so far**

| area | result |
|---|---|
| prefill | **+3.2% to +3.8%** landed (`pp2048` 1380-1389 -> **1432.10 +/- 1.24**, `tg128` unchanged at 42.53), from 16-byte LDS operand loads. Tile-config retune (PP-2) found no further gain - the shipped shape was already best |
| decode | **no headroom found.** Both candidate levers were measured and closed: KV quantization costs 5-10%, and MMVQ already runs at 94% of what the on-disk block layout permits |

Four hypotheses were tested and refuted along the way - epilogue arithmetic,
ILP/occupancy, KV quantization, and tile reshaping. That is why the projections
in this document fall as you read down it. **The original +50% to +80% prefill
and +18% to +30% decode targets are withdrawn as unsupported by measurement.**

The honest summary for this model on this GPU: prefill had one real, modest
win; decode was already within ~10% of its floor. Getting materially more
decode throughput on a 27B dense-ish model at 4.25 bits requires reading fewer
bytes per token, which is exactly what speculative decoding does and what the
constraints exclude. Revision 2 keeps that decode conclusion and adds one
quantified prefill mechanism that the probes could not see because the probes
never paid the conflict (see Operand path, amended below).

## Goal

Raise prefill (`pp`) and non-speculative decode (`tg`) throughput for the
hybrid Qwen3.8-27B ROCmI4 model on discrete RDNA3, using instruction-level
hardware capability and kernel-efficiency work rather than algorithmic change.

Hard constraints:

- No speculative decoding. The GGUF carries an MTP block (`blk.64.nextn.*`,
  225.7 MB) that is deliberately unused; it stays unused.
- The on-disk `Q4_0_ROCMI4` format does not change.
- ROCmI4 today is either exactly-int8 or the opt-in lossy W4A4 profile. Any
  change to the numerics must go through the correctness plan below.

## Measured baselines

`llama-bench -ngl 999 -fa 1`, this host, build `build-rdna3-w4a4`
(`GGML_HIP_ROCMI4_W4A4=ON`, `GGML_HIP_FORCE_MMQ=ON`, `GGML_HIP_GRAPHS=ON`):

| test | t/s |
|---|---:|
| pp512 | 1332.45 +/- 173.23 |
| pp2048 | 1380.09 +/- 0.31 |
| tg128 | 42.46 +/- 0.19 |
| pp512 @ d8192 | 1048.32 +/- 124.58 |
| tg128 @ d8192 | 41.69 +/- 0.60 |
| pp512 @ d32768 | 660.99 +/- 51.85 |
| tg128 @ d32768 | 38.68 +/- 0.54 |

Same model, exact-int8 build (`build-rdna3-rocm10`, `GGML_HIP_ROCMI4_W4A4=OFF`):

| test | t/s |
|---|---:|
| pp2048 | 1090.75 +/- 1.59 |
| tg128 | 41.21 +/- 0.77 |

Two measurement notes that matter for later comparisons:

- **Use `pp2048`, not `pp512`.** `pp512` swings by +/-200 t/s between runs
  (1332 +/- 173, 1277 +/- 214); `pp2048` is stable to +/-0.3. Any pp claim
  must quote `pp2048` or a repeated `pp512`.
- W4A4 buys **+26.5% pp** and **+3.0% tg** over exact int8 today.

## Measured hardware ceilings

Microbenchmarks written for this plan and kept in `ggml/rocmfpx/probes/`,
compiled with `hipcc -O3 --offload-arch=gfx1100`, run on the same GPU:

| ceiling | measured | repeated |
|---|---:|---:|
| streaming read (uint4, grid-stride, unroll 4, 4 GB buffer) | **923.5 GB/s** | 928.6 GB/s |
| `V_WMMA_I32_16X16X16_IU4` issue | **282.4 TOPS** | 277.6 TOPS |
| `V_WMMA_I32_16X16X16_IU8` issue | 146.1 TOPS | - |
| `V_DOT8_I32_IU4` (VALU) issue | 77.3 TOPS | - |
| kernel dispatch floor (HIP graph) | **~2.7 us/kernel** | 2.76-3.01 us |
| kernel dispatch floor (plain launches) | 3.0 - 4.6 us/kernel | 3.17-4.48 us |

Run-to-run spread on the two load-bearing figures is 0.5% (bandwidth) and 1.7%
(WMMA issue), so neither is a knife-edge measurement. Build and run with:

```sh
cd ggml/rocmfpx/probes
hipcc -O3 --offload-arch=gfx1100 -o bwprobe bwprobe.hip && ./bwprobe 4 1536 256 4
hipcc -O3 --offload-arch=gfx1100 -o wmmaprobe wmmaprobe.hip && ./wmmaprobe 20000
```

923.5 GB/s is 96% of the 960 GB/s board spec, so the memory system is already
at its practical limit; there is no clock or configuration headroom hiding
there. IU4 measures exactly 2.0x IU8, which is the ratio the rest of this plan
leans on.

### Operand path: the 282 TOPS figure is not reachable by any real kernel

`wmmaprobe` keeps its operands in registers, so it measures the *issue* ceiling
only. A kernel that must fetch A and B from LDS every K step pays an operand
path, and that path is much narrower. Measured with `wmmalds2` / `wmmalds3`,
one `mma_iu4` = 2 builtin calls:

| operand source (per `mma_iu4`) | best TOPS | probe |
|---|---:|---|
| registers, no LDS | 270 - 282 | `wmmaprobe` |
| 2 x 16-byte LDS loads | **230 - 242** | `wmmalds2` |
| 2 x 8-byte LDS loads | 145 - 180 | `wmmaprobe` lineage |
| **4 x 8-byte LDS loads** | **152 - 170** | `wmmalds3` |

The last row is what the real kernel actually does. `mma.cuh`'s RDNA3
`load_ldmatrix` for `tile<16,4,T,dl>` emits **two 8-byte LDS loads per operand**:

```cpp
ggml_cuda_memcpy_1<8>(t.x + 0, xs0 + t.get_i(0)*stride + 0);
ggml_cuda_memcpy_1<8>(t.x + 2, xs0 + t.get_i(0)*stride + 2);
```

so each `mma_iu4` costs four 8-byte LDS loads instead of the two 16-byte loads
the tile layout would allow. **8-byte LDS loads cap WMMA IU4 at ~170 TOPS;
16-byte loads reach ~240.** That is where the prefill headroom is.

Also measured in the same probes: neither ILP nor occupancy moves the needle.
Sweeping 1, 2, 4 and 8 independent accumulator chains gives 154.7 -> 180.0 TOPS
(+16%) for 8-byte operands, and essentially flat for 16-byte operands. Sweeping
blocks from 192 to 3072 is likewise flat. The operand **load width** dominates
both.

### Revision 2 amendment: the ~240 TOPS figure is conflict-free by construction - DERIVATION SUPERSEDED BY MEASUREMENT

`wmmalds2` gives each lane the next consecutive 16-byte chunk. The real kernel
reads row-major at `sram_stride = 44` (x) and `MMQ_TILE_Y_K = 36` (y) dwords,
one 16-byte read per row, 16 rows per wave ([mmq-vec-dot.cuh:1279, 1286]).
Any stride keeping the 16-byte loads legal is a multiple of 4 dwords, so the
16 rows cover at most 8 of the 32 LDS banks - a structural 2-way conflict for
any legal stride, any row permutation, any legal pad. That derivation is
arithmetically correct and **performance-irrelevant**: wmmalds4 measured the
strided pattern at 227-236 TOPS, equal to the conflict-free control. A 2-way
conflict signature evidently does not slow `ds_read_b128` on gfx1100 the way
dword-grained models predict, and at the probe's ~920 GB/s the LDS has
headroom to absorb it. The derivation is kept for the record; the conclusion
it pointed at (P-3, k-major staging) was derived, then killed by P-1 before
any layout code was written. Read the
wmmalds4 table in P-1 for the measured state.

What survives from this line of inquiry: the operand-path question is closed
by measurement - 2x16-byte reads are already the right width, and no pad,
permutation or staging transpose buys anything. The kernel's 592 us
epilogue-free floor is NOT operand-load-pattern cost. P-1c attributes it.

## Model geometry (drives all the arithmetic)

Parsed from the GGUF:

- 14.523 GB of tensors; 14.298 GB excluding the unused MTP block.
- 26.91e9 weights at 17 bytes / 32 weights = 4.25 bits.
- 65 blocks, 64 in use: **16 full-attention** (every 4th layer) and **48
  gated-delta-net (GDN) linear-attention** layers.
- hidden 5120, FFN 17408, 24 q heads / 4 kv heads, head_dim 256.
- KV cache: 16 layers x 2 x 4 x 256 x 2 B = **64 KiB per token of context**.
- FFN weights are 64.6% of all bytes.

## Where the time actually goes

### Prefill (kernel trace, per `pp512` pass, ~384 ms of kernel time)

| component | ms | share |
|---|---:|---:|
| `mul_mat_q<Q4_0_ROCMI4,128,*>` (MMA) | 302 | 78.7% |
| `gated_delta_net_cuda<128,false,false>` | 33 | 8.6% |
| everything else (quantize, norms, silu, concat, FA, rope) | 49 | 12.7% |

MMQ sustained rate: 512 x 26.91e9 = 13.78e12 MACs in 302 ms = **45.6 TMAC/s =
91.3 TFLOPS** (ggml's 2-op convention) = **32% of the 282 TOPS IU4 ceiling**.
Isolated best case (`test-backend-ops perf`, m=4096 n=512 k=14336) is
**78.78 TFLOPS = 28%** of that ceiling.

Weight bytes / MMQ time = 14.3 GB / 302 ms = 47 GB/s. **DRAM is at 5% of
peak during prefill.** Prefill is not bandwidth-bound.

### The decisive measurement

If MMQ were limited by IU4 MMA issue, doubling the MMA rate (IU8 -> IU4)
would nearly double prefill. It does not:

```
pp2048 exact IU8 : 1090.75 t/s
pp2048 W4A4  IU4 : 1380.09 t/s   -> +26.5%
MMA ceiling ratio IU4/IU8        -> 2.0x
```

Solving `T_iu8/T_iu4 = 1/((1-x)/2 + x) = 1.265` for the non-MMA fraction `x`:

- **~58% of the exact-path MMQ pipeline is non-MMA work.**
- **~74% of the W4A4 MMQ pipeline is non-MMA work** (operand staging,
  `ldmatrix` LDS reads, and the fp32 block-scale epilogue).

Cross-check: the MMA-issue portion of the W4A4 MMQ run (26% of 302 ms =
~78 ms) is within noise of the 98 ms implied by the *register-fed* ceiling.
Note that this comparison is generous - see "Operand path" above, where the
register-fed ceiling is shown to be unreachable by any kernel that must fetch
operands from LDS. Read against the realistic 16-byte-operand ceiling the same
MMA work would take ~124 ms, and the "non-MMA" share shrinks accordingly. The
74% figure is therefore an upper bound on non-MMA work, and the IU4 WMMA work
is **not** confirmed to be at any hardware limit.

The leading hypothesis for that 74% was the fp32 block-scale epilogue in
`ggml_cuda_mmq_vec_dot_rocmi4_w4a4_wmma`: per `mma_iu4` call (2 WMMA
instructions, 8192 MACs) each lane executes 8 x (`LDS load dA`, `imul *16`,
`int->float`, 2 x `fmul`, `fadd`) = ~48 instructions against 2 MMA
instructions, and RDNA3 WMMA is documented as "internally using the DOT
instructions" (ISA 7.9), so they compete for issue rather than running on a
separate pipe.

**That hypothesis was tested by ablation and is largely refuted.** Three
builds of the same kernel, measured on `m=4096 n=512 k=14336`:

| variant | us/run | TFLOPS | vs baseline |
|---|---:|---:|---:|
| baseline (full epilogue) | 763.30 | 78.78 | - |
| B: scale LDS loads kept, multiply chain removed | 714.86 | 84.11 | -6.3% |
| A: no scale loads, no scale arithmetic | 592.14 | 101.55 | -22.4% |

Decomposition of the baseline:

| phase | us | share |
|---|---:|---:|
| MMA issue + A/B staging + loop/stall overhead | 592.1 | **77.6%** |
| per-element scale LDS gathers | 122.7 | 16.1% |
| per-element multiply/convert chain | 48.4 | 6.3% |

The entire epilogue is 22.4%, not the dominant cost. The multiply chain that
looked most expensive in the instruction count is only 6.3%.

Worse, variant A - which does *no* epilogue work at all - still reaches only
**101.55 TFLOPS = 36% of the 282 TOPS register-resident issue ceiling**.
Against the ideal MMA-issue time for this shape (60.13 GFLOP / 282 TOPS =
213 us), even the epilogue-free kernel is 2.8x slower than ideal.

**Revised conclusion: the bottleneck is the MMA + operand-staging path itself,
not the scale application.** Both are implemented in the same non-unrolled
`k01` loop, so the 592 us floor is most likely exposed latency rather than
issue throughput. See PP-1 for the redirected work.

### Revision 2 amendment: the scale gathers are redundant by J/16

The ablation decomposition stands, but one of its costs is not mandatory. In
`ggml_cuda_mmq_vec_dot_rocmi4_w4a4_wmma` ([mmq-vec-dot.cuh:1298]) the gather
`dA = x_df[i*sram_stride + k0/QI8_0]` sits inside the `j0` loop, while `i`
depends only on `l` and the warp row - never on `j0`. Runtime `J_best` picks
J=128 for this model (the heuristic at [mmq.cuh] minimizes j-tiles), so the
`j0` loop runs `J/(ntx*tile_C::J) = 8` times and re-gathers the same 8 `dA`
values 8 times per k-step. The measured 122.7 us gather share is dominated by
loads the compiler cannot legally hoist across the shared-memory barrier
structure of the enclosing loop (tile_x is re-staged every iteration; the
gather addresses are loop-invariant but the compiler does not know x_df
contents are unchanged, and more importantly the loads sit under `#pragma
unroll` j0/n loops whose live ranges then multiply).

Hoisting `dA` (8 floats, +8 VGPRs) above the `j0` loop removes 7/8 of the
gather traffic. Folding the `*16` into it is exact in fp32: `16*dA` is a
power-of-two exponent shift, and `float(C*16)*dA == float(C)*(16*dA)` because
both round the same real product of integers times fp32 constants. This is
work package P-2 and it is gated by Tier 2 token-exactness, not Tier 3.

The ablation was a disposable local build of `mmq-vec-dot.cuh`; the source was
restored byte-identical afterwards and the restored build re-measured to
confirm the baseline.

### Decode (kernel trace, per token, 23.55 ms at 42.46 t/s)

| component | dispatches | ms | notes |
|---|---:|---:|---|
| `mul_mat_vec_q<ROCMI4,1,*>` | 436 | 17.20 | streams ~13.62 GB = **792 GB/s** |
| small kernels | 1414 | 2.75 | see below |
| inter-kernel gap | - | ~3.6 | 1850 dispatches x ~2 us |

792 GB/s is **86% of the 923.5 GB/s achievable**. The matvec itself is close to
optimal; decode is genuinely bandwidth-bound, and `V_DOT8_I32_IU4` already
removed the unpack + 2x DP4A work (it moved tg by ~0).

Small-kernel breakdown per token (top entries):

| kernel | count | ms |
|---|---:|---:|
| `quantize_q8_1` | 436 | 0.51 |
| `rms_norm_f32<1024>` | 130 | 0.43 |
| `k_get_rows_float_vec` | 48 | 0.29 |
| `gated_delta_net_cuda` | 48 | 0.21 |
| `k_bin_bcast` | 97 | 0.16 |
| `l2_norm_f32<32>` | 97 | 0.15 |
| `rms_norm_f32<256>` | 81 | 0.13 |
| `cpy_scalar` | 65 | 0.12 |
| `rope_multi` | 32 | 0.11 |
| `flash_attn_tile` | 16 | 0.10 |

`quantize_q8_1` is launched exactly once per matvec (436 = 436).

### Decode is within 1.6x of a hard wall

Per token the model must read 13.62 GB of weights. At the measured 923.5 GB/s
that is 14.75 ms, i.e. **67.8 t/s absolute maximum**, before any real-kernel
inefficiency. Measured 42.46 t/s is 63% of that. There is no algorithmic route
around this without spec decoding, which is excluded.

Long-context behaviour is confirmed by the same model. Using the measured
MMVQ streaming rate (792 GB/s) and the measured fixed cost (2.75 ms of small
kernels + 3.6 ms of gaps = 6.35 ms):

```
bytes/token  = 13.62 GB + depth * 65536 B
ms/token     = bytes/token / 792 GB/s + 6.35
```

| depth | bytes/token | predicted | measured |
|---|---:|---:|---:|
| 0 | 13.62 GB | 23.55 ms / 42.5 t/s | 42.41 t/s |
| 8192 | 14.16 GB | 24.2 ms / 41.3 t/s | 41.69 t/s |
| 32768 | 15.77 GB | 26.3 ms / 38.1 t/s | **38.68 t/s** |

Good to ~1% across the range, so it can be trusted for sizing.

## Opportunity sizing (honest)

**Prefill.** MMQ is 78.7% of the time. Applying the ablations and probe
ceilings as relative reductions on that 302 ms:

| scenario | basis | pp512 t/s | vs today |
|---|---|---:|---:|
| today | measured | 1333 | - |
| epilogue removed in full | measured (PP-0 ablation) | **1620** | +22% |
| operand loads widened to 16 B, at probe ceiling | modelled | 2000 | +50% |
| + chunked GDN and PP-4 | modelled | 2150 | +61% |

Only the first two rows are measurements. Row 3 assumes the real kernel reaches
the efficiency of a four-line probe loop, which it will not - it is a bound,
not a forecast.

**Honest current statement: pp512 +22% is supported by measurement; +30% to
+40% is plausible if the operand-load widening lands well.** This is revised
down from the +50% to +80% estimated before any of this was measured, and again
from the +40% to +55% estimated after PP-0 alone. Two candidate mechanisms
(epilogue arithmetic, ILP) have been measured and refuted; one (8-byte operand
loads) is measured and is the only remaining quantified lever.

The `pp2048` equivalent runs somewhat higher.

**Decode.** Both decode levers have now been measured and closed (TG-1, TG-3),
so this is no longer a projection:

| step | ms/token | t/s |
|---|---:|---:|
| today | 23.55 | 42.5 |
| MMVQ at the layout-limited ceiling (840 GB/s) | 17.20 -> 16.21 | |
| small kernels + gaps (unchanged) | 6.35 | |
| best case | 22.56 | **44.3 (+4.3%)** |

The 923.5 GB/s streaming figure does not apply to decode: MMVQ must read the
17-byte on-disk block, which by itself costs 9.3%, and MMVQ already achieves
94% of what that layout permits. The only remaining decode lever of any size is
TG-2 (1414 small kernels per token), worth perhaps +5% to +8%, and it is
kernel-fusion work with no measured bounds yet.

**Bottom line for decode: ~42.5 t/s is within roughly 10% of what this model,
this format and this GPU permit without speculative decoding.** The original
plan's +18% to +30% decode target is withdrawn. It assumed KV quantization
would help at long context, and measurement shows it costs 5-10% instead.

The server's existing `-ctk f16 -ctv f16` is already the fastest KV setting
measured; there is no configuration change to make.

## Work packages

Ordered by measured payoff. Each WP states its own acceptance test.

### PP-0 - Ablation to locate the MMQ bottleneck (DONE)

Completed. Three disposable local builds of `mmq-vec-dot.cuh` on
`m=4096 n=512 k=14336`:

| variant | us/run | TFLOPS |
|---|---:|---:|
| baseline | 758.67 - 763.30 | 78.78 - 79.26 |
| B: scale loads kept, multiply chain removed | 714.86 | 84.11 |
| A: no scale loads, no scale arithmetic | 592.14 | 101.55 |

Outcome: the epilogue hypothesis is largely **refuted**. The epilogue is 22.4%
of the kernel, and the epilogue-free kernel still only reaches 36% of the IU4
issue ceiling. Source was restored byte-identical (md5 verified) and the
restored build re-measured at 79.26 TFLOPS, confirming a clean experiment.

The harness is kept in `ggml/rocmfpx/probes/ablate.py` (`save` / `A` / `B` /
`restore`). It round-trips the source by md5 and refuses to patch if the anchor
has moved. Both variants are numerically wrong and must never be merged.

### PP-1 - Attack the MMA + operand-staging path (redirected)

PP-0 moved the target here, and the operand probes then narrowed it further.
Item 1 has since been implemented and measured; the rest remain open.

**Item 1 DONE - 16-byte operand loads, landed and measured.** Added
`load_ldmatrix_16` in `mma.cuh` (a single 16-byte copy, with a comment stating
the alignment precondition and why it must not be used for Q6_K), and switched
the ROCMI4 W4A4 dot to it for both operands. Alignment was verified from the
real values: `sram_stride(ROCMI4) = 44`, `MMQ_TILE_Y_K = 36`, `QI8_0 = 8`, so
`kp = k0/2` steps by 4; `get_i(0)` is `threadIdx.x % 16` and does not depend on
`l`, so the two 8-byte copies are contiguous and the substitution is exact.

| measurement | before | after |
|---|---:|---:|
| `perf` m=4096 n=512 k=14336 | 758.67 - 763.30 us | 727.61 - 739.31 us |
| same, TFLOPS | 78.78 - 79.26 | **81.33 - 82.64** |
| `pp2048` | 1380.09 - 1388.75 | **1425.44 +/- 0.93** |
| `tg128` | 42.46 +/- 0.09 | 42.01 +/- 0.19 (unaffected) |
| `test-backend-ops` MUL_MAT q4_0_rocmi4 | OK | OK |

**Realised prefill gain: +2.9% to +3.3%.** The probe model predicted a ~1.4x
operand-path ceiling, and it did not transfer: the operand loads are a much
smaller share of the real kernel than of the probe loop. This is a real,
validated, correctly-scoped improvement - but it is an order of magnitude
smaller than the probe suggested, and the lesson is that probe-loop ceilings
bound, they do not forecast.

The change is scoped to the W4A4 ROCMI4 dot only. Extending it to the other
quant types is possible but must be per-layout: `Q6_K`'s `sram_stride` is 79,
which is not a multiple of 4, so odd rows would load misaligned there.

Remaining in PP-1 - superseded by Revision 2 packages P-1/P-2/P-8, which
replace items 2-4 with mechanisms that name a cause:

2. ~~**Explain the residual.**~~ Explained by P-1c: 343 us MMA+epilogue,
   190 us staging+barriers (already partly hidden by co-residency), 137 us
   operand loads. The conflict explanation once proposed here was killed by
   P-1.
3. ~~**Bank conflicts and SRAM strides.**~~ Resolved by derivation, then
   measured: every legal row-major stride conflicts at least 2-way on paper,
   and wmmalds4 shows that costs nothing for `ds_read_b128` on gfx1100. No
   pad, no layout change warranted.
4. ~~**The epilogue itself (secondary).**~~ Landed as P-2 (+7.2% pp2048,
   token-exact). The real redundancy was 2x on the gathers (the j0 loop
   re-reads them once), not 7/8 as first estimated here - corrected in P-2.

Refuted by measurement - do not retry without new evidence:

- **ILP / unrolling the `k01` loop.** Sweeping 1 -> 2 -> 4 -> 8 independent
  accumulator chains moves an 8-byte-operand loop from 154.7 to 180.0 TOPS
  (+16%) and is flat for 16-byte operands. The non-unrolled single chain is
  not what is costing the time.
- **Occupancy.** Sweeping 192 -> 768 -> 3072 blocks is flat in both probes.
  The kernel is compiled at 224 VGPRs, but residency is not the lever.

Acceptance: `test-backend-ops perf -o MUL_MAT -p type_a=q4_0_rocmi4` TFLOPS
per change, then `pp2048`. Each change is kept only if it wins on both, and
none of them may change numerics outside the existing W4A4 NMSE bound.

### PP-2 - RDNA3.0 MMQ tile configuration (DONE, no change made)

Swept the tile shape in `ggml/rocmfpx/rocmfpx_mmq_rdna3.cuh`. `nthreads` is
tied to `I` (WMMA `rows_per_warp()` is 16, so `I = nthreads/2`); the variants
below hold that relation:

| nthreads | occupancy | I | perf TFLOPS | pp2048 |
|---:|---:|---:|---:|---:|
| **128** | **2** | **64** | 81.90 | **1431.24 +/- 1.13** (shipped) |
| 256 | 2 | 128 | 82.29 | 1406.96 +/- 1.06 |
| 64 | 2 | 32 | 60.85 | 1159.76 +/- 1.20 |
| 128 | 4 | 64 | 82.34 | 1428.24 +/- 0.89 |
| 128 | 1 | 64 | 82.59 | 1429.50 +/- 1.08 |

**The shipped configuration is already the best measured.** No change made.
`occupancy` 1/2/4 are within run-to-run noise of each other (+/-1.1 t/s), so
that field is not a lever. The 32-row tile is 19% worse, which confirms the
64-row tile is the right granularity for four WMMA warps.

Two notes worth keeping:

- **The synthetic perf shape misranks the configs.** `nthreads=256, I=128` has
  the *second-best* perf TFLOPS (82.29) but the worst `pp2048` of the viable
  configs (1406.96). `m=4096 n=512 k=14336` is one point in a wide shape
  distribution; judge tile changes on `pp2048`, and treat perf TFLOPS as a
  diagnostic only. This also means the `test-backend-ops perf` deltas quoted
  elsewhere in this document are directional, not predictive.
- Extending the 16-byte operand load from PP-1 to other quant types is still
  open, but must be per-layout: `Q6_K`'s `sram_stride` is 79.

### PP-3 - Chunked GDN prefill (sized, NOT implemented - see assessment)

`gated_delta_net_cuda` iterates tokens sequentially
(`for (int t = 0; t < n_tokens; t++)`) and carries an explicit
`//TODO: Add chunked kernel for even faster pre-fill`. It costs 48 launches x
681 us = 33 ms per pp512 pass, about 8.9% of prefill.

**Measured scaling** (`test-backend-ops perf -o GATED_DELTA_NET`, head_count=32,
n_seqs=1, default kernel):

| n_seq_tokens | us/run | ratio |
|---:|---:|---|
| 64 | 108.60 | - |
| 256 | 322.37 | 2.97x for 4x tokens |
| 512 | 615.97 | 1.91x for 2x |
| 1024 | 1206.45 | 1.96x for 2x |

Above ~256 tokens the cost is **exactly linear in token count** with a marginal
cost of ~1.15 us/token. That is the signature of the serial scan, and it rules
out occupancy or launch overhead as the cause. For reference the same op does
about 1.3e10 MACs across a pp512 pass, which at any reasonable FLOP rate should
be well under 1 ms - it takes 33 ms.

**Projected value if a chunked kernel reached ~4x fewer effective serial
steps: roughly +8% on `pp512`** (33 ms -> ~5 ms off a ~371 ms pass). This is the
largest single remaining item in the plan.

**Assessment: this is a kernel-development project, not an experiment, and it
was deliberately not started here.** Reasons, stated so the next person can
disagree with them explicitly:

- No chunked GPU implementation exists to port. The CPU backend has
  `ggml_compute_forward_gated_delta_net_one_chunk` in `ggml-cpu/ops.cpp` as a
  reference formulation, and `ssm-scan.cu`'s `ssm_scan_ssd_f32_cuda` shows the
  chunked structure the project already uses for Mamba-2, but neither is
  droppable into the HIP kernel.
- The chunked delta rule needs the WY/triangular-solve structure: intra-chunk
  causal attention, a chunk-local C x C inverse or solve, and an inter-chunk
  state recurrence. Getting this subtly wrong produces fluent-looking wrong
  output rather than a crash, which is the worst failure mode for an attention
  kernel.
- Budget reality: the payoff is +8% on prefill only. Starting it without the
  time to validate would risk exactly the outcome the constraints forbid.

**Required before this ships** (unchanged from the original acceptance):

- A/B the chunked and sequential kernels at the state and output level on
  random inputs across n_tokens = 1, 64, 256, 512, 1024, with the tolerance
  derived from fp32 accumulation-order analysis *before* the first run, not
  tuned until it passes.
- `test-backend-ops` GDN cases including PP-512.
- Tier 2 (T=0 token-exact) can NOT be used here, because the change reorders
  fp32 accumulation; use Tier 3 (perplexity/KL band) instead.
- `pp2048` improvement, plus a `GGML_CUDA_DISABLE_GDN_CHUNKED`-style switch
  following the existing `GGML_CUDA_DISABLE_SSD` rollback convention.

### PP-4 - Trim the remaining prefill overhead

49 ms/pass across ~2400 dispatches, dominated by `quantize_mmq_q8_1` (992
launches per trace), `unary_gated_op` silu (224 x 61 us), `concat_non_cont`
(96 x 85 us), and `rms_norm_f32<256>` (160 x 43 us). Lower priority than
PP-1/PP-2 but cheap: fusion here is mechanical.

Acceptance: `pp2048` improves; per-op numerics unchanged.

### TG-1 - MMVQ streaming efficiency (resolved, no meaningful headroom)

792 GB/s measured. The ROCMI4 block is 17 bytes (16-byte `qs` field + 1-byte
UE4M3 scale, `static_assert`ed to have no padding), so block `k` starts at byte
`17k` and a 16-byte load of `qs` is never 16-byte aligned.

Measured with `ggml/rocmfpx/probes/layoutprobe.hip` (4 GB of blocks, DRAM
resident, same access shape MMVQ uses):

| layout | achieved |
|---|---:|
| 16-byte blocks, aligned loads | 918.3 GB/s |
| **17-byte blocks, load at `17k+1`** | **840.1 GB/s** |

So the on-disk layout itself costs 9.3% of bandwidth, and MMVQ at 792 GB/s is
running at **94% of the ceiling that layout permits**. The remaining 6% is not
worth chasing: recovering all of it would take tg from 42.5 to about 44.4 t/s
(+4.5%), and no kernel gets 100% of a synthetic streaming number.

The layout cannot be changed - it is the GGUF contract - and a repacked 16-byte
in-VRAM copy does not fit alongside the 14.5 GB model in 24 GB.

Do not reopen this without a new mechanism. MMVQ warp-count and vector-ratio
tuning was already swept on this branch (nwarps 4 ties, 8 regresses, VDR 4 is
numerically illegal); the tuning knobs were not merged.

Revision 3 (2026-09-20, exclusive-GPU session; probe variants live in
`ggml/rocmfpx/probes/layoutprobe.hip`): the table above mixes byte conventions.
On true bytes touched (17 per block) the rates are: 16-byte misaligned loads
890.8 GB/s, the real MMVQ shape (4 misaligned 4B loads + scale byte) 890.5
GB/s, aligned cooperative loads staged through LDS with funnel-shift realign
911.6 GB/s, aligned 16-byte control 918.4 GB/s. So the 17-byte layout costs
about 3% on true bytes, not 9.3%, and a cooperative-realign kernel recovers
only ~2.4% of streaming rate: NO-GO (LDS roundtrip, barriers, and P-8-style
residency risk for ~1% end to end).

The corrected denominator puts MMVQ at 792 GB/s = ~89% of what its own load
shape sustains, not 94%. A second refutation closes that gap as structural:
a manually double-buffered, prefetch-ahead k-loop (template tag, runtime env
`GGML_CUDA_DISABLE_ROCMI4_MMVQ_PIPELINE`, bit-exact three-way greedy decode,
Tier 1 clean in both states) measured **-3.3% tg128 at every depth**
(44.47 -> 42.97 t/s at d0, noise floor 0.2%) and was reverted. Code-object
metadata rules out the register-pressure theory: VGPR 15 vs 23, no spills,
occupancy unaffected on either instantiation; the cost is the larger loop body
itself (extra address math and guarded tail selects). With nwarps=4 already tying, the
kernel is not thread- or latency-starved either; the residual ~11% versus the
synthetic probe is the cost of 436 small tensors per token read as 2.7-9.2 KB
rows by 32-thread blocks, versus one 4 GB sequential stream in the probe.

Fresh HEAD baseline this session (tg128, -r 10): 44.36 +/- 0.20 (d0),
43.60 +/- 0.16 (d8192), 40.23 +/- 0.14 (d32768). The serve-config lever
(-b 2048 -ub 1024, the measured +10.2% pp2048) is now applied in
`scripts/serve-qwen38-rocmi4-9001.sh`. TG kernel-body work stays closed;
remaining decode upside is dispatch-side (P-5, WP5).

### TG-2 - Cut the 1414 small kernels per token (superseded by P-4 + P-5)

At ~2.7 us of dispatch floor each, this stream costs ~3.8 ms/token of dispatch
alone against 2.75 ms of execution. Revision 2 splits this into P-4 (act
quantization into MMVQ) and P-5 (GDN elementwise fusion), each with its own
gate; the sizing above (+5% to +8%) is replaced by the per-package estimates.

### TG-3 - KV cache quantization (REFUTED, do not ship)

This was the plan's headline decode lever, predicted at +12% (q8_0) and +20%
(q4_0) at 80k context from the 64 KiB/token KV figure. **Measured, it is
backwards.** `llama-bench -p 0 -n 128 -d 0,32768 -r 2`:

| KV type | d0 | d32768 |
|---|---:|---:|
| f16 | **42.54 +/- 0.26** | **38.77 +/- 0.54** |
| q8_0 | 41.21 +/- 0.25 (-3.1%) | 36.89 +/- 0.55 (-4.8%) |
| q4_0 | 40.41 +/- 0.26 (-5.0%) | 35.05 +/- 0.51 (-9.6%) |

KV quantization is slower at every depth, and the penalty *grows* with context,
which is the opposite of the byte-count model. The byte model was not wrong
about the bytes: it predicted 38.1 t/s at d32768 for f16 and measured 38.77.
It was wrong about the marginal cost - dequantizing the KV inside the attention
kernel costs more than the bytes saved.

Do not enable KV quantization on this model/hardware for decode. The server
currently runs `-ctk f16 -ctv f16`, which is the correct setting. Re-open only
with a fundamentally cheaper dequant path, and re-measure rather than
extrapolating.

### TG-4 - Graph coverage and pp256 variance (resolved, keep graphs ON)

`GGML_HIP_GRAPHS` is ON by default and in this build. Settled at `-r 5`:

| test | graphs ON | graphs OFF |
|---|---:|---:|
| pp256 | 1240.91 +/- 207.89 | 1321.92 +/- 19.92 |
| pp2048 | **1388.75 +/- 0.80** | 1381.96 +/- 1.08 |
| tg128 | **42.46 +/- 0.09** | 40.58 +/- 0.03 |

Conclusion: **keep graphs ON.** Decode gains 4.6%; `pp2048` is neutral within
noise (0.5%, against +/-1 t/s repeatability).

The apparent `pp256` regression is a **variance** problem, not a mean shift:
graphs ON measures +/-207.89 against +/-19.92 with graphs OFF. That is a
stability issue for short prompts worth tracking separately. It is not a
throughput lever, and graphs must not be disabled to chase it - doing so costs
4.6% on decode to buy an unstable short-prompt prefill number.

## Revision 2 work packages (2026-09-19)

Ordered by (confidence x payoff) / cost. P-1 is a kill test and comes first by
construction. Every package states its own acceptance; nothing merges on a
probe number alone (the PP-1 lesson).

### P-1 - Confirm the LDS conflict theory before any kernel work (hours) - EXECUTED, THEORY REFUTED

Two checks were run on 2026-09-19:

1. **Probe extension (`ggml/rocmfpx/probes/wmmalds4.hip`): REFUTED.** The
   kernel's actual pattern - one 16-byte read per operand per `mma_iu4`, row =
   lane % 16, row-major stride 44/36 dwords - measures **227-236 TOPS** across
   chains 1-4 and 192-3072 blocks, against the consecutive-chunk control at
   211-246 TOPS in the same binary. Identical within noise. The predicted
   drop to ~120-140 TOPS did not happen.
2. **rocprofv3 counters: superseded** - no point counting conflicts on a
   pattern the probe already shows to be conflict-free-equivalent.

Post-mortem, so nobody re-derives the wrong theory a third time: at 236 TOPS
the probe streams 32 B per `mma_iu4` = ~920 GB/s, about 26% of per-CU LDS
capacity, so neither pattern is LDS-bound and a 2-way conflict penalty cannot
surface. The earlier per-CU demand arithmetic that put the kernel near LDS
saturation was wrong (it mixed GPU-wide TOPS with per-CU bytes). What the
probe family actually shows, combining wmmalds2/wmmalds3/wmmalds4:

| reads per mma_iu4 | bytes | best TOPS | effective LDS GB/s |
|---|---|---:|---:|
| 4 x 8 B (wmmalds3) | 32 B | 152-170 | ~640 |
| 2 x 16 B, consecutive (wmmalds2) | 32 B | 230-246 | ~950 |
| 2 x 16 B, row-major 44/36 (wmmalds4) | 32 B | 227-236 | ~920 |

Same bytes, 1.5x rate: the cost driver is **LDS instruction count / width**,
not bank conflicts. That is consistent with the +3% the 16-byte widening
bought in the real kernel and closes the operand-path question: there is no
large operand-load lever left.

**P-3 (k-major staging) is killed by its own kill criterion. Do not build it.**

P-1b is repurposed: attribute the real kernel's 592 us epilogue-free floor
directly, with an ablation that separates staging from compute inside the k
loop, plus rocprofv3 counters on the surviving variant. See P-1c.

### P-1c - Attribute the residual floor: staging vs compute - EXECUTED 2026-09-19

Ablation of the current (P-2) kernel on the perf shape, baseline 670.47 us,
same discipline as PP-0 (diagnostic wrong builds, restored after):

- V1: `vec_dot` keeps every operand LDS load (A rows, B rows, dA gathers,
  dB), loads kept alive by a checksum; no MMA, no epilogue: **327.52 us**
- V2: `vec_dot` body empty; staging + barriers + y copies only: **190.49 us**

| bucket | us | share |
|---|---:|---:|
| MMA issue + fp32 epilogue | 343.0 | 51% |
| staging + barriers + loop | 190.5 | 28% |
| operand LDS loads inside vec_dot | 137.0 | 20% |

The 343 us bucket sits ~1.4x above the ~250 us realistic MMA-issue floor, so
the epilogue's VALU work still competes with WMMA issue (ISA 7.9), but its
per-element cost is near-minimal: cvt + 2 mul + add per output element per
k-step, and the scales differ every 32-element block, so pre-accumulating in
integer across blocks is not legal. The 327 us of staging+loads motivated
P-8 (in-kernel pipelining), which measured the model incomplete - see P-8.

rocprofv3 kernel trace on the same shape (P-2 build): `mul_mat_q<q4_0_rocmi4,
128, false>` 607.98 us average over 1479 dispatches (profiler durations
exclude inter-kernel gaps; consistent with the 670 us event timing), VGPR 224
(confirming P-2 caused no spills). `quantize_mmq_q8_1` runs 26.29 us per
launch and 38.9 ms of the pass = 4.1% - that is P-7's measured anchor.

### P-2 - Epilogue: hoist `dA`, fold `*16` - EXECUTED 2026-09-19, LANDED

**Correction to the sizing below, made during implementation.** For the
J_MAJOR C tile on RDNA3, `tile_C::get_i(l) = 2*l + tid/16`: each lane's 8
accumulator elements sit in 8 different rows, so `dA` is per (n, l) - 32
distinct values per lane per k-step, and the `j0` loop only re-reads them 2x,
not 8x. The honest expectation was therefore "about half the gather cost plus
the removed `*16` integer multiply", ~10% of the kernel, not the 7/8-gather
figure first written here.

Implemented: `dA16[ntx][tile_C::ne]` loaded above the `j0` loop, `*16` folded
into it, and the epilogue keeps the original multiply association -
`(float(C.x[l]) * dA16[n][l]) * dB` - because folding the constant into `dB`
instead would change the fp32 rounding order and break Tier 2. First version
had exactly that bug and was fixed before building.

| measurement | before | after |
|---|---:|---:|
| `pp2048` (fresh baseline same day) | 1434.20 +/- 2.85 | **1537.27 +/- 0.40 (+7.2%)** |
| `tg128` | 42.11 +/- 0.15 | 42.40 +/- 0.16 (noise; MMVQ untouched) |
| perf shape m=4096 n=512 k=14336 | 727.61-739.31 us | **670.47 us / 89.68 TFLOPS** |
| VGPR count (rocprofv3) | 224 | 224 (no spill from the +32 registers) |
| Tier 1: MUL_MAT / MUL_MAT_ID / IU4 oracle | OK | 12/12 / OK / OK |
| Tier 2 greedy decode | reference | token-exact |

Realised gain +7.2% `pp2048`, inside the corrected +7-9% window. Landed in
`ggml/src/ggml-cuda/mmq-vec-dot.cuh` on this branch.

### P-3 - k-major SRAM staging - KILLED BY P-1, DO NOT BUILD

Written before P-1 ran; kept so the reasoning and the kill are on record.
The mechanism (structural 2-way conflict) was derived correctly from the bank
model and measured to be irrelevant for `ds_read_b128` on gfx1100 (P-1a:
227-236 TOPS strided vs 211-246 consecutive). No k-major transposition of the
x or y staging tiles is warranted. The `load_ldmatrix_16` alignment and
`test-backend-ops` notes below remain valid for any future staging change.

- ~~Gate: Tier 1 at unchanged limits, Tier 2 token-exact, the IU4 oracle,
  `test-backend-ops` across all shapes and J values.~~
- ~~Sizing: kernel +30-60% -> pp2048 +25-45%.~~ Withdrawn.

### P-8 - Double-buffered k loop - EXECUTED 2026-09-19, REFUTED, REVERTED

Implemented behind a `double_buffer` field in `ggml_cuda_mmq_config` (default
false, set only for the W4A4 ROCMI4 config): two x tile sets and two y tile
sets in an `ids | y0 | y1 | x0 | x1` layout, y group g+1 staged during
vec_dot 1, the next x tile and y group g+2 staged during vec_dot 2, two
barriers per k-step instead of four. Same accumulation order, so numerics are
exact.

The gates earned their keep twice during bring-up:

1. First build failed Tier 1 on every MMQ shape (ERR 0.92 at m=1 n=64
   k=256): the loop staged only one y group per iteration, so vec_dot 1 of
   every iteration after the first read a stale group. `QK8_1_MMQ = 128`, so
   one iteration consumes two y groups. Fixed with a fixed-parity y schedule
   (even groups in y0, odd in y1, x swapped alone).
2. Second build still failed that shape (ERR 0.53): the kernel preamble
   places `tile_x` directly after the single y tile, so an naively placed
   `tile_y2` aliased the x tile. Fixed with the explicit five-region layout.

Final build: Tier 1 green (12/12, MUL_MAT_ID OK, oracle OK), Tier 2
token-exact, and **pp2048 = 1361.25 +/- 1.32 against 1537.27 for P-2: an
11.4% regression. Reverted; reverted state re-verified at 1535.03 +/- 1.84
with Tier 1 green.**

Why it lost, for the record: the P-1c staging+barriers bucket is not simply
serialized time waiting to be pipelined. The single-buffered J=128 tile set
is 30.2 KB of LDS, which allows two workgroups per CU, and a co-resident
workgroup already covers barrier drains. PP-2's occupancy sweep could not see
this because it varied the occupancy target at unchanged LDS. Doubling the
tiles to ~60 KB forces one workgroup per CU and surrenders that overlap; the
intra-workgroup pipeline does not win it back.

Do not re-attempt k-loop double buffering for this config without first
shrinking a tile set below ~16 KB, which the 4.25-bit format does not allow.
Recorded as a trap: an ablation bucket measured on a 2-workgroup kernel is
not recoverable by a change that costs residency.

### P-4 - Fuse activation quantization into MMVQ - REFUTED BY BYTE ARITHMETIC, DO NOT BUILD

Fusing the quantizer into `mul_mat_vec_q` makes every MMVQ block re-read the
f32 activation row (20 KB at K=5120) instead of the q8_1 row (5.5 KB). At
roughly 80-270 blocks per matvec that adds on the order of 1 GB of L2 traffic
per token to save about 1.2 ms of dispatch floor (436 launches x ~2.7 us).
The added bytes cost more than the dispatch saves. The standalone
`quantize_q8_1` kernel (0.51 ms exec per token) stays.

### P-5 - Fuse the GDN elementwise chain (open, scoped, not started)

~390 of 1414 dispatches per token (`l2_norm_f32`, `k_bin_bcast`,
`cpy_scalar`, `unary_gated_op`, `ssm_conv_f32`), estimated +3-6% tg. Needs
new fused kernels plus graph plumbing. The P-8 residency lesson applies: any
fusion whose LDS footprint crosses half of 64 KiB per workgroup can lose more
than it saves - keep fused tiles small and re-measure dispatches per token
alongside tg (Tier 2 for pure reorderings, Tier 3 with pre-declared tolerance
for accumulation-order changes).

### P-6a - GDN token-loop operand prefetch - EXECUTED 2026-09-19, NEUTRAL END-TO-END, REVERTED

Before attempting full chunking, the existing serial token loop was rewritten
with double-buffered operand registers (k/q/g/v/beta of token t+1 load while
token t computes). All loads are independent of the state chain, values and
arithmetic order are unchanged, gates green (GATED_DELTA_NET OK, Tier 2
token-exact).

Two lessons, one per attempt:

1. First version indexed the register buffers with the runtime `t & 1`. The
   compiler demoted them to scratch memory (VGPR collapsed 224 -> 56 class,
   here to 56) and the kernel ran 3.4x slower (631 -> 2126 us). A register
   array indexed by a runtime value is not a register array. Rewritten with a
   compile-time buffer tag and the token loop peeled into pairs; static
   indexing restored.
2. The fixed kernel wins the isolated op decisively: 631.35 -> 518.25 us at
   n_seq_tokens=512 (-18%), and flash-attention aside, GDN measured 649.6 us
   per 256-token launch in the real pipeline. But `pp2048` did not move:
   1538.52 +/- 1.20 against 1537.27 +/- 0.40. The GDN durations seen in
   rocprofv3 traces are inflated (the trap noted below), and the kernel's true
   wall-time share of `pp2048` is too small for an 18% kernel win to clear
   the noise floor. By the judge-on-pp2048 rule this is a no-gain change to a
   core kernel, and it was reverted. The reverted build re-measured 1537.42
   +/- 1.27.

### P-6 - Chunked GDN prefill (sizing corrected: expected value much lower than +8%)

The original +8% pp sizing came from the same trace method P-6a just
discredited: 33 ms of traced GDN time per pp512 pass. In-model, GDN shows
649.6 us per 256-token launch under the profiler, but an isolated -18%
kernel improvement (which includes most of what chunking could win at
256-token granularity) produced **zero** end-to-end movement. The chunked
reformulation remains the only way to cut the serial-scan cost itself, and
would still help long-prompt prefill where GDN launches stretch to 2048+
tokens, but its projected value must be treated as unknown until a trace
method that survives the profiler's inflation exists (e.g. comparing wall
time with `GGML_CUDA_DISABLE_GDN_CHUNKED`-style A/B on a real build). The
1-2 week project estimate stands; the payoff estimate does not.

### P-7 - Prefill elementwise cleanup (open, measured anchor)

`quantize_mmq_q8_1` costs 26.29 us per launch and 4.1% of the perf-shape pass
(rocprofv3, P-1c). Sharing one quantized buffer between `ffn_gate`/`ffn_up`
and between the q/k/v projections (which share their input norm) would remove
on the order of half the launches. Requires cross-op plumbing in the CUDA
backend of the kind that already exists for the MMVQ GLU fusion. Estimated
ceiling +2-3% pp. `concat_non_cont` (96 x 85 us per pp512 pass) may be
removable at the graph level - investigate, do not force it. Gate: Tier 2.

## Revision 2 profiling pass (2026-09-19, rocprofv3)

Purpose: replace trace-derived shares (shown unreliable by P-6a) with a
reconciliation of three evidence classes, and ground the remaining levers in
wall time. All runs on the committed P-2 build, `GGML_HIP_FORCE_MMQ=ON`,
gfx1100, ROCm 10 container, rocprofv3 1.3.5.

### What is and is not trustworthy on this stack

Reliable:
- Unprofiled `llama-bench` wall time (the arbiter for everything).
- Dispatch counts per kernel from `--kernel-trace` (they match layer
  arithmetic exactly).
- Isolated, event-timed op benchmarks (`test-backend-ops perf`, probes).

Not reliable, with evidence:
- Per-kernel durations under the profiler, in both directions. In-context GDN
  traced at 0.63 us/token while the isolated rate is 1.03 (post-P-6a attempt)
  and 1.23 (baseline); the pp-trace MMQ busy-time sum implies ~115 TMAC/s
  where MAC arithmetic and the doc's own sustained rate say 42-46; a decode
  trace captured 16 of 128 tokens before the profiler stopped recording.
- Any ROI argument built on traced durations or trace-derived shares. Both
  P-6a and the original 8.6% GDN share came from that method and both were
  wrong.

Therefore: attribution = traced COUNTS x ISOLATED rates, reconciled against
unprofiled wall. Durations under the profiler are used only as order-of-
magnitude sanity checks.

### Output-token caps (deliberate)

- Prefill traces run with `-n 0`: prompt cost does not depend on generated
  tokens; generating any would add a second workload to the trace.
- Decode trace ran with `-n 128` at depth 0, not more: decode cost is
  depth-dependent, not token-count-dependent, so generating more changes the
  quantity being measured (every token would see a longer context) while
  inflating the trace roughly 1850 dispatches per token. 128 samples pin the
  mean token time to a couple of percent.
- Depth-dependent decode was already measured separately with `-d` (TG-3).

### Measured: ubatch sweep is the largest remaining pp lever, and it is a config

| -ub | pp2048 t/s (unprofiled, -r 3) |
|---:|---:|
| 256 (serve script today) | 1416.94 +/- 0.69 |
| 512 (llama-bench default) | 1533.41 +/- 0.80 |
| 1024 | **1562.20 +/- 1.02** |
| 2048 | 1561.15 +/- 64.53 (unstable) |

Mechanism: at ub=256 each MMQ launch covers 2 j-tiles (128 blocks per launch
against 96 CUs); at ub=1024 it covers 8 j-tiles (~5x the blocks), so the GPU
is filled. ub=2048 matches the mean but with 60x the variance - same
signature as the pp256 instability in TG-4, single-giant-chunk runs.
Recommendation: serve with `-b 2048 -ub 1024`. Worth **+10.2% pp2048 for
prompt processing against today's serve config**, zero code. Decode is
unaffected (n=1 steps do not care about ub).

### Decode dispatch budget per token (counts are exact, from trace)

With FORCE_MMQ (bench env): ~496 MMQ + ~496 quantize_mmq + 448 eltwise +
305 norms + 112 copy + 96 get-rows + 48 GDN + 32 rope + 16 FA ~ 2050
dispatches per token. Serving (no FORCE_MMQ) swaps the 496+496 MMQ/quantize
pair for ~436 MMVQ + ~436 quantize_q8_1, ~1450 total. At the measured 23.6 ms
token wall, the launch floor alone (~2.7 us each) is 4-6 ms/token: dispatch
reduction (TG-2 class) is bounded near +15% tg before byte trade-offs, and
P-4's refutation already removes the biggest single candidate.

### Prefill kernel budget at ub=1024 (count x isolated-rate, wall-anchored)

Shares of the 1311 ms pass, using isolated rates where they exist:
- MMQ: ~74% of kernel time (MAC arithmetic at the realized 42-46 TMAC/s).
  Isolated kernel efficiency is 90.6 TFLOPS at the n=512 perf shape; the
  whole-kernel mechanism list is exhausted (P-2 landed, P-8 refuted).
- GDN: 96 launches x ~1.01 ms ~ 97 ms ~ 7.4% by isolated rates, yet an -18%
  kernel change was invisible end to end (P-6a). Treat the true wall share
  as unresolved in the 2-7% band until a chunked-kernel A/B exists.
- FA: 32 launches x ~1.0 ms ~ 2.4% at ub=1024 (it is 8.6% at ub=256 - large
  ubatch dissolves most of it).
- quantize + eltwise + norms + copy + rope + get-rows: ~9% combined; P-7
  (shared act-quant buffers) ceilings near +2%.

### Ranked remaining levers, with their empirical bounds

1. Serve config `-b 2048 -ub 1024`: +10.2% pp2048, measured, free. Do this
   first.
2. MMQ shape tuning at n>=1024: the perf suite stops at n=512 (90.6 TFLOPS);
   whether the kernel holds that efficiency at n=1024/2048 is unmeasured
   because the suite cannot express it. A one-off op benchmark at real
   shapes would either close the question or expose a tuning target. Cheap.
3. TG-2-class dispatch reduction for decode: bound ~+15% tg before byte
   trade-offs; P-4 (the best single piece) is refuted; the rest is invasive
   graph plumbing. Size any piece by wall-time prototype first.
4. Chunked GDN (P-6): payoff unknown (projection withdrawn); helps long
   chunks most; requires the full WY kernel and a graphs-off wall A/B.
5. Tensor-core FA via rocWMMA: dependency not present in the container;
   ceiling ~2-3% at ub=1024 anyway. Lowest priority.

### Revision 4 (2026-09-20): post-WP4 decode census, WP5 closed, P-5 re-ranked

Graphs-off `rocprofv3 --kernel-trace`, HEAD build, 128 tokens at d0; counts
normalized by the exact MMVQ census (55,857 launches = 436/token). Post-WP4
decode is ~1,240 launches/token (436 MMVQ + ~804 others, down from ~1,850
pre-AR). Unprofiled d0 wall 22.54 ms (44.36 t/s). Attribution follows the
Revision 2 rule: counts exact, traced durations inflated upper bounds
(est-real = 2.35 us/launch dispatch floor + ~0.4x traced execution).

| candidate | n/tok | traced ms/tok | % wall (bound) | est-real % |
|---|---:|---:|---:|---:|
| k_get_rows_float_vec (WP5) | 48.3 | 0.337 | 1.5% | ~1.1% |
| rms_norm leftovers (1024 + 256) | 209.5 | 0.826 | 3.7% | ~3.6% |
| copy/gather (cpy_scalar, get_rows_f32, set_rows, concat_cont) | 193.9 | 0.444 | 2.0% | ~2.8% |
| elementwise (silu, sigmoid, bin_bcast) | 112.7 | 0.227 | 1.0% | ~1.6% |
| quantize_q8_1 residual | 129.9 | 0.273 | 1.2% | ~1.8% |
| ssm_conv | 48.3 | 0.141 | 0.6% | ~0.8% |
| rope + flash_attn pair | 64.5 | 0.235 | 1.0% | ~1.1% |

Verdicts:

- **WP5 closed.** 1.5% upper bound < the 3% gate; an engine-level GDN
  cache-read contract cannot pay for itself. Recorded in the AR plan.
- **P-5 re-ranked.** Fusion order: (1) copy/gather family - mechanical,
  plain GGML nodes, no engine contract; (2) elementwise epilogues - the WP3
  pattern; (3) rms_norm leftovers - largest share but needs new epilogue
  patterns (WP2 already took the producer-side ones). A batch covering (1)+(2)
  removes ~306 launches/token for ~0.7-1.0 ms/token, **est. +3-4.5% tg at
  d0**. The whole non-MMVQ launch stream floors at ~12.9% of wall - the
  unreachable ceiling for TG-2-class work.

### Revision 5 (2026-09-21): P-5 batch round 1 - F1 non-firing, F2a blocked, protocol trap found

Round 1 implemented two dispatcher fusions from the Revision 4 re-rank:

- **F1 mul_mat+reshape+add** (GDN residual ADD, 48 launches/token): implemented
  in `ggml-cuda.cu` behind `GGML_CUDA_DISABLE_MUL_MAT_RESHAPE_ADD` (kept on
  the experiment branch, not merged). Passes a CPU-builder graph replication, compiles clean, Tier 1
  host asserts verified - but a rocprofv3 trace at the correct protocol
  (`-n 128`, graphs off) shows it NEVER fires on the real decode graph:
  `k_bin_bcast` stays 48.3/token and fused-MMVQ stays 241.7. The real backend
  graph (buffer placement, COMPUTE/view flags, or can_fuse single-use rules)
  rejects some predicate the replication satisfies. Next step: instrument the
  matcher rejection reason on the live graph. Dead code until then; harmless
  behind its switch.
- **F2a conv-state chain fusion** (CONCAT+VIEW+CPY+SSM_CONV+silu+l2, 96
  launches): was fully implemented (kernel `ssm_conv_state_f32`, matcher,
  builder reorder in `qwen35.cpp` hoisting the ssm-state fetch) and compiled,
  then reverted. The revert reason (allocator aliasing from the reorder) was
  an artifact of a bad protocol - see below - so the refutation is VOID. The
  code is no longer in the tree; re-implementation is cheap (matcher pins the
  empirically verified node order CONCAT,VIEW,VIEW,CPY,SSM_CONV,SILU,L2s with
  the ssm-state fetch hoisted before the conv chain; kernel reads old state +
  qkv directly and writes the cache update view; layouts verified: old[c+3*ch]
  for c<3, qkv[ch] for c=3, new[c+3*ch] = old[(c+1)+3*ch] for c<2 else qkv).
  Re-test ONLY at -n 128.

**Protocol trap (important for all future fusion work): decode fusion firing
depends on `-n`.** At `-p 0 -n 32 -d 0` the UNMODIFIED HEAD binary does not
fire WP4 (`ssm_conv_f32<...,false>` + 96.7 standalone `l2_norm_f32`/token);
at `-n 128` it does (`<...,true>`, no l2 launches). Same binary, same flags
otherwise. Mechanism unknown (suspect: graph rebuild/buffer-reuse thresholds
crossed at different token positions). All AR-plan A/B numbers used -n 128
throughout and are unaffected. Rule: any fusion census or trace must pin
`-n 128` (or the production workload shape) and record graphs on/off.

WP5 stays closed (Revision 4). The census-silence side quest: llama-bench
installs a log callback that swallows the per-node `cuda_graph_census_exec`
INFO lines after startup; census verification needs a different harness or
log level - rocprofv3 kernel counts are the working substitute.

## Correctness plan

Every WP above runs behind these gates. Gates are declared before the work, not
after.

**Tier 1 - kernel numerics (every build).**

- `test-backend-ops` `MUL_MAT` and `MUL_MAT_ID` for `q4_0_rocmi4`. Limits stay
  as today: NMSE <= 5e-4 for exact ROCmI4, <= 1e-2 only in a build that
  advertises the `ROCMI4_W4A4` backend feature. Do not raise these limits.

- **Known flake, found while validating PP-1 (open).** One run in 35 observed
  `MUL_MAT` `q4_0_rocmi4` failing at `m=16 n=1 k=256` with
  `ERR = 0.011686 > 0.010000` - 1.17x the limit, on the smallest shape in the
  suite, where the lossy 4-bit activation quantization gets the least
  averaging. It did not reproduce in 10 baseline runs or 20 runs with the PP-1
  change, so it is rare rather than absent, and it is a property of the W4A4
  activation path, not of any single change.

  This matters beyond bookkeeping: a Tier 1 gate that fails a few percent of
  the time is not a usable gate. Do not resolve it by loosening the bound -
  that is the one move that would make the gate meaningless. Either state the
  bound per shape family with evidence for why the small shape is legitimately
  worse, or make the check deterministic (fixed seed) so a real regression is
  distinguishable from a sampling event. Until then, treat a single Tier 1
  failure on `m=16 n=1` as inconclusive and re-run before acting.
- `test-rocmi4-iu4-dot` host oracle: packed IU4 x IU4 DOT8 must equal
  unpack + DP4A against Q8-stored IU4 with bit-identical integer sums.
- For the chunked GDN kernel: A/B against the sequential kernel on random
  inputs at several `n_tokens`, and the existing `test_gated_delta_net` cases
  including the PP-512 configuration. A tolerance must be stated up front,
  derived from the operation (fp32 accumulation order will differ), not tuned
  until it passes.

**Tier 2 - end-to-end determinism (every non-lossy change).**

Greedy decode, temperature 0, fixed prompt set, fixed seed, exact token match
against the pre-change build. Any token mismatch on a change that was supposed
to be a pure reordering means the change is wrong, not "close enough".

**Tier 3 - quality gate (every lossy change: W4A4 activation paths, KV quant,
any new lossy kernel).**

- Perplexity and/or KL divergence against a fixed corpus, at several context
  depths for KV changes, comparing against the exact-int8 build.
- The existing recorded figure is +5.4% perplexity for W4A4 activations
  (gfx1151 measurement). Any new lossy change must state its own number, and
  the total must stay inside a band agreed before the change.
- This needs `llama-perplexity`, which is not built in the current build
  directories; build it once for the gate.

**Tier 4 - reporting discipline.**

- Every reported throughput names the build directory, CMake options, model
  hash, context depth, `-ngl`/`-fa`, and repetition count.
- Never compare a config against a differently-configured baseline.
- Never quote `pp512` as the primary prefill metric; it is not stable enough.

## Measurement traps to avoid

Recorded here because each of them has already appeared while producing this
plan.

- **`pp512` variance is +/-200 t/s.** It produced 1332 and 1277 for identical
  configurations. Use `pp2048`.
- **`test-backend-ops perf` shapes are L2-resident.** A 4096 x 14336 ROCmI4
  tensor is ~31 MB, which fits the Infinity Cache. Those TFLOPS numbers measure
  kernel efficiency, not memory behaviour. They are valid for the MMQ
  comparison here (prefill runs at 47 GB/s, far below DRAM peak) but must not
  be used to reason about decode.
- **`rocprofv3` inflates inter-kernel gaps.** The traced decode timeline showed
  49.5% gaps against 15% inferred from the unprofiled wall time. Use the trace
  for kernel-duration attribution and per-dispatch counts, not for gaps.
- **A probe ceiling bounds, it does not forecast.** The operand probes showed
  the 16-byte-load pattern raising the LDS-fed ceiling from ~170 to ~240 TOPS,
  a ~1.4x operand-path improvement. Implemented in the real kernel it produced
  +3% end to end, because operand loads are a much smaller share of the real
  kernel than of a four-line probe loop. Use probes to decide *direction* and
  to rule things out; never to size a delivery.
- **A register array indexed by a runtime value is scratch memory.** P-6a's
  first version selected one of two operand buffers with `t & 1`; the compiler
  moved the arrays to local memory (VGPR count collapsed) and the kernel ran
  3.4x slower. Index register arrays only with compile-time constants (a
  template tag plus loop peeling works), and check the VGPR count of any
  change that adds indexed storage.
- **An ablation bucket measured on an N-workgroup kernel is not recoverable by
  a change that costs residency.** P-1c put 190 us on staging+barriers, but
  that bucket was partly hidden by a second co-resident workgroup (30.2 KB LDS
  per workgroup, two fit per CU). P-8's doubled tiles forced one workgroup per
  CU and the "recovered" bucket came back as an 11.4% regression. Check what
  the LDS footprint does to workgroups per CU before crediting a bucket as
  headroom.
- **Decode-side byte trades must be counted in both currencies.** P-4 looked
  like +5-7% from dispatch counts alone; counting the extra f32 re-read bytes
  (~1 GB/token) shows it is net negative. Dispatch savings and byte costs are
  the same budget on a bandwidth-bound GPU.
- **A model that fits the baseline is not validated for a change.** The
  `bytes / 792 GB/s + 6.35 ms` decode model predicted f16 depth scaling to
  within 1% and was then used to project KV quantization at +12% to +20%. KV
  quantization measured -5% to -10%. The model captured the byte cost and
  missed the dequant cost entirely. Fit on the configuration you have, then
  measure the configuration you are proposing.
- **Concurrent builds corrupt running benchmarks.** Running `ninja` in a build
  directory while a binary from that directory is executing relinks the shared
  libraries underneath it; the symptom was a 20-minute hang followed by
  "failed to load model". Build and benchmark serially.
- **`llama-bench` names the test after `-n`.** `-n 32` reports `tg32`, not
  `tg128`. Grepping for the wrong label silently reports every run as failed.
- **Microbenchmarks must have a live result.** A first version of the
  bandwidth probe accumulated into a buffer with a never-taken store; dead-load
  elimination could have deleted the whole loop. The accumulator now feeds a
  data-dependent branch that can never fire.
- **Peak-clock TOPS is not achievable TOPS.** The 282 TOPS figure is one
  kernel with all operands in registers. Real kernels must stage operands
  through LDS, which is why the ablation in PP-0 is required before optimising.

## Risk register

| Risk | Mitigation |
|---|---|
| Probe models disagree with the real kernel (happened twice: PP-1, P-1) | Decide direction with probes, size only with real-kernel ablations and rocprofv3; kill tests before builds |
| P-1c ablation variants are numerically wrong | They are diagnostic only, like PP-0; md5-verified restore + rebuilt baseline re-measure after each |
| P-2 hoist raises VGPR pressure and spills | Check the compiler's SGPR/VGPR usage; fall back to hoist-without-fold; keep only what wins `pp2048` |
| P-2 fold claimed bit-exact but tokens diverge | Then the exactness argument is wrong somewhere - stop, do not loosen Tier 2 to a tolerance |
| PP-0 shows the epilogue is not the bottleneck | Already happened; plan redirected to operand staging, and the operand probes then narrowed it to load width |
| Epilogue trimming is numerically sensitive | `*16` folding is exact in fp32 for these magnitudes; verify with Tier 1 + Tier 2, do not assume |
| Chunked GDN diverges from the sequential reference | Declare tolerance from accumulation-order analysis first; A/B on random inputs across lengths before touching the model |
| `ntx=2` raises VGPR pressure and lowers occupancy instead of helping | Measure per candidate, keep only what wins |
| KV quant passes NMSE but degrades real output | Tier 3 quality gate, at each depth, before shipping |
| TG gains get attributed to MMVQ when they came from fewer dispatches | Trace dispatch counts alongside every tg number |
| Multi-arch fat binary (gfx1100 + gfx115x) changes codegen | Out of scope here; do not mix into these WPs |

## Explicit non-goals

- Speculative decoding, MTP, or any draft model. The `blk.64` MTP head stays
  unused.
- Changing the GGUF, the `block_rocmi4` layout, or the 17-byte block size.
- Enabling W4A4 by default. It stays opt-in and lossy.
- RDNA2, RDNA4, or Vulkan paths.
- Repacking weights into an aligned layout at load time: a second 14.5 GB copy
  does not fit in 24 GB alongside the model.
- Multi-GPU or split-K across devices.

## Rejected approaches, with reasons

- **Raising clocks / memory tuning.** Streaming already measures 923.5 GB/s =
  96% of the 960 GB/s board spec. Nothing is left there.
- **Bigger J alone.** J does not reduce the per-accumulator-element epilogue,
  which the IU8-vs-IU4 measurement identifies as the dominant cost.
- **`V_WMMA_I32_16X16X16_IU8` with the existing exact path.** It is 2x slower
  in MMA issue terms and measures 26.5% slower end to end. Keep IU4.
- **VALU `V_DOT8_I32_IU4` for prefill.** Measures 77.3 TOPS against 282.4 for
  WMMA. MMVQ is the right place for DOT8, and it is already there.
- **Int8 exact path for prefill.** 1090.75 t/s vs 1380.09. W4A4 wins; the
  question is only whether its accuracy cost is acceptable per Tier 3.
- **Weight-stationary persistent MMQ** (stage each weight tile once instead of
  once per j-block). Total LDS write volume is 14.3 GB x (J=128/16 lanes) ~ 16
  re-reads = ~229 GB per pp2048 pass, ~7.6 ms against ~30 TB/s of aggregate
  LDS write bandwidth - under 1% of the pass. Not worth the invasive rewrite.
- **VOPD dual issue (ISA 7.6) for the epilogue.** VOPD accepts VOP2-class
  opcodes only; WMMA is VOP3P and cannot dual-issue. The epilogue's VALU work
  competes with WMMA for issue (ISA 7.9: WMMA internally uses the DOT
  instructions), which is a reason P-2 helps, not a dual-issue opportunity.
- **SLC/DLC streaming cache hints (ISA 4.1.1) on MMVQ weight loads.** MMVQ
  already streams at 94% of what the on-disk layout permits; the bound on this
  is ~4% and probably less. Revisit only if P-4/P-5 land and a gap remains.

## Suggested order

Completed and measured:

1. ~~TG-4 graph repeat~~ - keep graphs ON (+4.6% tg)
2. ~~PP-0 ablation~~ - epilogue is 22.4%, not the bottleneck
3. ~~PP-1 ILP / occupancy sweep~~ - both refuted
4. ~~PP-1 item 1, 16-byte operand loads~~ - landed, +3.2% to +3.8% `pp2048`
5. ~~TG-1 MMVQ layout probe~~ - MMVQ is at 94% of the layout-limited ceiling
6. ~~TG-3 KV quantization~~ - refuted, costs 5% to 10%
7. ~~PP-2 tile sweep~~ - no change; shipped config was already best

Revision 2 final state (all short items executed 2026-09-19):

1. ~~**P-1** conflict confirmation~~ - EXECUTED, theory refuted, P-3 killed.
   Probe kept as `ggml/rocmfpx/probes/wmmalds4.hip`.
2. ~~**P-2** epilogue hoist + `*16` fold~~ - **LANDED, +7.2% pp2048**
   (1434.20 -> 1537.27), Tier 1 + Tier 2 green.
3. ~~**P-1c** staging-vs-compute partition~~ - EXECUTED: 343/190/137 us split
   (MMA+epilogue / staging+barriers / operand loads).
4. ~~**P-8** double-buffered k loop~~ - EXECUTED, -11.4% pp2048, reverted.
   The staging bucket is already hidden by 2-workgroup co-residency.
5. ~~**P-4** quantize-into-MMVQ fusion~~ - refuted by byte arithmetic before
   implementation.
6. **P-7** prefill cleanup - open, +2-3% pp ceiling, measured anchor.
7. **P-5** GDN elementwise fusion - open, sized +3-6% tg from trace counts;
   that method is now suspect (see P-6a), size it from wall time before
   building.
8. ~~**P-6** chunked GDN~~ - attempted the cheap half (P-6a, operand
   prefetch): kernel -18% isolated, end-to-end neutral, reverted. The full
   chunked reformulation's +8% pp projection is withdrawn; re-size from wall
   time before starting it.

### Revision 2 outcome

| step | result |
|---|---|
| P-2 epilogue hoist | **+7.2% pp2048, landed** (1434.20 -> 1537.27 +/- 0.40) |
| P-1 conflict theory | refuted by probe; staging-transpose lever closed |
| P-1c floor partition | 343 us MMA+epilogue / 190 us staging / 137 us loads |
| P-8 k-loop pipelining | refuted by measurement (-11.4%); residency mechanism recorded |
| P-4 quantize fusion | refuted by arithmetic; dispatch cost < re-read bytes |
| decode | unchanged from Revision 1: 42 t/s is within ~10% of the format+GPU floor |

Where this leaves the model on this GPU: **pp2048 1537 t/s (+7.3% over the
branch baseline), tg128 42.1-42.4 t/s.** Both are within a few percent of
what this format and GPU permit without speculative decoding: prefill is
MMQ-issue-bound at ~90 TFLOPS on the perf shape with memory at 5% of peak,
decode streams weights at 94% of the layout-limited ceiling. The remaining
candidates (P-7, P-5) are single-digit-percent plumbings whose sizing methods
P-6a just discredited - each must first show a wall-time win in a quick
prototype before being built out. Nothing measured in Revision 2 changes the
decode floor: 13.62 GB of weight reads per token is untouchable without
speculative decoding (excluded) or a different format (excluded).

Decode keeps its Revision 1 conclusion: 42.5 t/s is within ~10% of what this
model, the 17-byte block format and this GPU permit without speculative
decoding. The work above targets ~45-48 t/s; the 67.8 t/s DRAM floor needs
fewer bytes per token, which without speculative decoding means changing the
GGUF - a non-goal.

## Magpie kernel evaluation and rocprofiler-compute status (2026-09-19)

The P-2 epilogue restructure was additionally evaluated with AMD's Magpie
framework (AMD-AGI/Magpie, run inside the ROCm 10 container). Magpie's
compatibility matrix does not list gfx1100 (MI300X-class plus a bounded
gfx1151 row only), so this run is itself the verification: the
correctness/compare path works on gfx1100.

Magpie harness (kept on the experiment branch, not merged): two kernel entries (baseline =
commit 15acbdbbf tree, optimized = HEAD), each compiling the real W4A4
vec-dot and tile loader from its tree into a standalone eval kernel that
replicates the J=128 process-tile loop, plus a correctness testcase.
Gate: outputs must be bit-exact against the merged-baseline golden
checksum (the restructure's claimed property). Data uses constant nibbles
per (row, block) so the fp64 reference is slot-order independent; the
nibble k-order itself is covered by the repo's IU4 oracle and
test-backend-ops.

Results:

- Correctness: 2/2 passed in `magpie compare`; both variants produce the
  identical golden checksum (bit-exact equivalence confirmed by an
  independent framework).
- Kernel perf (harness event timing): baseline 43.40-43.76 us vs
  optimized 39.87-40.28 us per tile-iteration, about -8 percent,
  consistent with the test-backend-ops measurement (728 -> 670 us) and
  the +7.2 percent pp2048 end-to-end result.
- rocprofiler-compute (omniperf 3.3.0 from source, with a shim for
  Magpie) was installed and committed as image `rocm-dev:10.0.0-rpc`.
  Magpie's HIP performance ranking still fails on this GPU: omniperf
  aborts with "Cannot find a supported arch in rocminfo: None" - gfx1100
  is outside its supported-arch table. Kernel-comparison perf therefore
  remains event-timed by the harness. Roofline ranking for kernel work on
  gfx1100 needs an omniperf release that adds the arch, or manual
  rocprofv3 counter queries.

### Close-out of the remaining Revision 2 items

- MMQ shape tuning at n >= 1024: RESOLVED as a config, not a kernel
  change. The ub sweep measured it end to end (ub=1024 gives +3.2 percent
  pp2048 over ub=512 and +10.2 percent over the old serve config).
  Per-tile work is identical at larger ub; the gain is grid fill (2 -> 8
  j-tiles per launch against 96 CUs). No separate kernel-tuning target
  hides behind n >= 1024.
- TG-2 dispatch reduction: bounded at +15 percent tg before byte
  trade-offs; its largest single piece (P-4) is refuted, the rest needs
  graph plumbing and a wall-time prototype before building.
- P-7 shared act-quant: designed, not built. Safe cache key is the src1
  tensor pointer (live graph nodes never overlap memory; distinct nodes
  are distinct tensors), reset per graph evaluation; q/k/v collapse 3
  quantizes to 1 and gate/up 2 to 1. Expected +1.5-2 percent pp from the
  rocprofv3 dispatch counts (992 launches x 26.3 us per pp512 pass).
  It touches pool ownership and graph capture, so it needs a session
  with GPU access for the Tier 2 + pp2048 gates.
- P-5 GDN elementwise fusion and P-6 chunked GDN: unchanged, sized from
  wall time only.

Skill note: of the local AMD skill collection, only magpie-kernel-evaluator
applies to this work. serving-llms-on-instinct and serving-llms-on-epyc
target vLLM on MI-series/EPYC; tracelens-analysis-orchestrator consumes
torch profiler traces, which the llama.cpp stack does not produce.

## References

- ISA: `rdna3-shader-instruction-set-architecture-feb-2023_0.md`, section 7.9
  (WMMA, including the lane replication rule and the "internally use the DOT
  instructions" note), 16.10 (V_DOT8_I32_IU4 opcode 24, V_WMMA_I32_16X16X16_IU4
  opcode 69)
- Prior work: `docs/rocmfpx/plans/rdna3-w4a4-iu4-mmq.md`
- Format: `ggml/rocmfpx/ROCMI4.md`
- Support tiers: `docs/rocmfpx/SUPPORT.md`
- Mamba-2 chunked precedent: `ggml/src/ggml-cuda/ssm-scan.cu`
  (`ssm_scan_ssd_f32_cuda`)
