# Plan: RX 7900 XTX (gfx1100) decode/prefill acceleration for Qwen3.8-27B Q4_0_ROCMI4

Branch: off `feature/rdna3-w4a4-iu4-mmq`
Target host: RX 7900 XTX (`gfx1100`), ROCm 10 container (HIP 7.15)
Model: `/home/ghazni/models/rocmfpx/cafonez/Qwen3.8-27B-ROCmI4/Qwen3.8-27B-Q4_0_ROCMI4.gguf`

Status: partially executed. Six work packages measured, one landed. Every
number below was measured on this host, not extrapolated from vendor peaks.

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
constraints exclude.

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

Remaining in PP-1:

2. **Explain the residual.** Epilogue-free the kernel runs at 101.55 TFLOPS
   against ~170 for its own load pattern, and the widening moved it less than
   expected. Re-derive where the time actually goes against the new baseline
   rather than trusting the probe model again.
3. **Bank conflicts and SRAM strides.** The `dA`/`dB` gathers in the epilogue
   and the `sram_stride` walk are the remaining untested suspects. A stride
   sweep (pad by 1-2 dwords) is cheap and would show up immediately as a
   conflict signature.
4. **The epilogue itself (secondary).** Folding the `*16` into a precomputed
   scale removes the measured 48.4 us arithmetic chain; vectorising the `dA`
   gather attacks part of the 122.7 us. Capped at 22.4% of the kernel.

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

### TG-2 - Cut the 1414 small kernels per token

At ~2.7 us of dispatch floor each, this stream costs ~3.8 ms/token of dispatch
alone against 2.75 ms of execution.

- Fuse `quantize_q8_1` into the MMVQ kernel, or at minimum avoid re-quantizing
  the same activation: 436 quantize launches for 436 matvecs suggests no reuse
  today. Verify whether `ffn_gate`/`ffn_up`, which share an input, quantize it
  twice.
- Fuse the SSM-layer elementwise chain: `l2_norm_f32`, `k_bin_bcast`,
  `unary_gated_op`, `cpy_scalar`, `ssm_conv_f32` together account for ~390
  launches/token.

Acceptance: dispatches/token falls measurably (re-run the rocprofv3 kernel
trace and compare counts); `llama-bench -n 128 -r 5` improves.

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
| 16-byte LDS operand load is illegal for some `sram_stride`/`k00` combinations | Derive alignment from the actual config values first; keep a guarded 2x8-byte path for the unaligned case; `test-backend-ops` across all shapes and J values, not one shape |
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

## Suggested order

Completed and measured:

1. ~~TG-4 graph repeat~~ - keep graphs ON (+4.6% tg)
2. ~~PP-0 ablation~~ - epilogue is 22.4%, not the bottleneck
3. ~~PP-1 ILP / occupancy sweep~~ - both refuted
4. ~~PP-1 item 1, 16-byte operand loads~~ - landed, +3.2% to +3.8% `pp2048`
5. ~~TG-1 MMVQ layout probe~~ - MMVQ is at 94% of the layout-limited ceiling
6. ~~TG-3 KV quantization~~ - refuted, costs 5% to 10%
7. ~~PP-2 tile sweep~~ - no change; shipped config was already best

Still open:

8. **PP-3** chunked GDN prefill - sized at **~+8% pp**, measured linear-in-token
   scaling confirms the serial scan is the cause, and there is no GPU
   implementation to port. This is its own kernel project with its own
   validation cycle (see the section above); it needs a focused pass, not a
   tail-end of this one.
9. **PP-1 items 2-3** - residual attribution against the new baseline.
   The bank-conflict stride sweep is likely a dead end: any stride satisfying
   the required `% 4` alignment conflicts 2-way across 16 rows, so it is
   structural rather than tunable.
10. **TG-2** dispatch reduction - +5% to +8% at best, needs core dispatch
    changes; poor value against the regression risk.
11. **PP-4** elementwise cleanup - small.

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
