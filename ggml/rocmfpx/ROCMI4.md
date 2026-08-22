# Q4_0_ROCMI4 — native signed-nibble 4-bit

True 4-bit integer path for gfx1151. Codes **are** the values (`[-8, +7]`).
No Codebook10 table and no `amdgcn_perm` expander.

## Why this exists

`Q8_0_ROCMFPX` is a 33-byte block: 32 int8 codes + one UE4M3 scale. HIP
decode is `load i8 + dp4a`. That is the FP8-codebook baseline.

`Q4_0_ROCMFP4_FAST` is 17 bytes but still expands each nibble through
Codebook10 `{0,±1,±2,±3,±4,±6,±8,±10}` before `dp4a`. That expansion is
why 4-bit can lose to the FP8 codebook on decode.

`Q4_0_ROCMI4` keeps the 17-byte FAST layout (32 nibbles + one UE4M3 scale)
but stores two's-complement nibbles and sign-extends them with shifts only:

```
even = q4 & 0x0F0F0F0F
sign = even & 0x08080808
even |= (sign<<1)|(sign<<2)|(sign<<3)|(sign<<4)
dp4a(even, q8, acc)
```

## Measured on this machine (gfx1151, ROCm0, 2026-08-17)

Qwen3-0.6B after I4 HIP MMQ, `llama-bench -ngl 99 -dev ROCm0 -fa on -p 512 -n 128 -r 3`:

| quant | size | pp512 | **tg128** |
|---|---:|---:|---:|
| Q8_0_ROCMFPX | 586 MiB | 12061 t/s | **222.0 t/s** |
| Q4_0_ROCMI4 | 345 MiB | **12730 t/s** | **287.4 t/s** |

Qwen3.8-27B, same flags, plus `-pg 512,128` (TTFT proxy):

| quant | size | pp512 | pp2048 | **tg128** | pp512+tg128 | TTFT@512 |
|---|---:|---:|---:|---:|---:|---:|
| Q8_0_ROCMFPX | 26.25 GiB | 333.2 t/s | 332.4 t/s | **7.88 t/s** | 35.9 t/s | 1536 ms |
| Q4_0_ROCMI4 | 13.53 GiB | **426.2 t/s** | **424.9 t/s** | **13.82 t/s** | **61.0 t/s** | **1201 ms** |

27B I4 vs FP8 codebook: prefill **+28%**, decode **+75%**, combined **+70%**, TTFT **-22%**.
MTP (`draft-mtp` n_max=2 p_min=0.75 strict-qwen) works: 27/32 then 26/26 drafts accepted, ~20–34 t/s decode.

## Status

- CPU reference quant/dequant/dot + CPU GET_ROWS
- HIP MMVQ (decode) + HIP MMQ (prefill) + HIP GET_ROWS
- `llama-quantize … Q4_0_ROCMI4`
- 27B GGUF: `/mnt/seconddrive/models/Qwen3.8-27B-ROCmFPX-GGUF/Qwen3.8-27B-Q4_0_ROCMI4.gguf`
- Vulkan integer-dot MMVQ / MMQ not wired yet

## Importance matrix (imatrix)

ROCMI4 **supports imatrix** as of 2026-08-21. It previously ignored it
(`GGML_UNUSED(imatrix)`), matching neither its FP2/FP3/FP6/FP8 siblings nor
llama.cpp convention.

ROCMI4 has exactly one free parameter per 32-value block -- the UE4M3 scale;
the codes then follow by rounding. So imatrix support means choosing the scale
that minimises the *importance-weighted* error over the block instead of just
taking `amax/7`. Implemented as `rocmfpx_choose_scale_i4_weighted_mse` +
`rocmfpx_quantize_row_i4_weighted`, mirroring the existing FP8 pattern and
reusing the shared `rocmfpx_prepare_mse_weights`.

Measured, Qwen3-0.6B **BF16 -> Q4_0_ROCMI4** (single quantisation), imatrix
calibrated on `wiki.train.raw` (100 chunks), PPL on `wiki.test.raw` (50 chunks):

| | PPL |
|---|---|
| no imatrix | 24.3900 +/- 0.70949 |
| **with imatrix** | **21.9280 +/- 0.63670** |

**-10.1% perplexity at zero runtime cost.** Quantisation time goes 1.6 s -> 4.9 s
for the 0.6B (the scale search), which is a one-off.

Two caveats learned the hard way:

- **Requantising ROCMI4 -> ROCMI4 is a no-op.** `llama-quantize` copies tensors
  that are already the target type, so an A/B from a ROCMI4 source shows
  *identical* PPL and proves nothing. Always test from BF16/F16.
- **Do not benchmark this on a tiny/over-quantised model.** On a 230M model
  already at PPL ~44, imatrix measured 1.8% *worse* -- the weighted-MSE proxy
  stops tracking output error in that regime. The 0.6B from BF16 is the honest test.

## Quantize

```bash
llama-quantize --allow-requantize in.gguf out.gguf Q4_0_ROCMI4

# with an importance matrix (recommended -- quantise from BF16/F16, not from a quant)
llama-imatrix  -m model-BF16.gguf -f calib.txt -o model.imatrix -ngl 99 -dev ROCm0 --chunks 100 -c 512
llama-quantize --imatrix model.imatrix model-BF16.gguf out.gguf Q4_0_ROCMI4
```

## Native IU4 tensor core (gfx1151 / RDNA3)

`v_wmma_i32_16x16x16_iu4` retires in **half the cycles** of the iu8 form at the
same 16x16x16 shape. Measured on this machine (identical instruction counts,
identical occupancy, 40 CU @ 2.9 GHz):

| instruction | throughput |
|---|---:|
| `v_dot4_i32_iu8` (dp4a) | 29.5 TOPS |
| `v_dot8_i32_iu4` | 58.9 TOPS |
| `v_wmma_i32_16x16x16_iu8` | 57.1 TOPS |
| `v_wmma_i32_16x16x16_iu4` | **110.6 TOPS** |
| `v_wmma_f32_16x16x16_f16` | 57.1 TFLOPS |

These land at 93-96% of AMD's documented RDNA3 rates (IU8 512, IU4 1024
ops/clock/CU). The widespread claim that RDNA3.5 gets no low-precision benefit
is true for INT8 and **stops one datatype short**. Memory roofline for
comparison: 231.5 GB/s measured.

ROCMI4 codes are already two's-complement `[-8,+7]`, i.e. a native signed iu4
operand with no unpacking -- which is what makes this format, and not the E2M1
codebook formats, the one that maps onto the 4-bit lane.

### Results (Qwen3.8-27B-Q4_0_ROCMI4, `-fa on`)

| mode | pp512 | vs base | PPL 27B | PPL 0.6B |
|---|---:|---:|---:|---:|
| int8 WMMA (baseline) | 462.2 | 1.00x | 6.0183 | 24.2381 |
| IU4 exact, 2 bit-planes | 360.5 | 0.78x | -- | 24.2381 (bit-exact) |
| **IU4 W4A4** | **559.3** | **1.21x** | **6.3411** | 34.1116 |

Across batch widths (t/s):

| batch | 32 | 48 | 64 | 96 | 128 | 512 | 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 147 | 281 | 341 | 394 | 431 | 462 | 455 |
| IU4 W4A4 | 239 | 290 | 375 | 458 | 527 | 559 | 552 |
| ratio | 1.62x | 1.03x | 1.10x | 1.16x | 1.22x | 1.21x | 1.21x |

No regression at any width. The 1.62x at batch 32 is partly the baseline's own
spill bug (see below), not an IU4 gain.

`tg128` is unchanged (13.78 -> 13.80): decode runs through MMVQ, which has no
matrix-core path on any arch.

### Speculative decoding (MTP) and MoE

IU4 *does* reach decode, but indirectly: it relieves the verify compute cliff, so
wider draft trees stay profitable. Retune `--spec-draft-n-max` after enabling it.

Qwen3.8-27B (dense), generation t/s:

| n_max | none | 2 | 4 | 8 | 12 | 16 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 13.8 | **19.6** | 19.3 | 18.4 | 19.5 | 18.2 | -- |
| IU4 | 13.8 | 20.3 | 19.7 | 20.9 | **21.6** | 21.3 | 20.6 |

Baseline peaks at n_max=2 and declines; IU4 keeps scaling to n_max=12, for
**+10% peak decode**. Note a plain `llama-bench -p 2,4,8` probe shows no
difference and is misleading -- prefill-shaped batches are not tree verification.

Qwen3.6-35B-A3B (MoE, requantised to ROCMI4):

| | pp512 | pp2048 | tg128 | MTP n=2 | MTP n=8 | MTP n=12 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1557 | 1545 | 68.2 | 78.4 | **80.7** | 79.3 |
| IU4 | **1774** | **1748** | 68.1 | 80.7 | **85.3** | 83.4 |
| ratio | 1.14x | 1.13x | 1.00x | 1.03x | 1.06x | 1.05x |

With the AEON DFlash drafter instead of MTP (generation t/s):

| n_max | baseline | IU4 |
|---|---:|---:|
| 4 | 74.8 | **85.9** (+14.8%) |
| 8 | 56.8 | 57.1 (+0.5%) |

DFlash at n_max=4 is the single biggest decode win from IU4 (+14.8%). The
collapse at n_max=8 is *unaffected*, which confirms that cliff is the MoE
expert-union bandwidth penalty (a wider draft activates more experts, so more
bytes are read) and not a compute limit -- the 4-bit lane cannot help there.

Best decode config measured overall: MoE + DFlash n_max=4 + IU4 = 85.9 t/s.

**On "60+ t/s" reports:** a 27B *dense* Q4 model cannot reach that -- 13.53 GiB
per token at 231.5 GB/s caps it at 15.9 t/s, 21.6 with MTP. 60-90 t/s is an MoE
result (35B-A3B is ~3B active), i.e. an architecture effect, not a 4-bit-lane
effect. Plain MoE decode here is 65-68 t/s before any speculation.

### Build flags

```
-DGGML_HIP_ROCMI4_IU4_MMQ=ON     # enable the IU4 MMQ path (default OFF)
-DGGML_HIP_ROCMI4_IU4_EXACT=ON   # bit-exact two-plane reference; requires IU4_MMQ
```

### Notes for anyone extending this

- **Exact mode cannot win.** Splitting int8 into a signed high plane and an
  unsigned low plane (`x == 16*(x>>4) + (x & 0xF)`) is bit-identical to the int8
  path, but costs 2 IU4 ops per K=16 -- only break-even at 2x rate -- plus
  packing and an extra accumulator. It is a correctness reference, not a
  fast path. Its PPL matches the int8 path to six significant figures, which is
  what proved the kernel and fragment layout correct.
- **Amdahl is the real ceiling.** Halving the matmul math bought 1.17x
  end-to-end, putting MMQ matmul at ~28% of prefill on this model. Further ISA
  wins need the other 72% (attention, activation quantize, norms).
- **Do the rounding at quantize time.** Truncating activations toward -inf in
  the inner loop is a systematic -7.5/16 LSB bias and gives PPL 1988. Rounding
  in-kernel fixes accuracy but costs ~9 ALU ops per dword and erases the entire
  speedup (540 -> 448). `quantize_mmq_q8_1<.., i4_grid=true>` puts activations on
  the 4-bit grid up front so the kernel's `>>4` is lossless and free.
- **K-order is free to permute.** A dot product is invariant under any
  permutation of K applied to both operands, so both packers emit the
  byte-interleaved order `[b+0,b+4,b+1,b+5,b+2,b+6,b+3,b+7]`, which falls out of
  one mask-and-shift-by-4 merge instead of a ~22-op nibble compaction.
- **Watch VGPR spills.** Adding `#pragma unroll` to the k01 loop (which the int8
  path deliberately omits) took spills from 14 to 412 and cost 3x.
- **Watch per-mmq_x register pressure.** `mul_mat_q` is compiled once per mmq_x,
  and mmq_x is chosen to equal the batch width, so a spill in one instantiation
  collapses that batch size and nothing else. Building tile_B in-kernel from an
  8-int tile spilled 136 words at mmq_x=64 and cost 2.5x *there* while pp512 was
  untouched. Having quantize emit pre-packed nibbles keeps tile_B at 4 registers
  and clears it. Do NOT instead disable the j0 unroll: that zeroes every spill
  but collapses pp512 from 552 to 233.
- **Pre-existing bug, unrelated to IU4:** the int8 kernel spills 46 words at
  mmq_x=32, which is why baseline pp32 (147 t/s) is *below* pp48 (281 t/s).
  This affects every quant type on this arch and is worth fixing separately.
- **Decode cannot benefit from any of this.** At 13.53 GiB and 13.80 t/s the 27B
  moves 200.5 GB/s against 231.5 GB/s achievable -- 87% of the memory roofline.
  Batch-1 decode is also MMVQ, which has no matrix-core path. The levers for
  decode are fewer bytes per token (smaller quant, 4-bit KV) or more tokens per
  pass (speculation), not faster math. Note the IU4 gain only starts at batch
  >=96; at typical MTP draft widths (2-8) the verify is still memory-bound and
  IU4 changes nothing.
- **The weight tile is 44 ints/row, not 76** (`MMQ_MMA_TILE_X_K_ROCMI4`): packed
  nibbles need only 32 ints of quantized data instead of 64. Worth ~1.7% at
  pp512 and ~9% at batch 64. `x_df` moves from +64 to +32 accordingly.
- **Occupancy is not the limiter.** LDS at mmq_x=128 is ~30KB = 2 workgroups/CU;
  capping mmq_x to 64 (~21KB, 3 workgroups/CU) measured identical (560.3 vs
  561.0). So halving the y-tile -- the obvious next LDS saving, and invasive
  since MMQ_TILE_Y_K has 79 references -- would buy nothing. Inside mul_mat_q
  the achieved rate is ~43 of 110.6 TOPS (39%); the gap is LDS/staging traffic.
- rocprofv3 pp512 breakdown after IU4: mul_mat_q 70.7%, gated_delta_net 7.8%,
  quantize_mmq_q8_1 5.6%, concat_non_cont 4.1%, unary_gated_op 4.0%,
  rms_norm 1.8%, flash_attn_tile 1.6%.
- Remaining accuracy work: QuaRot-style online Hadamard rotation should reclaim
  most of the +0.32 PPL. QuaRot reports <=0.47 on Llama2-70B W4A4; this is 0.32
  with no rotation at all.
