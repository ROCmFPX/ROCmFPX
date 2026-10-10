# ROCmI4 W4A4 / native IU4 experiment

ROCmI4 normally uses the exact int8 path (IU8 WMMA MMQ, unpack+DOT4 MMVQ).
An optional W4A4 build instead quantizes activations to signed 4-bit and uses
native IU4 hardware on the RDNA3 family (`gfx110x` / `gfx115x`):

```sh
# discrete RDNA3 (RX 7900 XTX)
cmake -S . -B build -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DGGML_HIP_ROCMI4_W4A4=ON -DGGML_VULKAN=OFF
# or: scripts/build-rdna3-w4a4.sh
```

The option defaults to `OFF`. Device code is selected when the runtime GPU
reports RDNA3 (`amd_wmma_iu4_available`). Other architectures keep the exact
int8 path.

## Accuracy and scope

This is a deliberately lossy activation-quantization experiment, not a new
exact ROCmI4 representation. ROCmI4 weights are unchanged.

- **MMQ (prefill / large-batch):** packed IU4 WMMA (`V_WMMA_I32_16X16X16_IU4`).
  Accumulator scale is `C*16` before `dA*dB` (WMMA IU4 units).
- **MMVQ (decode / n=1):** activations packed in the same Q4_0 nibble layout
  as the weights (`lo = elem j`, `hi = elem j+16`) and dotted with
  `V_DOT8_I32_IU4`. DOT8 accumulators are true integer sums (no `*16`).
  Exact MMVQ remains the default in non-W4A4 builds (unpack + `V_DOT4_I32_IU8`
  against Q8 activations).

It does not affect ROCmFP2, ROCmFP3, or other tensor types.

Do not enable this option in builds that require exact backend agreement. A W4A4 build advertises the `ROCMI4_W4A4=1` backend feature. Backend operation tests retain the normal `5e-4` NMSE limit for exact ROCmI4 and use a `1e-2` limit only when that feature is present. This keeps expected IU4 activation quantization error bounded and still fails regressions above the declared limit; it does not relax any other tensor type or backend.

Across five randomized gfx1151 qualification runs, the highest observed NMSE was `0.005883` for `MUL_MAT` and `0.004868` for `MUL_MAT_ID`.

## Measured tradeoff

On the development gfx1151 host with Qwen3.8-27B Q4_0_ROCMI4, a fixed-shape
pp512 test measured 566.46 prompt tokens/s versus 465.50 tokens/s for exact
int8 MMQ. Plain non-speculative tg128 stayed near 13.8 tokens/s.

The practical decode gain appears when strict MTP batches target verification.
A matched 10-task HumanEval pilot measured 49.40 tokens/s mean with W4A4 versus
41.63 tokens/s with exact int8 MMQ, an 18.66 percent gain. A full W4A4
HumanEval run measured 44.39 tokens/s mean and 45.23 tokens/s median. A
25-chunk perplexity sample increased by about 5.4 percent, so the speed path
remains opt-in.

On RX 7900 XTX (`gfx1100`) with the same Qwen3.8-27B Q4_0_ROCMI4 GGUF,
W4A4 pp512 measured 1277.66 ± 213.32 t/s versus 1018.28 ± 136.80 t/s exact
(~+25%). W4A4-MMVQ tg128 measured 42.42 ± 0.15 t/s versus 42.26 ± 0.10 t/s
exact (flat; decode remains HBM-bound even with `V_DOT8_I32_IU4`). The same
W4A4 binary passed `test-rocmi4-iu4-dot` and `test-backend-ops` MUL_MAT
`q4_0_rocmi4` 12/12 (including `n=1`) plus MUL_MAT_ID 1/1.

Treat these values as host-specific qualification data, not a universal
performance guarantee. Re-run exact-versus-W4A4 quality and throughput checks
on the intended model before deployment.

## Rollback

Disable `GGML_HIP_ROCMI4_W4A4` and rebuild. Because the exact implementation
remains in the same source tree and is the default branch of every dispatch,
rollback requires no model conversion or data changes.
