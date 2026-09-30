# RDNA3 ROCmI4 autoregressive kernel acceleration plan

## Objective

Speed up real, batch-1 autoregressive token generation for:

`~/models/rocmfpx/cafonez/Qwen3.8-27B-ROCmI4/Qwen3.8-27B-Q4_0_ROCMI4.gguf`

on the ROCmFPX HIP backend, with all changes landing on `feature/rdna3-w4a4-iu4-mmq` from base commit `8fd06fe85`.

The primary acceptance metric is generated tokens/second, including the complete model graph. Speculative decoding, model changes, reduced context, altered sampling, lower-quality KV formats, skipped work, and benchmark-only shortcuts are excluded.

## What the ISA and existing measurements imply

The relevant RDNA3 primitives are already correctly split by workload:

- Decode (`n = 1`) uses Wave32 `V_DOT8_I32_IU4` through `ggml_cuda_dot8_iu4` / ROCmI4 MMVQ. This is the correct primitive for a matrix-vector operation.
- Prefill/large batches use `V_WMMA_I32_16X16X16_IU4` through `mma_iu4` and the ROCmI4 MMQ SRAM layout. WMMA is not a useful replacement for the bandwidth-bound batch-1 matvec.
- RDNA3 Wave32 reductions, lane shuffles, 16-byte loads, LDS, barriers, and software scheduling are the tools for fusing producer/consumer kernels. LDS use must remain below the occupancy cliff: prior measurements show the current approximately 30 KiB MMQ workgroup permits two residents, while a roughly 60 KiB workgroup loses that residency and regresses.
- The landed 16-byte LDS MMQ loads and MMQ epilogue work should be retained. MMQ tile sweeps, k-loop double buffering, and GDN operand prefetch have already been measured; reopening them without new evidence would waste time.

Measured branch facts establish the optimization budget:

- Baseline is about 42.5 generated tokens/s at depth 0, 41.7 at depth 8192, and 38.7 at depth 32768 with HIP graphs enabled.
- ROCmI4 MMVQ accounts for about 17.20 ms/token and streams the on-disk layout at about 94% of its layout-limited ceiling. Rewriting the dot-product loop can recover at most a few percent of that component and is not the first target.
- Approximately 1,414 small kernels plus about 3.6 ms of inter-kernel gaps are the remaining material decode lever. There are 436 `quantize_q8_1` launches per token even though several matvecs consume the same activation.
- The GDN path contributes repeated L2 norms, elementwise operations, state gathers/copies, and projection epilogues. The core `gated_delta_net_cuda` kernel itself is only about 0.21 ms/token, so optimizing its inner recurrence first cannot produce a meaningful end-to-end gain.
- HIP graphs are worth about +4.6% generation throughput and remain enabled in every primary result. Graphs-off runs are diagnostic only.

The realistic cumulative target is +3% to +6% batch-1 AR throughput. Any larger claim requires measured end-to-end evidence; it will not be extrapolated from kernel traces.

## Non-negotiable measurement contract

### Frozen workload

Use the exact GGUF, one GPU, batch 1, one generated token at a time, fixed seed, temperature 0, fixed prompt/tokenization, and identical CLI/environment between baseline and candidate. Keep:

- `GGML_HIP_ROCMI4_W4A4=ON`
- `GGML_HIP_FORCE_MMQ=ON`
- `GGML_HIP_GRAPHS=ON` for primary results
- F16 K/V cache, because the existing KV quantization sweep regressed every tested depth

Report at least:

1. depth 0, 128 generated tokens;
2. 2K prompt followed by 128 generated tokens;
3. 8K prompt followed by 128 generated tokens;
4. 32K prompt followed by 128 generated tokens, if memory permits;
5. server-style single-stream generation with the production launcher.

Prompt ingestion and token generation must be reported separately. A `-ub` change may be reported as a prefill/runtime configuration result, but it is not a kernel gain and does not satisfy this plan's AR acceptance gate.

### Experimental method

- Build baseline and candidate in separate directories from the same base, differing only by the candidate patch.
- Record commit, compiler/ROCm version, HIP architecture, build flags, GPU clocks, power state, temperature, and command lines.
- Warm the model and GPU before collection. Alternate baseline/candidate order (A/B/B/A) to avoid thermal and clock drift.
- Use at least 10 timed repetitions after warmup. Report all samples, median, mean, standard deviation/CV, and a bootstrap 95% confidence interval for the paired throughput delta.
- Primary acceptance requires a positive paired 95% interval and no context tier regressing by more than 1%. Kernel timing and dispatch counts explain a result; they never replace end-to-end throughput.
- Keep a runtime disable switch for each optimization so the same binary can perform an honest A/B when practical.

### Correctness gates

For every phase:

- Run the relevant `test-backend-ops` cases on CPU and HIP, including awkward dimensions, non-contiguous views, and multi-token fallbacks.
- Compare intermediate fused outputs against the unfused path. Reuse-only and operation-order-preserving work must be bit-exact; do not relax tolerance.
- Run deterministic generation and compare token IDs and logits. Pure reuse and exact epilogue fusions must remain token-exact.
- Run `llama-perplexity` on a fixed corpus before and after. Any statistically meaningful quality movement rejects the change.
- Verify graph capture and replay across changing token positions and context lengths; stable pointers must not imply stale activation data.

## WP0 — Reproduce and map the decode graph

Before changing a kernel:

1. Build the current branch with `scripts/build-rdna3-w4a4.sh` and establish the frozen depth-0/2K/8K/32K baseline.
2. Capture `rocprofv3` traces with graphs on and off. Graphs-off is used to map launches to GGML nodes; graphs-on remains the performance judge.
3. Add a debug-only graph census that records, for each executed node: operation, shape/strides, source tensor identities, data pointers, selected CUDA/HIP path, launch count, and fusion decision. It must be disabled by default and excluded from timing runs.
4. Count exact shared-activation groups: attention q/k/v, FFN gate/up, and any GDN projections that consume the identical logical tensor.
5. Dump the actual Qwen3.5 GDN node order around SSM conv, q/k L2 norms, alpha/beta projections, state gather, GDN, and state scatter. Do not design an adjacency-based matcher from model-source intuition alone.

Deliverable: a checked-in measurement table and graph census identifying how many launches each proposed fusion can actually remove. If the census does not show repeated quantization or the expected GDN chains, stop and re-rank the work rather than forcing the design.

## WP1 — Quantize each shared activation once per graph execution

This is the first implementation because it attacks AR dispatch overhead without making every MMVQ block reread F32 activations.

### Design

Add a graph-evaluation-scoped Q8_1 activation cache used by `ggml_cuda_mul_mat_vec_q`:

- Cache only an exact `src1` tensor identity initially. Key by device, stream, tensor object/data pointer, type, dimensions, strides, and required Q8_1 layout. Do not coalesce merely overlapping views.
- The validity epoch is one graph execution/replay. Activations at stable addresses change every generated token, so cached bytes must be regenerated once on every replay.
- Retain the destination allocation owner for the lifetime of the captured graph executable. A bare pointer from a destroyed `ggml_cuda_pool_alloc` is invalid and may be recycled. Tie stable buffers to the CUDA graph/context object and clear them when graph topology or tensor metadata changes.
- Within one execution, the first MMVQ consumer launches `quantize_q8_1`; later consumers use the same buffer and depend on the same stream-ordered producer. No cross-stream reuse is allowed without an explicit event dependency.
- Maintain a conservative uncached fallback for unsupported strides, aliases, split backends, graph-update failures, or memory pressure.
- Add `GGML_CUDA_DISABLE_SHARED_Q8_1` for same-binary A/B.

Keep MMVQ's `V_DOT8_I32_IU4` loop unchanged. This work removes redundant producer launches and activation reads; it does not fuse F32 quantization into each MMVQ workgroup, which prior byte arithmetic showed would add about 1 GB of L2 traffic per token.

### Files

- `ggml/src/ggml-cuda/mmvq.cu`
- `ggml/src/ggml-cuda/quantize.cu`
- `ggml/src/ggml-cuda/quantize.cuh`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- CUDA graph/context ownership files located during WP0
- focused backend tests

### Acceptance

- Q8_1 buffers are bit-identical to independently quantized buffers.
- Quantize launch count falls by the exact number predicted by WP0.
- No stale-buffer failure across at least 1,024 generated tokens, context growth, graph recapture, and alternating prompt shapes.
- Accept as a standalone patch at >=1.0% end-to-end AR gain, or retain as enabling infrastructure only if WP2 makes the cumulative gain >=1.5%.

### WP0/WP1 execution record — 2026-09-20

Environment: base `8fd06fe85566`, Radeon RX 7900 XTX (`gfx1100`, Wave32, 24 GiB), HIP 7.15.26333, AMD clang 23, `rocm-dev:10.0.0`, and the frozen model/build flags above. The primary command was:

`llama-bench -p 0 -n 128 -d 0,2048,8192,32768 -r 10 --delay 1 -o jsonl --progress`

Frozen pre-change baseline:

| depth | median tok/s | mean tok/s | stddev | CV |
|---:|---:|---:|---:|---:|
| 0 | 42.6043 | 42.5724 | 0.1052 | 0.247% |
| 2048 | 43.0824 | 42.9892 | 0.2946 | 0.685% |
| 8192 | 42.1635 | 42.0771 | 0.2733 | 0.650% |
| 32768 | 39.1005 | 39.0234 | 0.2431 | 0.623% |

Raw baseline samples, in depth order:

- 0: `[42.2735, 42.5994, 42.6068, 42.6081, 42.6100, 42.6143, 42.5991, 42.6110, 42.5999, 42.6017]`
- 2048: `[42.1510, 43.0982, 43.0848, 43.0829, 43.0827, 43.0784, 43.0839, 43.0768, 43.0821, 43.0712]`
- 8192: `[41.2992, 42.1637, 42.1650, 42.1678, 42.1618, 42.1657, 42.1656, 42.1633, 42.1605, 42.1587]`
- 32768: `[38.3315, 39.1030, 39.1057, 39.1018, 39.0951, 39.0952, 39.0981, 39.1014, 39.0997, 39.1023]`

The opt-in `GGML_CUDA_GRAPH_CENSUS` measured 2,100 compute nodes, 497 ROCmI4 MMVQ consumers, and 257 exact activation identities per decode graph. Group widths were `{1: 129, 2: 64, 3: 16, 4: 48}`. The exact multi-consumer classes were:

| activation producer/class | groups | consumers/group | redundant quantize launches | named consumers |
|---|---:|---:|---:|---|
| GDN `attn_norm` | 48 | 4 | 144 | qkv, alpha, beta, attention gate |
| full-attention `attn_norm` | 16 | 3 | 32 | q, k, v |
| FFN `attn_post_norm` | 64 | 2 | 64 | gate, up |
| single-consumer GLU/gated/GDN-output/final tensors | 129 | 1 | 0 | down/output/final projections |
| **total** | **257** | **497 consumers** | **240** | |

The same census predicts 128 remaining normalization-to-quantization boundaries for WP2, because all 128 multi-consumer groups are `MUL` outputs from the already fused RMSNorm/scale path. It also records these per-token launch-removal ceilings for later work: WP3 removes 144 GDN epilogue launches (48 layers times alpha add + alpha softplus/multiply + beta sigmoid), WP4 removes 96 q/k L2-normalization launches, and WP5 can remove 48 state-gather launches.

The measured block-0 GDN order is `SSM_CONV -> SILU -> q L2_NORM -> k L2_NORM -> alpha MMVQ -> RESHAPE -> ADD -> SOFTPLUS -> MUL -> beta MMVQ -> RESHAPE -> SIGMOID -> GATED_DELTA_NET -> VIEW -> VIEW -> CPY -> RMS_NORM -> MUL -> gate MMVQ -> SILU -> MUL -> output MMVQ`. Existing fusion already combines SSM convolution/SILU, softplus/multiply, GDN/state scatter, RMSNorm/multiply, and SiLU/multiply; the two L2 norms and projection epilogues remain separate as assumed by WP3/WP4.

WP1 was implemented with graph-owned stable buffers, an exact device/stream/tensor/data/type/shape/stride/layout key, a per-execution epoch, topology invalidation, no allocation during capture, allocation-failure fallback, and `GGML_CUDA_DISABLE_SHARED_Q8_1`.

Same-binary A/B/A/B results pool 20 samples per path. Bootstrap intervals are paired by repetition within each A/B leg (50,000 resamples):

| depth | disabled median | enabled median | median gain | paired mean gain (95% CI) | disabled -> enabled mean |
|---:|---:|---:|---:|---:|---:|
| 0 | 41.8466 | 43.0558 | +2.89% | +3.27% `[3.12%, 3.42%]` | 41.8220 -> 43.1901 |
| 2048 | 42.3320 | 43.4741 | +2.70% | +2.88% `[2.81%, 2.96%]` | 42.3431 -> 43.5642 |
| 8192 | 41.4258 | 42.5245 | +2.65% | +2.91% `[2.80%, 3.02%]` | 41.4307 -> 42.6351 |
| 32768 | 38.4813 | 39.3792 | +2.33% | +2.57% `[2.47%, 2.68%]` | 38.4943 -> 39.4844 |

Raw same-binary samples (`disabled run 1; disabled run 2` / `enabled run 1; enabled run 2`):

- 0 disabled: `[40.8772, 41.8555, 41.8360, 41.8376, 41.8286, 41.8339, 41.8085, 41.8202, 41.8118, 41.8305; 41.0221, 42.0268, 42.0217, 42.0397, 42.0101, 41.9878, 41.9972, 42.0024, 41.9897, 42.0033]`
- 0 enabled: `[42.1828, 43.0461, 43.0438, 43.0071, 43.0401, 43.0536, 43.0371, 43.0530, 43.0573, 43.0544; 42.5912, 43.5019, 43.4979, 43.5211, 43.5064, 43.5269, 43.5220, 43.5192, 43.5198, 43.5207]`
- 2048 disabled: `[41.4004, 42.3201, 42.3183, 42.3123, 42.3175, 42.3238, 42.3295, 42.3166, 42.3345, 42.3207; 41.6658, 42.5554, 42.5567, 42.5552, 42.5584, 42.5371, 42.5296, 42.5353, 42.5386, 42.5366]`
- 2048 enabled: `[42.6570, 43.4760, 43.4551, 43.4438, 43.4641, 43.4589, 43.4649, 43.4616, 43.4723, 43.4686; 43.0050, 43.8320, 43.8359, 43.8181, 43.8196, 43.8360, 43.8325, 43.8264, 43.8310, 43.8258]`
- 8192 disabled: `[40.5907, 41.4223, 41.4258, 41.4235, 41.4225, 41.4259, 41.4214, 41.4149, 41.4213, 41.4128; 40.7433, 41.6092, 41.6097, 41.6039, 41.6088, 41.6099, 41.6183, 41.6147, 41.6093, 41.6056]`
- 8192 enabled: `[41.7481, 42.5231, 42.5259, 42.5217, 42.5182, 42.5224, 42.5152, 42.5076, 42.5175, 42.5131; 42.1333, 42.9107, 42.9019, 42.9129, 42.9060, 42.9092, 42.9112, 42.9024, 42.9044, 42.8978]`
- 32768 disabled: `[37.7421, 38.4711, 38.4822, 38.4805, 38.4711, 38.4730, 38.4697, 38.4754, 38.4772, 38.4746; 37.9284, 38.6549, 38.6612, 38.6561, 38.6582, 38.6664, 38.6544, 38.6608, 38.6656, 38.6623]`
- 32768 enabled: `[38.6588, 39.3798, 39.3787, 39.3687, 39.3700, 39.3680, 39.3649, 39.3665, 39.3642, 39.3513; 39.0482, 39.7369, 39.7462, 39.7392, 39.7332, 39.7436, 39.7402, 39.7436, 39.7415, 39.7451]`

An 8-token `rocprofv3 --kernel-trace --stats` run reduced `quantize_q8_1<true>` dispatches from 3,897 to 2,313 (-1,584, -40.6%) and aggregate profiled quantizer time from 6.147 ms to 3.898 ms (-36.6%). The trace includes warmup/capture work; the steady decode graph census remains the exact 240-launch removal count. Stable cache storage is 2.33 MiB across 257 entries. MMVQ device code and resources are unchanged.

Correctness/robustness evidence: targeted HIP `test-backend-ops -b ROCm0 -o MUL_MAT` passed; deterministic temperature-0 text, all 16 token IDs, and every returned top-10 token probability were exact with the switch on/off; a 1,024-token graph-replay/context-growth run completed; all four depth tiers ran sequentially in one process (exercising graph updates); and fixed-corpus perplexity was identical at every reported chunk and final estimate (`9.7350 +/- 1.15572`). A production-style 80K-context server request was also text-exact and measured 41.22 tok/s disabled versus 42.87 tok/s enabled (+4.01%, single sample).

## WP2 — Produce reusable Q8_1 alongside fused RMSNorm/scale

After WP1 proves ownership and graph replay, eliminate the remaining common normalization-to-quantization boundary.

Implement an RDNA3 decode-specialized producer that writes both the normal F32 RMSNorm/scale output required by graph semantics and its block-Q8_1 representation for downstream MMVQs.

Use the existing RMS reduction order. After the row scale and learned scale are applied in F32, each Wave32 computes the Q8_1 block maximum/scale and packs the same values that standalone `quantize_q8_1` would observe. The stored F32 output and Q8_1 blocks must be bit-exact to the two-kernel sequence.

Graph plumbing should recognize only the measured pattern, allocate the secondary Q8_1 output from the WP1 stable cache, and let later MMVQs consume it. Do not create a model-specific hidden tensor or alter GGUF weights. Fall back when the norm shape, strides, multiply/broadcast pattern, or consumers differ.

Keep LDS small, inspect VGPR/SGPR/LDS use, and preserve at least the occupancy of the existing norm kernel. A fused kernel that removes a launch but spills or lowers residency is rejected.

Files: `norm.cu`, `quantize.cu[h]`, `ggml-cuda.cu`, and the WP1 cache/context code.

Acceptance: F32 and Q8_1 outputs are bit-exact; resource reports show no scratch spills or material occupancy loss; launch count falls as predicted; cumulative WP1+WP2 AR improvement is >=1.5% with graphs enabled.

### WP2 execution record — 2026-09-20

The decode-only RMSNorm/multiply variant now writes the ordinary F32 destination and ROCmI4 block-Q8_1 cache entry in one kernel. It is selected only for contiguous, 512-aligned F32 activations with a measured ROCmI4 MMVQ consumer; every other shape/backend/path retains the previous kernel and standalone quantizer. `GGML_CUDA_DISABLE_FUSED_RMS_Q8_1` provides same-binary rollback.

An 8-token profiler run removed exactly 1,152 standalone quantizer dispatches, or the predicted 128 per graph execution (`2,313 -> 1,161`). The 1,024-thread kernel moves from 10 to 21 VGPRs, uses zero static shared memory and zero local/spill memory in both variants, and retains two active blocks per multiprocessor. Runtime dynamic shared memory remains the same 128-byte reduction buffer.

Same-binary A/B/B/A results (20 samples/path) are positive at all depths:

| depth | WP1-only median | WP1+WP2 median | median gain | paired mean gain (95% bootstrap CI) | WP1-only -> WP1+WP2 mean |
|---:|---:|---:|---:|---:|---:|
| 0 | 42.9287 | 43.2709 | +0.80% | +0.83% `[0.80%, 0.87%]` | 43.0647 -> 43.4234 |
| 2048 | 43.3433 | 43.5722 | +0.53% | +0.41% `[0.35%, 0.47%]` | 43.5400 -> 43.7194 |
| 8192 | 42.3887 | 42.5984 | +0.49% | +0.48% `[0.45%, 0.51%]` | 42.5710 -> 42.7746 |
| 32768 | 39.2771 | 39.4813 | +0.52% | +0.46% `[0.41%, 0.51%]` | 39.4376 -> 39.6179 |

Raw samples (`WP1-only run 1; run 2` / `WP1+WP2 run 1; run 2`):

- 0 WP1-only: `[42.5122, 43.4111, 43.4119, 43.3715, 43.3920, 43.4002, 43.4004, 43.4002, 43.3882, 43.4007; 42.0094, 42.8866, 42.8919, 42.9168, 42.9206, 42.9277, 42.9175, 42.9297, 42.9117, 42.8943]`
- 0 WP1+WP2: `[42.9394, 43.7371, 43.7815, 43.7480, 43.7490, 43.7695, 43.7358, 43.7735, 43.7691, 43.7478; 42.4340, 43.2532, 43.2765, 43.2484, 43.2314, 43.2632, 43.2438, 43.2504, 43.2506, 43.2653]`
- 2048 WP1-only: `[43.0733, 43.9037, 43.9062, 43.9112, 43.8994, 43.9003, 43.9039, 43.9053, 43.9004, 43.9019; 42.5491, 43.3437, 43.3414, 43.3422, 43.3429, 43.3415, 43.3312, 43.3308, 43.3367, 43.3349]`
- 2048 WP1+WP2: `[43.3014, 44.0256, 44.0323, 44.0352, 44.0170, 44.0149, 44.0246, 44.0149, 44.0131, 44.0189; 42.8514, 43.5673, 43.5678, 43.5540, 43.5666, 43.5580, 43.5373, 43.5510, 43.5603, 43.5767]`
- 8192 WP1-only: `[42.1457, 42.9189, 42.9237, 42.9283, 42.9212, 42.9100, 42.9012, 42.9043, 42.9078, 42.9044; 41.6417, 42.3720, 42.3849, 42.3774, 42.3838, 42.3608, 42.3770, 42.3924, 42.3790, 42.3850]`
- 8192 WP1+WP2: `[42.4170, 43.0944, 43.0921, 43.0868, 43.0933, 43.0895, 43.0915, 43.0919, 43.0977, 43.0943; 41.9251, 42.5897, 42.5993, 42.5937, 42.5915, 42.5813, 42.5879, 42.5869, 42.5910, 42.5975]`
- 32768 WP1-only: `[39.0754, 39.7272, 39.7396, 39.7437, 39.7330, 39.7309, 39.7346, 39.7288, 39.7356, 39.7340; 38.6118, 39.2772, 39.2755, 39.2708, 39.2763, 39.2648, 39.2695, 39.2769, 39.2729, 39.2731]`
- 32768 WP1+WP2: `[39.2874, 39.8704, 39.8799, 39.8756, 39.8748, 39.8773, 39.8816, 39.8725, 39.8719, 39.8720; 38.9024, 39.4812, 39.4803, 39.4700, 39.4815, 39.4770, 39.4730, 39.4795, 39.4759, 39.4729]`

Thirty-two deterministic token IDs and all returned top-10 probabilities are exact against the WP1-only path. Targeted HIP RMSNorm backend tests pass. WP2 therefore clears its cumulative acceptance gate and remains enabled.


## WP3 — Fuse GDN projection epilogues into ROCmI4 MMVQ

Target the exact Qwen3.5 chains found by WP0:

1. beta projection -> sigmoid;
2. alpha projection -> bias add -> softplus -> multiply by `ssm_a`.

Extend `ggml_cuda_mm_fusion_args_host/device` with a small explicit MMVQ post-op descriptor and optional broadcast multiplier. Add matchers before the generic mul-mat-plus-add matcher so the longer chain is not consumed early. Require exact source edges, types, shapes, strides, broadcast semantics, and a single supported consumer.

In MMVQ writeback, apply the same sigmoid or softplus/multiply device operations after the existing accumulation and bias. Share scalar device helpers with `unary.cu`; do not duplicate formulas that can drift. Do not change accumulation order or the IU4 dot loop. Compile separate fusion variants so unrelated MMVQs pay no branch/register cost.

Files: `mmvq.cu`, the current fusion-argument declaration, `unary.cu[h]`, and `ggml-cuda.cu`.

Acceptance: bit-exact final output; predicted two post-projection launches per GDN layer disappear; no MMVQ VGPR-class, scratch, or occupancy regression; >=0.5% additional AR gain.

### WP3 execution record — 2026-09-20

Same-binary A/B/B/A with `GGML_CUDA_DISABLE_GDN_MMVQ_EPILOGUE` (20 samples/path) is positive at every frozen depth. The matcher covers the measured Qwen3.8-27B chains `MMVQ -> RESHAPE -> SIGMOID` and `MMVQ -> RESHAPE -> ADD -> SOFTPLUS -> MUL(ssm_a)` and writes the epilogue in MMVQ writeback. Rollback is `GGML_CUDA_DISABLE_GDN_MMVQ_EPILOGUE`.

| depth | disabled median | enabled median | median gain | paired mean gain (95% bootstrap CI) | disabled -> enabled mean |
|---:|---:|---:|---:|---:|---:|
| 0 | 43.7458 | 44.6062 | +1.97% | +2.06% `[1.88%, 2.25%]` | 43.7134 -> 44.6152 |
| 2048 | 44.0299 | 44.8828 | +1.94% | +1.92% `[1.86%, 1.97%]` | 43.9787 -> 44.8214 |
| 8192 | 43.0136 | 43.7634 | +1.74% | +1.77% `[1.61%, 1.94%]` | 43.0141 -> 43.7757 |
| 32768 | 39.7960 | 40.4413 | +1.62% | +1.69% `[1.52%, 1.87%]` | 39.7995 -> 40.4730 |

Resource inspection of the ROCmI4 MMVQ writeback (`ncols=1`, 32 threads) reports `base_regs=15 fused_regs=22`, zero shared/local/spill in both variants, and occupancy `64` active blocks/CU for both. Unrelated MMVQs keep the `has_fusion=false` kernel. A review pass required `reshape->src[0]==mm` on the softplus arm (now landed); isolating `post_op` from other fused MMVQ variants was left as a non-blocking residual because occupancy did not change.

WP3 therefore clears its >=0.5% additional AR gate and remains enabled.

## WP4 — Fuse SSM convolution with q/k L2 normalization

Proceed only if WP0 confirms a safely matchable graph segment.

Qwen3.5 decode groups q, k, and v channels in 128-wide heads. The SSM convolution maps a 128-thread block to that natural group. For q/k groups, retain convolution/SILU results, reproduce the standalone `l2_norm_f32<32>` reduction order, and write directly to q/k norm destinations. For v groups, write the ordinary convolution output. At `n_tokens == 1`, this requires about 512 bytes of temporary LDS per group and should not affect occupancy.

The host fusion must validate the complete dependency pattern and all view offsets. It may write multiple graph destinations, but must not skip an intervening node with an external consumer. If current contiguous-skip fusion cannot represent the actual topological order, add a small pre-execution fusion-plan/marking pass; do not reorder arbitrary nodes at runtime. Keep original kernels for prefill, unsupported dimensions/layouts, and non-Qwen patterns.

Acceptance: convolution, q norm, and k norm outputs are bit-exact; two L2 launches per GDN layer disappear; no LDS/VGPR occupancy regression; >=0.5% additional AR gain or cumulative accepted gain exceeds 3%.

### WP4 execution record — 2026-09-20

Decode-only custom matcher `ggml_cuda_try_fuse_ssm_conv_qk_l2` extends the existing SSM_CONV+SILU launch so each 128-thread block also writes the q or k L2 destination when `n_t==1` and `head_k_dim==128`. SILU remains fully written for the unnormalized v VIEW. Rollback is `GGML_CUDA_DISABLE_SSM_CONV_L2`. A review pass judged the matcher/kernel/host path correct for Qwen3.8-27B batch-1 decode.

Census (`GGML_CUDA_GRAPH_CENSUS=1`, llama-bench `-n 1 -d 0`): 48/48 GDN layers match `SSM_CONV fusion=accepted nodes=6 last_op=L2_NORM`. Standalone `path=L2_NORM` exec lines are gone (96 L2 launches removed per graph). Resource inspection of the d_conv=4 128-thread kernel:

`cuda_graph_census_ssm_conv_l2_resources threads=128 base_regs=20 fused_regs=22 base_shared=0 fused_shared=512 base_local=0 fused_local=0 base_blocks=16 fused_blocks=16`

Same-binary A/B/B/A with `GGML_CUDA_DISABLE_SSM_CONV_L2` (20 samples/path):

| depth | disabled median | enabled median | median gain | paired mean gain (95% bootstrap CI) | disabled -> enabled mean |
|---:|---:|---:|---:|---:|---:|
| 0 | 44.1796 | 44.7129 | +1.21% | +1.52% `[1.33%, 1.71%]` | 44.2976 -> 44.9716 |
| 2048 | 44.4819 | 44.9601 | +1.07% | +1.42% `[1.21%, 1.64%]` | 44.5632 -> 45.1983 |
| 8192 | 43.4907 | 43.8556 | +0.84% | +1.20% `[0.96%, 1.44%]` | 43.5423 -> 44.0664 |
| 32768 | 40.1683 | 40.5364 | +0.92% | +1.22% `[1.02%, 1.42%]` | 40.2406 -> 40.7329 |

Raw samples (`disabled run 1; run 2` / `enabled run 1; run 2`):

- 0 disabled: `[43.4420, 44.1820, 44.1640, 44.1748, 44.1707, 44.1681, 44.1771, 44.1591, 44.1568, 44.1438; 43.8774, 44.5685, 44.5880, 44.5759, 44.5683, 44.5734, 44.5814, 44.5731, 44.5568, 44.5513]`
- 0 enabled: `[43.9542, 44.6404, 44.6329, 44.6793, 44.6402, 44.6421, 44.6132, 44.6542, 44.6137, 44.6416; 44.7465, 45.4464, 45.4513, 45.4601, 45.4267, 45.4422, 45.4191, 45.4493, 45.4310, 45.4485]`
- 2048 disabled: `[43.8519, 44.4820, 44.4791, 44.4777, 44.4819, 44.4807, 44.4724, 44.4682, 44.4695, 44.4671; 44.1420, 44.7981, 44.7905, 44.7762, 44.7734, 44.7734, 44.7685, 44.7709, 44.7735, 44.7668]`
- 2048 enabled: `[44.3069, 44.8824, 44.8732, 44.8753, 44.8711, 44.8786, 44.8695, 44.8590, 44.8663, 44.8621; 45.0378, 45.6563, 45.6438, 45.6363, 45.6340, 45.6491, 45.6414, 45.6382, 45.6482, 45.6369]`
- 8192 disabled: `[42.8590, 43.4696, 43.4788, 43.4882, 43.4931, 43.4813, 43.4799, 43.4656, 43.4750, 43.4843; 43.1364, 43.7289, 43.7275, 43.7214, 43.7260, 43.7237, 43.7306, 43.7277, 43.7303, 43.7196]`
- 8192 enabled: `[43.2256, 43.7589, 43.7560, 43.7574, 43.7514, 43.7468, 43.7603, 43.7514, 43.7438, 43.7482; 43.9510, 44.4811, 44.4905, 44.4892, 44.4874, 44.4778, 44.4769, 44.4934, 44.4904, 44.4902]`
- 32768 disabled: `[39.6398, 40.1555, 40.1588, 40.1528, 40.1623, 40.1516, 40.1623, 40.1743, 40.1557, 40.1575; 39.9187, 40.4157, 40.4321, 40.4288, 40.4231, 40.4190, 40.4307, 40.4203, 40.4227, 40.4307]`
- 32768 enabled: `[39.9728, 40.4484, 40.4510, 40.4484, 40.4479, 40.4474, 40.4476, 40.4440, 40.4499, 40.4509; 40.6218, 41.1068, 41.1133, 41.1065, 41.1107, 41.1192, 41.1137, 41.1183, 41.1146, 41.1246]`

Deterministic temperature-0 generation (`-n 64 --seed 4242`) is text-exact with the switch on/off. Targeted HIP `test-backend-ops -b ROCm0` is `L2_NORM` 20/20 OK and `SSM_CONV` 45/45 OK. A dedicated review pass judged the matcher/kernel/host path correct for Qwen3.8-27B batch-1 decode (skip stops at the later L2; SILU remains fully written for v VIEWs). Cumulative vs the frozen pre-change baseline means is +5.64% / +5.14% / +4.73% / +4.38% at depths 0/2K/8K/32K. WP4 therefore clears both its additional-AR and cumulative 3% gates and remains enabled.

## WP5 — Remove GDN state gather only with an explicit cache contract

The remaining 48 `k_get_rows_float_vec` launches cost about 0.29 ms/token plus gaps. The backend already fuses the post-GDN snapshot scatter; do not duplicate it.

If this remains a top measured cost, extend the GDN operation/backend contract so the kernel reads selected state directly from cache using sequence IDs. Preserve the existing API as fallback and implement identical CPU/reference semantics. Validate rollback slots, multi-sequence batches, partial cache writes, and graph replay. This is an engine change, not an unsafe pointer substitution in the dispatcher.

Acceptance requires bit-exact states/outputs, complete GDN cache tests, and >=0.5% additional end-to-end AR gain.

### WP5/WP6 status after WP4 — 2026-09-20

WP1 through WP4 remain enabled. Cumulative AR vs the frozen pre-change baseline already exceeds the 3% depth-0 gate (enabled means +5.64% / +5.14% / +4.73% / +4.38% at 0/2K/8K/32K). The WP2-era 8-token `rocprofv3` stats still list `k_get_rows_float_vec` at 1.50% of profiled kernel time (432 calls, 2.84 ms). That is still a real launch, but removing it needs an explicit GDN cache-read contract rather than another dispatcher fusion. WP5/WP6 are therefore left as follow-up: do not start them until a post-WP4 graphs-off trace still shows a kernel above 3% of token wall time, or a controlled gather ablation proves >=0.5% end-to-end AR.

### WP5 gate verdict - 2026-09-20 post-WP4 trace

Post-WP4 graphs-off `rocprofv3 --kernel-trace` decode census (HEAD binary,
128 tokens, MMVQ count 55,857 = 436/token x 128.1 steps, self-consistent):
`k_get_rows_float_vec` is 48.3 launches/token, 337 us/token traced = **1.5%
of the 22.54 ms unprofiled d0 wall**. Traced durations are an inflated upper
bound (Revision 2 rule); dispatch floor plus deflated execution puts it nearer
1.1%. That is below the 3% trace gate, and the ablation alternative
(>=0.5% end-to-end) cannot clear its own bar with an upper bound this small
against an engine-level GDN cache-read contract. **WP5 stays closed.** The
same census re-ranks the P-5 fusion batch; see Revision 4 of the master plan
(`docs/rocmfpx/plans/rdna3-gfx1100-tg-pp-acceleration.md`).

## WP6 — Profile-gated kernel tuning, not fishing

After fusion, reprofile. Tune a kernel only when it exceeds 3% of token wall time or a controlled ablation establishes an end-to-end ceiling.

Allowed evidence-driven work includes RDNA3 Wave32 broadcasts for truly uniform scalars, removal of redundant loads found in disassembly, aligned vector loads, and scheduling independent VALU/memory work without lengthening dependencies. Context-sensitive flash-attention work is allowed only if 8K/32K traces show it is material; existing GQA sharing must be measured before proposing a replacement.

For every variant, inspect GCN ISA and resource reports. Confirm decode still contains `V_DOT8_I32_IU4`, prefill contains `V_WMMA_I32_16X16X16_IU4`, intended 16-byte loads are emitted, barriers are minimal, and VGPR/LDS changes do not lower occupancy.

Do not keep sub-0.5% isolated tweaks unless they compose into a statistically positive end-to-end patch.

## Explicitly rejected paths

- Speculative decoding or multi-token verification.
- Changing weights, quantization, context, output length, sampling, prompt, or quality settings.
- KV quantization: measured 5-10% slower.
- Disabling HIP graphs: measured about 4.6% slower.
- Fusing F32 quantization independently into every MMVQ workgroup: extra F32 rereads exceed launch savings.
- Repeating the MMQ tile sweep, GDN prefetch experiment, large-LDS double buffering, or MMVQ dot-loop rewrite without new profile evidence.
- Counting `-ub 1024`, prefill-only gains, microbenchmarks, profiler-inflated durations, or graphs-off numbers as AR kernel speedup.

## Commit sequence

1. `bench: freeze RDNA3 ROCmI4 AR baseline and graph census`
2. `hip: reuse Q8_1 activations across ROCmI4 MMVQ consumers`
3. `hip: emit shared Q8_1 from fused RMSNorm decode producer`
4. `hip: fuse Qwen3.5 GDN projection epilogues into MMVQ`
5. `hip: fuse SSM conv and qk L2 normalization on RDNA3 decode`
6. Optional cache/tuning work only after its gate is met
7. `docs: record accepted and rejected AR experiments with raw data`

Each performance commit carries a disable switch, correctness evidence, dispatch-count delta, resource delta, and end-to-end A/B table. Failed experiments are reverted but documented.

## Definition of done

- Accepted code is on `feature/rdna3-w4a4-iu4-mmq` and the tree is clean.
- Batch-1 single-stream AR throughput improves by at least 3% at depth 0 and remains improved at 2K/8K/32K, with paired 95% confidence interval above zero.
- HIP graphs remain enabled; no speculative path or workload shortcut exists.
- Deterministic generation is token-exact for order-preserving fusions, backend tests pass, and perplexity does not regress.
- Final report includes raw samples, commands, dispatch counts, traces, ISA/resource inspection, rejected attempts, and an updated roofline.
