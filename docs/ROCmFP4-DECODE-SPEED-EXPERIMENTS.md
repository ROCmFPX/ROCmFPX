# ROCmFP4 Decode Speed Experiments

This branch is for controlled decode-speed work on top of the stable ROCmFP4
tree. The default build remains unchanged; every tuning path below is opt-in.

## Branch

```bash
git switch experiment-ROCmFP4-decode-speed
```

The branch was created from `4795079b0`.

## Rules

- Stability wins over decode speed.
- Do not change the ROCmFP4 GGUF layout or quantized value semantics for a speed
  experiment.
- Do not revert the `SPLIT_MODE_TENSOR` graph-reuse safety guard unless the
  replacement proves it cannot leave dangling backend refs.
- Promote only changes that pass focused ROCmFP4 kernel guards and the real
  slowed-model decode guard.
- Keep rejected experiments documented so they are not repeated.

## Build Profiles

Use a separate build directory for every profile:

```bash
ROCMFP4_DECODE_TUNE=strix-moe-rpb1 \
BUILD_DIR=build-strix-rocmfp4-rpb1 \
scripts/build-strix-rocmfp4-mtp.sh
```

The same `ROCMFP4_DECODE_TUNE` profiles are also accepted by
`scripts/build-rocmfp4.sh`, so the RDNA2, RDNA3, and RDNA4 wrappers inherit
them.

Available profiles:

| Profile | Effect | Intended use |
|---|---|---|
| `stable` | Current default flags | Baseline |
| `strix-moe-rpb1` | `-DGGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK=1` | MoE decode launch-shape check |
| `strix-moe-rpb2` | `-DGGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK=2` | Explicit current default |
| `strix-moe-rpb3` | `-DGGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK=3` | MoE decode launch-shape check |
| `strix-moe-rpb4` | `-DGGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK=4` | MoE decode launch-shape check |
| `strix-nwarps1` | `-DGGML_ROCMFP4_RDNA35_NWARPS=1` | RDNA3.5 MMVQ launch check |
| `strix-nwarps2` | `-DGGML_ROCMFP4_RDNA35_NWARPS=2` | Explicit current default |
| `strix-nwarps4` | `-DGGML_ROCMFP4_RDNA35_NWARPS=4` | RDNA3.5 MMVQ launch check |
| `strix-mmid3` | `-DGGML_ROCMFP4_RDNA35_MMID_MAX_BATCH=3` | MMVQ/MMQ routing threshold check |
| `strix-mmid4` | `-DGGML_ROCMFP4_RDNA35_MMID_MAX_BATCH=4` | MMVQ/MMQ routing threshold check |

You can still pass direct HIP flags when a profile is too narrow:

```bash
CMAKE_HIP_FLAGS="-DGGML_ROCMFP4_FATTN_V_NTHREADS=4" \
BUILD_DIR=build-strix-rocmfp4-fa-v4 \
scripts/build-strix-rocmfp4-mtp.sh
```

## Graph Build Timing

If a model slowed down after the tensor-split graph-reuse safety change, first
measure graph rebuild overhead before changing kernels:

```bash
LLAMA_GRAPH_BUILD_TIMING=1 \
BUILD_DIR=build-strix-rocmfp4 \
scripts/check-rocmfp4-qwen35-a3b-mtp-regression.sh
```

The perf footer prints:

- `graph rebuilds`
- `graph build time`
- `graph reset+build`

If rebuild time is a small fraction of total eval time, focus on kernels and
speculative profile settings. If it is large, work on a safe graph-reuse
replacement rather than reverting the tensor-split guard.

## Code Tweak Map

| Area | File | Safe experiment |
|---|---|---|
| ROCmFP4 MoE MMVQ launch shape | `ggml/src/ggml-cuda/mmvq.cu` | `GGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK` |
| RDNA3.5 ROCmFP4 MMVQ warps | `ggml/src/ggml-cuda/mmvq.cu` | `GGML_ROCMFP4_RDNA35_NWARPS` and `GGML_ROCMFP4_RDNA35_NWARPS_MAX_NCOLS` |
| MMVQ vs MMQ routing | `ggml/src/ggml-cuda/mmvq.cu` | `GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH` |
| ROCmFP4 FlashAttention shape | `ggml/src/ggml-cuda/fattn-vec.cuh` | `GGML_ROCMFP4_FATTN_*` compile flags |
| Build profiles | `scripts/build-strix-rocmfp4-mtp.sh` | Add opt-in profiles only |
| Graph rebuild timing | `src/llama-context.cpp` | `LLAMA_GRAPH_BUILD_TIMING=1` |
| Runtime queue behavior | `common/common.cpp` | Runtime-only `GPU_MAX_HW_QUEUES` A/B |

## Validation

At minimum, run the relevant focused model guard plus the ROCmFP4 runtime guards:

```bash
BUILD_DIR=build-strix-rocmfp4-rpb1 \
scripts/check-rocmfp4-qwen35-a3b-mtp-regression.sh

BUILD_DIR=build-strix-rocmfp4-rpb1 \
scripts/check-rocmfp4-rocm-runtime-regression.sh
```

For a candidate that looks faster, also run:

```bash
INCLUDE_QWEN35_A3B_GUARD=1 \
BUILD_DIR=build-strix-rocmfp4-rpb1 \
scripts/check-rocmfp4-all-regression.sh
```

## Larger Quantization Research

ROCmFP8, ROCmFP6, and ROCmFP3 should be treated as new GGUF formats, not as
kernel-only speed patches. A real format needs all of the following:

- A precise block layout and scale policy.
- CPU quantize/dequantize reference code.
- GGUF type IDs and quantization presets.
- ROCm/HIP copy, dequant, MMVQ, MMQ, and FlashAttention support.
- Vulkan decode and copy support if the format is user-visible.
- Focused numerical tests and end-to-end model regression guards.

Recommended order:

1. `ROCmFP8`: easiest to prototype because values are byte-aligned. It can be a
   quality-oriented intermediate format or a faster fallback when FP4 quality is
   too tight.
2. `ROCmFP6`: more complex because packing is awkward, but it may be useful as a
   quality/speed middle ground if memory bandwidth still improves.
3. `ROCmFP3`: highest risk. It needs stronger tensor-aware protection because
   the value grid is very small and quality regressions are likely.

Do not call a format `ROCmFP8`, `ROCmFP6`, or `ROCmFP3` in release notes unless
the README clearly states that it is this repository's experimental GGUF format,
not an AMD hardware-native data type.
