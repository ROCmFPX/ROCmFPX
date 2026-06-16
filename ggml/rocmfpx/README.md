# ROCmFPx Experimental Formats

This folder contains the reference layer for the proposed ROCmFP3, ROCmFP6, and
ROCmFP8 quantization family. It is intentionally separate from `ggml/rocmfp4/`
so the promoted ROCmFP4 GGUF formats and kernels are not affected while the new
layouts are evaluated.

Current status (June 16, 2026):
- CPU reference quantize/dequantize exists for all three formats.
- `Q3_0_ROCMFPX`, `Q6_0_ROCMFPX`, and `Q8_0_ROCMFPX` are registered as
  experimental GGUF tensor types.
- ROCm/HIP and Vulkan kernels support `CPY`, `GET_ROWS`, `SET_ROWS`, and
  `MUL_MAT`/`MUL_MAT_ID` for all three formats.
- Qwen3-0.6B BF16 smoke tests pass on CPU, ROCm0, and Vulkan0.
- Default quant presets now include lean coherency routing:
  - `Q3_0_ROCMFPX`: selective `Q5_K` on attention Q/O and early K/V, boosted
    FFN-down at `Q5_K`, selective FFN-gate at `Q6_0_ROCMFPX`, bulk FFN-up on
    `Q3_0_ROCMFPX`, embeddings/output at `Q4_0_ROCMFP4_FAST`.
  - `Q6_0_ROCMFPX`: early attention and boosted FFN-down at `Q8_0_ROCMFPX`,
    embeddings/output at `Q6_0_ROCMFPX`, bulk gate/up on `Q6_0_ROCMFPX`.
  - `Q8_0_ROCMFPX`: pure FP8-family preset.
- Opt-in `*_AGENT` presets boost attention/FFN routing for tool-call /
  Hermes / OpenClaw style workloads:
  - `Q3_0_ROCMFPX_AGENT`, `Q6_0_ROCMFPX_AGENT`, `Q8_0_ROCMFPX_AGENT`.
  - Routing is layered on top of LEAN; default presets are unchanged.
- FP3 and FP6 quantization use reconstruction-MSE scale selection per
  16-weight half-block.

## Validation Script Index

```text
scripts/check-rocmfpx-reference.sh        # CPU reference math
scripts/check-rocmfpx-qwen-all.sh         # core Qwen gates
scripts/check-rocmfpx-all.sh              # qwen-all + optional smokes
scripts/check-rocmfpx-summary.sh          # full JSON summary runner
scripts/sweep-rocmfpx-backend-ops.sh      # test-backend-ops per backend
scripts/sweep-rocmfpx-agent-size-table.sh # LEAN vs AGENT MiB/BPW
scripts/sweep-rocmfpx-perplexity.sh       # calibration PPL sweep
scripts/sweep-rocmfpx-decode-tune.sh      # decode-tune matrix
scripts/build-rocmfpx-agent-fixtures.sh   # proxy Hermes/OpenClaw AGENT GGUFs
```

## Layouts

All formats use 32-weight blocks.

| Format | Payload | Scale bytes | Block bytes | BPW | Purpose |
|---|---:|---:|---:|---:|---|
| `Q3_0_ROCMFPX` | 32 packed 3-bit codes | 2, one per 16 weights | 14 | 3.50 | Experimental low-bit candidate |
| `Q6_0_ROCMFPX` | 32 packed 6-bit codes | 2, one per 16 weights | 26 | 6.50 | Experimental quality candidate |
| `Q8_0_ROCMFPX` | 32 signed 8-bit codes | 1, one per 32 weights | 33 | 8.25 | Experimental high-quality reference |

`ROCmFP3` uses a tiny signed codebook: `0, +/-1, +/-2, +/-4`.
`ROCmFP6` uses signed-magnitude integer levels up to `31`.
`ROCmFP8` uses signed int8 levels clamped to `[-127, 127]`.

## Validation

Reference math only:

```bash
scripts/check-rocmfpx-reference.sh
```

Focused backend sweep from the experiment worktree:

```bash
cmake --build build-strix-rocmfp4 --target test-backend-ops -j 8
timeout 120 build-strix-rocmfp4/bin/test-backend-ops test -o MUL_MAT,GET_ROWS,CPY,SET_ROWS -b CPU
timeout 180 build-strix-rocmfp4/bin/test-backend-ops test -o MUL_MAT,GET_ROWS,CPY,SET_ROWS -b ROCm0
timeout 180 build-strix-rocmfp4/bin/test-backend-ops test -o MUL_MAT,GET_ROWS,CPY,SET_ROWS -b Vulkan0
```

Qwen3 BF16 coherency and decode-speed gates:

```bash
MODEL=/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-LEAN.gguf BACKEND=ROCm0 scripts/check-rocmfpx-qwen-coherency.sh
MODEL=/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-LEAN.gguf BACKEND=ROCm0 scripts/check-rocmfpx-qwen-bench.sh
MODEL=/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-LEAN.gguf BACKEND=ROCm0 scripts/check-rocmfpx-qwen-strict-json.sh
```
