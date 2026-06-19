# ROCmFPX Handoff

This is the working handoff for the experimental ROCmFPX branch. It is written
for the next reviewer or agent that needs to continue the family without
reconstructing the intent from commit history.

## What This Branch Is

ROCmFPX is a model-weight quantization family that sits beside ROCmFP4 in this
tree. It is not a new K/V-only compression layer. The model-file presets are:

- `Q3_0_ROCMFPX`
- `Q6_0_ROCMFPX`
- `Q8_0_ROCMFPX`

The ROCmFP4 path stays the reference for block shape, scale discipline, kernel
coverage, and dequant semantics. K/V cache flags remain runtime settings in
llama.cpp and should not be treated as model quant formats.

## Current Branch State

The local experimental branch carries:

- ROCmFPX format docs and validation notes
- agent-oriented quant wrapper scripts
- agentic smoke tests for chat, coding, JSON, tool-call, coherency, and stream
- ROCmFP4 and ROCmFPX cache handling adjustments
- the K-cache coherency guard that promotes fp3 K cache to a safer higher-bit
  type at runtime

## Family Contract

Keep these rules stable unless a test proves otherwise:

- Use 32-weight blocks.
- Use finite unsigned UE4M3 scale bytes.
- Reject invalid scale bytes such as `0x7f` and sign-bit scale bytes.
- Use explicit integer-code-times-decoded-scale dequant math.
- Treat ROCmFP4 as the kernel/quant template, not as a separate inference stack.
- Preserve normal llama.cpp runtime features:
  MTP, EAGLE3, speculative decoding, RoPE/attention scaling, and tool calling.

ROCmFPX code ranges are intentionally simple:

| Format | Code range | Scale policy |
|---|---|---|
| `Q3_0_ROCMFPX` | `0, +/-1, +/-2, +/-4` | 2 scales, 1 per 16 weights |
| `Q6_0_ROCMFPX` | signed magnitude up to `31` | 2 scales, 1 per 16 weights |
| `Q8_0_ROCMFPX` | signed int8 clamped to `[-127, 127]` | 1 scale per 32 weights |

## Agentic Presets

Use the agent presets when the model needs to behave well in tool-driven or
JSON-heavy workloads:

- `Q3_0_ROCMFPX_AGENT`
- `Q6_0_ROCMFPX_AGENT`
- `Q8_0_ROCMFPX_AGENT`
- `Q4_0_ROCMFP4_COHERENT` for ROCmFP4 agent-oriented builds

These presets do not change the tensor layout. They spend extra precision on the
parts that matter most for agents:

- token and output embeddings
- attention Q/K/V/O
- selected FFN-down tensors
- selective FFN-gate tensors
- bulk FFN-up tensors

The practical rule is simple: keep straight quants for size/speed sweeps, use
agent presets when the workload depends on structured output or external tool
calls.

## K/V Cache Rule

The ROCmFPX model family is separate from cache quantization, but cache types
still matter at runtime.

- `Q3_0_ROCMFPX` is usable for V cache.
- `Q3_0_ROCMFPX` is not acceptable for K cache in the current build.
- `common/common.cpp` promotes `-ctk q3_0_rocmfpx` to `q6_0_rocmfpx` and logs a
  warning.

This is a coherency safeguard, not a new compression scheme. It keeps fp3 K
cache above the observed tool-call / agent floor.

## How To Build

Use the repo’s existing build scripts for the relevant GPU family:

| Target | Script |
|---|---|
| Strix Halo / RDNA3.5 (`gfx1151`) | `scripts/build-strix-rocmfp4-mtp.sh` |
| RDNA2 (`gfx1030` class) | `scripts/build-rdna2.sh` |
| RDNA3 (`gfx1100` class) | `scripts/build-rdna3.sh` |
| RDNA4 (`gfx1200` class) | `scripts/build-rdna4.sh` |

The usual local build pattern is:

```bash
cmake --build build-strix-rocmfp4 --target llama-quantize llama-cli llama-server llama-bench test-backend-ops -j 8
```

For Strix Halo runs, the working runtime pattern is:

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
GGML_HIP_ENABLE_UNIFIED_MEMORY=1 \
./build-strix-rocmfp4/bin/llama-cli -m /path/to/model.gguf -dev ROCm0 -ngl 999
```

## How To Quant

Use BF16 or F16 source GGUFs for real quantization. Re-quantizing an already
heavy quant is useful for smoke tests, not for best quality.

The main wrapper is:

```bash
scripts/quantize-rocmfpx-agent.sh
```

Family quants:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q3_0_ROCMFPX.gguf \
  FORMAT=rocmfp3 PROFILE=straight scripts/quantize-rocmfpx-agent.sh

SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q6_0_ROCMFPX.gguf \
  FORMAT=rocmfp6 PROFILE=straight scripts/quantize-rocmfpx-agent.sh

SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q8_0_ROCMFPX.gguf \
  FORMAT=rocmfp8 PROFILE=straight scripts/quantize-rocmfpx-agent.sh
```

Agent quants:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q3_0_ROCMFPX_AGENT.gguf \
  FORMAT=rocmfp3 PROFILE=agent scripts/quantize-rocmfpx-agent.sh

SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q6_0_ROCMFPX_AGENT.gguf \
  FORMAT=rocmfp6 PROFILE=agent scripts/quantize-rocmfpx-agent.sh

SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q8_0_ROCMFPX_AGENT.gguf \
  FORMAT=rocmfp8 PROFILE=agent scripts/quantize-rocmfpx-agent.sh
```

The wrapper maps to these presets:

- straight FP3 -> `Q3_0_ROCMFPX`
- agent FP3 -> `Q3_0_ROCMFPX_AGENT`
- straight FP4 -> `Q4_0_ROCMFP4`
- agent FP4 -> `Q4_0_ROCMFP4_COHERENT`
- straight FP6 -> `Q6_0_ROCMFPX`
- agent FP6 -> `Q6_0_ROCMFPX_AGENT`
- straight FP8 -> `Q8_0_ROCMFPX`
- agent FP8 -> `Q8_0_ROCMFPX_AGENT`

For ROCmFP4 agent use, the matching form is:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q4_0_ROCMFP4_COHERENT_AGENT.gguf \
  FORMAT=rocmfp4 PROFILE=agent scripts/quantize-rocmfpx-agent.sh
```

Agent-specific dequant/routing policy:

- keep token and output embeddings protected
- keep attention Q/K/V/O at higher precision than the bulk model
- keep selected FFN-down tensors higher precision
- keep selective FFN-gate tensors higher precision
- leave bulk FFN-up tensors on the family format

In practice, the agent presets are still ROCmFPX quants, but the tensor mix is
tuned for structured output and tool use rather than minimum size. They do not
introduce a new dequant kernel; they use the same ROCmFPX math and a more
protected tensor selection.

## How To Test

Use the dedicated smoke harness:

```bash
scripts/check-rocmfpx-agentic-smoke.sh
```

It validates:

- chat coherency
- coding output
- strict JSON output
- tool-call JSON shape
- three-bullet coherency summary
- streaming protocol

The script now refuses to start if ROCm already reports an active KFD process,
and it idles briefly before loading the next model. That was added to make
“clear VRAM before each test” a concrete rule instead of a memory-based one.

## Observed Local Results

These are the current local reference points from the Strix Halo workspace:

- ROCmFP8 agent quant from BF16: `31,568.94 MiB`, `8.39 BPW`
- ROCmFP4 agent quant from BF16: `17,136.79 MiB`, `4.55 BPW`
- ROCmFP4 agent bench: `pp512 650.63 t/s`, `tg128 76.55 t/s`
- ROCmFP4 agent smoke: pass
- ROCmFP8 agent smoke: pass
- BF16 baseline smoke: pass

## Review Notes

If you are extending the family, check these first:

1. `docs/ROCmFPX-EXPERIMENT.md`
2. `ggml/rocmfpx/README.md`
3. `scripts/quantize-rocmfpx-agent.sh`
4. `scripts/check-rocmfpx-agentic-smoke.sh`
5. `src/llama-kv-cache.cpp`
6. `common/common.cpp`

The point of the branch is to keep the ROCmFPX family straightforward:
straight quants when the user wants size and speed, agent presets when they want
better structured output and tool behavior, without making either path hard to
use.
