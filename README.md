# ROCmFP4 for llama.cpp

Experimental AMD-focused FP4 quantization and backend work for `llama.cpp`,
developed on Framework Desktop with AMD Strix Halo 395+ and 128 GB unified RAM.

ROCmFP4 adds new GGUF tensor formats, quantization presets, ROCm/HIP kernels,
Vulkan shader support, and reproducible regression guards for long-context MTP
inference. The goal is a practical 4-bit format for AMD systems that keeps model
coherence protected while improving memory use and decode speed.

> Status: experimental research build. Results are hardware-, driver-, model-,
> and prompt-sensitive. Do not treat these numbers as upstream llama.cpp claims
> until they are independently reproduced.

## What Is ROCmFP4?

ROCmFP4 is a custom 4-bit weight format for GGUF models:

- `Q4_0_ROCMFP4`: dual-scale 4.50 BPW layout using two finite UE4M3 scale bytes
  per 32-weight block.
- `Q4_0_ROCMFP4_FAST`: single-scale 4.25 BPW layout for speed-sensitive tensor
  roles.
- Tensor-aware presets that mix ROCmFP4 layouts with protected higher-precision
  tensors where quality matters.
- ROCm/HIP vector-dot, copy, dequant, and FlashAttention handling for the new
  layouts.
- Vulkan decode and MMQ shader support for the same GGUF tensor types.
- MTP regression guards for long-context target/draft decode.

ROCmFP4 is not MXFP4, NVFP4, or a renamed Q4 format. It uses a Codebook10 4-bit
value table and finite unsigned E4M3 half-scale semantics tuned for the current
AMD backend paths in this tree.

## Proven Local Results

Current strongest reproduced local result:

| Hardware | Model | Backend | Context | Profile | Decode |
|---|---|---:|---:|---|---:|
| Framework AMD Strix Halo 395+, 128 GB unified RAM | Qwen3.6 35B A3B MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | reasoning on, draft-MTP, q8 main KV, q4 draft KV | 104.4 tok/s short, 89.3 tok/s sustained |
| Framework AMD Strix Halo 395+, 128 GB unified RAM | Qwen3.6 27B MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | draft-MTP | 33.6 tok/s short, 28.0 tok/s sustained |

The benchmark policy is intentionally conservative: a microbenchmark win is not
promoted unless end-to-end decode guards hold or improve.

## Validation Matrix

| Target | Script | Status |
|---|---|---|
| Strix Halo / RDNA3.5 (`gfx1151`) | `scripts/build-strix-rocmfp4-mtp.sh` | Validated locally on Framework Desktop / Ryzen AI MAX+ 395 |
| RDNA3 (`gfx1100` class) | `scripts/build-rdna3.sh` | Build target provided; community validation wanted |
| RDNA2 (`gfx1030` class) | `scripts/build-rdna2.sh` | Build target provided; community validation wanted |
| RDNA4 (`gfx1200` class) | `scripts/build-rdna4.sh` | Experimental; requires ROCm support for `gfx1200` device libraries |
| Vulkan fallback | Manual CMake path | Recommended when HIP support is incomplete or a GPU is not mapped cleanly |

## Latest Validated Snapshot

- Validated integration snapshot: `4860505ee`
- Validation date: `2026-06-13`

This snapshot passed the promoted Strix Halo gate with:

- ROCmFP4 quantization tests
- MTMD C API smoke test
- ROCm copy and FlashAttention backend tests
- ROCmFP4 copy, FlashAttention, and runtime regression guards
- Qwen3.6 35B A3B MTP ROCmFP4 smoke test

## Documentation

| Guide | Who it's for |
|---|---|
| [`docs/STRIX-HALO-QUICKSTART.md`](docs/STRIX-HALO-QUICKSTART.md) | Strix Halo users — full install, quantize, run, validate |
| [`docs/BUILD-AMD-ARCHITECTURES.md`](docs/BUILD-AMD-ARCHITECTURES.md) | AMD GPU build flags and scripts, including RDNA2 through RDNA4 plus experimental gfx906 |
| [`docs/ROCmFP4-REPRODUCIBILITY.md`](docs/ROCmFP4-REPRODUCIBILITY.md) | Regression guards and proof commands |
| [`docs/ROCmFP4-MTP-COMPARISON.md`](docs/ROCmFP4-MTP-COMPARISON.md) | Benchmark history and promoted profiles |
| [`docs/ROCmFP4-DECODE-SPEED-EXPERIMENTS.md`](docs/ROCmFP4-DECODE-SPEED-EXPERIMENTS.md) | Opt-in decode tuning profiles, graph timing, and future quantization research notes |
| [`docs/ROCmFPX-EXPERIMENT.md`](docs/ROCmFPX-EXPERIMENT.md) | Experimental ROCmFP3/ROCmFP6/ROCmFP8 format staging |
| [`docs/ROCmFPX-HANDOFF.md`](docs/ROCmFPX-HANDOFF.md) | Operational handoff for ROCmFPX family review, build, quant, and test flow |
| [`ggml/rocmfp4/README.md`](ggml/rocmfp4/README.md) | Format details and expert HIP tuning knobs |

## ROCmFPX Quantization

ROCmFPX is the family for ROCmFP3, ROCmFP6, and ROCmFP8. Use BF16 or F16 as
the source. The straight presets are the default family quants; the agent
presets keep the same family but spend extra precision on agent-sensitive
tensors.

Use the wrapper:

```bash
scripts/quantize-rocmfpx-agent.sh
```

Straight family quants:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q3_0_ROCMFPX.gguf \
  FORMAT=rocmfp3 PROFILE=straight scripts/quantize-rocmfpx-agent.sh

SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q6_0_ROCMFPX.gguf \
  FORMAT=rocmfp6 PROFILE=straight scripts/quantize-rocmfpx-agent.sh

SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q8_0_ROCMFPX.gguf \
  FORMAT=rocmfp8 PROFILE=straight scripts/quantize-rocmfpx-agent.sh
```

Agent family quants:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q3_0_ROCMFPX_AGENT.gguf \
  FORMAT=rocmfp3 PROFILE=agent scripts/quantize-rocmfpx-agent.sh

SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q6_0_ROCMFPX_AGENT.gguf \
  FORMAT=rocmfp6 PROFILE=agent scripts/quantize-rocmfpx-agent.sh

SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q8_0_ROCMFPX_AGENT.gguf \
  FORMAT=rocmfp8 PROFILE=agent scripts/quantize-rocmfpx-agent.sh
```

If you want ROCmFP4 for agent use, the matching call is:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q4_0_ROCMFP4_COHERENT_AGENT.gguf \
  FORMAT=rocmfp4 PROFILE=agent scripts/quantize-rocmfpx-agent.sh
```

Agent-specific dequant/routing policy:

- keep token and output embeddings protected
- keep attention Q/K/V/O at higher precision than the bulk model
- keep selected FFN-down tensors higher precision
- keep selective FFN-gate tensors higher precision
- leave the bulk FFN-up tensors on the family format

That gives you the agent behavior without turning the whole model into a large
generic high-bit quant.

## Repository Layout

- `ggml/rocmfp4/` — ROCmFP4 format definitions, CPU reference quant/dequant, and
  HIP helper kernels
- `ggml/rocmfpx/` — isolated experimental ROCmFP3/ROCmFP6/ROCmFP8 formats
  with CPU reference paths plus ROCm/HIP and Vulkan staging hooks
- `ggml/src/ggml-cuda/` — upstream HIP/CUDA backend files with AMD ROCmFP4
  integration (HIP builds use this directory even with `-DGGML_CUDA=OFF`)
- `ggml/src/ggml-vulkan/vulkan-shaders/` — Vulkan shader support for ROCmFP4
- `scripts/build-*.sh` — build scripts per AMD GPU generation
- `scripts/check-rocmfp4-*.sh` — correctness and performance regression guards

## Build

```bash
git clone https://github.com/charlie12345/rocmfp4-llama.git
cd rocmfp4-llama
git checkout mtp-rocmfp4-strix
```

Pick the script that matches your GPU:

| Your GPU | Build command | Output folder |
|---|---|---|
| Strix Halo / RDNA3.5 | `env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh` | `build-strix-rocmfp4/` |
| RDNA2 (RX 6000) | `env JOBS=16 scripts/build-rdna2.sh` | `build-rdna2/` |
| RDNA3 (RX 7000, including RX 7600-class cards) | `env JOBS=16 scripts/build-rdna3.sh` | `build-rdna3/` |
| RDNA4 (RX 9000) | `env JOBS=16 scripts/build-rdna4.sh` | `build-rdna4/` |
| Vega 20 / gfx906 experimental (MI50 / MI60) | `env JOBS=16 scripts/build-gfx906.sh` | `build-gfx906/` |
| Windows RDNA2 | `build-hip.bat` | `build-hip/` |

Not sure which GPU you have? See
[`docs/BUILD-AMD-ARCHITECTURES.md`](docs/BUILD-AMD-ARCHITECTURES.md) for the full
`gfx` target table, runtime environment variables, and Vulkan-only builds.

Strix Halo users: follow
[`docs/STRIX-HALO-QUICKSTART.md`](docs/STRIX-HALO-QUICKSTART.md) for
prerequisites, MTP flags, and troubleshooting.

Key binaries after a successful build:

```text
bin/llama-cli
bin/llama-server
bin/llama-quantize
bin/llama-bench
bin/test-backend-ops
bin/test-quantize-fns
bin/test-quantize-perf
```

## Quantize a Model

Start from an F16 or BF16 GGUF source model. Quantizing an already heavily
quantized GGUF into ROCmFP4 is useful for smoke tests only; for real quality,
use an F16/BF16 source.

Compact Strix profile:

```bash
./build-strix-rocmfp4/bin/llama-quantize \
  /path/to/source-bf16.gguf \
  /path/to/model-ROCmFP4-STRIX_LEAN.gguf \
  Q4_0_ROCMFP4_STRIX_LEAN
```

Quality-biased Strix profile:

```bash
./build-strix-rocmfp4/bin/llama-quantize \
  /path/to/source-bf16.gguf \
  /path/to/model-ROCmFP4-STRIX.gguf \
  Q4_0_ROCMFP4_STRIX
```

Pure experimental formats:

```bash
./build-strix-rocmfp4/bin/llama-quantize source.gguf out-dual.gguf Q4_0_ROCMFP4
./build-strix-rocmfp4/bin/llama-quantize source.gguf out-fast.gguf Q4_0_ROCMFP4_FAST
```

Use the Strix presets for serious testing. The pure FAST format is smaller and
can be faster, but may trade away too much coherence on sensitive tensors.

## Run a ROCmFP4 Model

Example interactive run:

```bash
cd rocmfp4-llama
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
GGML_HIP_ENABLE_UNIFIED_MEMORY=1 \
./build-strix-rocmfp4/bin/llama-cli \
  -m /path/to/model-ROCmFP4-STRIX_LEAN.gguf \
  -dev ROCm0 \
  -ngl 999 \
  -c 262144 \
  -b 512 \
  -ub 512 \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-n-min 0 \
  --spec-draft-p-min 0.0 \
  --spec-draft-p-split 0.10 \
  --spec-draft-type-k q4_0 \
  --spec-draft-type-v q4_0 \
  --reasoning on \
  --jinja
```

For models that do not support MTP or reasoning, remove the `--spec-*` and
`--reasoning` flags.

## Regression Guards

Run the full promoted gate:

```bash
env HSA_OVERRIDE_GFX_VERSION=11.5.1 scripts/check-rocmfp4-all-regression.sh
```

Focused guards:

```bash
scripts/check-rocmfp4-quant-regression.sh
scripts/check-rocmfp4-rocm-runtime-regression.sh
scripts/check-rocmfp4-rocm-fattn-regression.sh
scripts/check-rocmfp4-vulkan-runtime-regression.sh
scripts/check-rocmfp4-qwen-mtp-regression.sh
scripts/check-rocmfp4-qwen35-a3b-mtp-regression.sh
```

The scripts accept environment overrides for model paths, binary paths, context,
cache type, MTP settings, and speed floors. See each script for the exact
variables.

## Design Principles

- Coherence first: tensor-aware presets are preferred over pure speed profiles.
- AMD-specific work must be isolated and measurable.
- ROCm and Vulkan paths must avoid silent fallback where ROCmFP4-specific
  kernels exist.
- Rejected experiments are recorded so they are not repeatedly rediscovered.
- Claims must cite model, backend, context window, flags, hardware, and date.

## Current Limitations

- This is not native FP4 tensor-core execution. rocWMMA FP4 input support is not
  available in the local headers used by this build.
- ROCmFP4 is currently optimized for AMD Strix Halo behavior and may need
  retuning on other GPUs.
- TurboQuant and TriAttention are not runtime flags in this isolated tree.
- Some scripts reference local model defaults; override `MODEL`, `ROCMFP4_MODEL`,
  or `BASELINE_MODEL` for your checkout.

## License

This repository is based on `llama.cpp` and keeps the upstream MIT license. See
`LICENSE` for details. Bundled third-party notices are listed in
`THIRD_PARTY_NOTICES.md`.
