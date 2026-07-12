# ROCmFPX for llama.cpp

ROCmFPX adds experimental AMD-focused 3-, 4-, 6-, and 8-bit GGUF model-weight
formats to `llama.cpp`, with CPU reference paths and accelerated HIP/ROCm and
Vulkan kernels.

> **Status:** ROCmFPX is an experimental feature family on the canonical `main`
> branch. APIs, tuning choices, and performance can change. Results depend on
> hardware, drivers, model, prompt, and quantization recipe; use BF16/F16 sources
> for quality comparisons.

## Why ROCmFPX?

- **AMD-first weight formats:** ROCmFP3, ROCmFP4, ROCmFP6, and ROCmFP8 are real
  GGUF model-weight quants, not just K/V-cache compression.
- **Native accelerated paths:** HIP/ROCm and Vulkan kernels are backed by CPU
  reference implementations for correctness testing.
- **Speed and size choices:** ROCmFP4 is the speed-first 4-bit family; existing
  Qwen comparisons put its files about 12% below the matched Q4_K_M size.
- **Agent-aware presets:** coherent/agent recipes protect tensors that matter for
  code, JSON, tool calling, and structured output.
- **Built-in MTP acceleration:** models with an MTP/NextN head—including
  M-RoPE Qwen models—can use target-verified self-speculative decoding without
  loading a separate draft model.
- **Broad validation:** the promoted source was exercised through local
  CPU/Vulkan/ROCm tests and cross-platform CI covering Windows, macOS/Metal,
  WebUI provisioning, and Apple packaging.

Start with [Quick Start](#quick-start-strix-halo--gfx1151), choose a format in
[Which Format Should I Pick?](#which-format-should-i-pick), or jump directly to
[MTP Speculative Decoding](#faster-decode-mtp-speculative-decoding).

## Verified MTP Results — Strix Halo, 2026-07-12

These are local command-line decode results from the promoted `main` source on
Strix Halo (`gfx1151`). Throughput is the final `Generation:` rate from
`llama-cli`; runs used full GPU offload, FlashAttention, `-c 4096`, greedy
sampling (`--temp 0`), `-b 512 -ub 512`, the same prompt within each row, and
one model at a time.

| Model and backend | Tokens | MTP profile | No MTP | MTP result | Speedup |
|---|---:|---|---:|---:|---:|
| Qwable-5-27B-Coder ROCmFP4 COHERENT_AGENT, Vulkan0 | 64 | `n6 / p0.60` | 14.0 t/s | 33.2 t/s | 2.37x |
| Qwen3.6-35B-A3B ROCmFP4 STRIX_LEAN-FRESH, Vulkan0 | 256 | `n4 / p0.55` | 76.5 t/s | 116.1 t/s median, 118.3 peak | 1.52x median |
| Qwen3.6-35B-A3B ROCmFP4 STRIX_LEAN-FRESH, ROCm0 | 256 | `n4 / p0.55` | — | 106.2 t/s median | — |

Qwable is a matched single 64-token pair. Qwen no-MTP is one 256-token run;
the MTP values are the median and peak of three matched 256-token runs.

The promoted source and the pre-promotion experimental build were effectively
tied on Qwen3.6: median differences were `-0.7%` on Vulkan and `-0.3%` on ROCm.
On a longer 512-token Vulkan run, the promoted source reached `110.7 t/s` versus
`107.2 t/s` for the experimental build. Qwable's 256-token branch comparison
was also tied: promoted/experimental measured `32.9/33.0 t/s` on Vulkan and
`32.3/32.3 t/s` on ROCm.

MTP gains are content-dependent: predictable code, JSON, and lists usually
accept more draft tokens than creative prose. Treat the profiles above as tested
starting points, not universal defaults.

## Quick Start (Strix Halo / `gfx1151`)

Four commands from clone to a running model. For other AMD GPUs, swap the build
script using the [Clone And Build](#clone-and-build) table.

```bash
# 1. Get the code (canonical main branch)
git clone https://github.com/charlie12345/ROCmFPX.git
cd ROCmFPX && git checkout main

# 2. Build for Strix Halo
env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh          # -> build-strix-rocmfp4/

# 3. Quantize a BF16/F16 GGUF to ROCmFP4 (4.25 bpw, speed-first layout)
build-strix-rocmfp4/bin/llama-quantize model-BF16.gguf model-ROCMFP4_FAST.gguf Q4_0_ROCMFP4_FAST

# 4. Run it with the fastest backend measured on this Strix Halo system
build-strix-rocmfp4/bin/llama-cli \
  -m model-ROCMFP4_FAST.gguf -dev Vulkan0 -ngl 999 -fa on --jinja
```

That is the whole loop: **build → quantize → run.** The sections below explain
each format, how to convert an existing NVFP4 model, and how to squeeze more
decode speed with speculative decoding.

For a model that contains an MTP/NextN head, add a tested starting profile:

```bash
build-strix-rocmfp4/bin/llama-cli \
  -m model-with-MTP.gguf -dev Vulkan0 -ngl 999 -fa on --jinja \
  --temp 0 --spec-type draft-mtp \
  --spec-draft-n-max 6 --spec-draft-p-min 0.6
```

For Qwen3.6-35B-A3B on the tested Strix Halo system, `n4 / p0.55` was faster:

```text
--spec-draft-n-max 4 --spec-draft-p-min 0.55
```

To use HIP/ROCm instead of Vulkan:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export GGML_HIP_ENABLE_UNIFIED_MEMORY=1
build-strix-rocmfp4/bin/llama-cli \
  -m model-ROCMFP4_FAST.gguf -dev ROCm0 -ngl 999 -fa on --jinja
```

## Tested Support

| Target | Status |
|---|---|
| Strix Halo / RDNA3.5 (`gfx1151`) | locally built and benchmarked with Vulkan and HIP/ROCm; Vulkan was fastest for tested decode workloads |
| RDNA2 (`gfx1030`), RDNA3 (`gfx1100`), RDNA4 (`gfx1200`) | dedicated build scripts are provided; results vary by GPU and ROCm version |
| CPU | reference and correctness paths; not the recommended performance backend |
| Vulkan | accelerated and the recommended decode starting point on tested Strix Halo hardware |
| HIP/ROCm | accelerated and validated on the tested Strix Halo system |

## Which Format Should I Pick?

| Goal | Use | Why |
|---|---|---|
| **Smallest + speed-first decode** | `Q4_0_ROCMFP4_FAST` | 4.25 bpw, single scale/block — the speed-oriented default |
| **Balanced 4-bit** | `Q4_0_ROCMFP4` | 4.50 bpw, dual per-16 scale — a touch more precision |
| **Agents / tools / JSON / code** | `Q4_0_ROCMFP4_COHERENT` (or any `*_AGENT`) | protects the tensors that keep structured output correct |
| **Strix Halo tuned recipe** | `Q4_0_ROCMFP4_STRIX_LEAN` | attn-K/V quality recipe tuned on `gfx1151` |
| **Higher quality reference** | `Q6_0_ROCMFPX` / `Q8_0_ROCMFPX` | 6.5 / 8.25 bpw ROCmFPX references |
| **Smallest experimental** | `Q3_0_ROCMFPX` | 3.5 bpw — smallest, most lossy; test coherency first |

Rule of thumb: start with **`Q4_0_ROCMFP4_FAST`** for speed, or a **`*_COHERENT` /
`*_AGENT`** preset if the model does tool-calling, JSON, or coding. Always compare
against your BF16/F16 source for real quality checks.

## What Is ROCmFPX?

ROCmFPX is a family of GGUF model-weight quants:

| Family name | GGUF preset | Role |
|---|---|---|
| ROCmFP3 | `Q3_0_ROCMFPX` | smallest experimental ROCmFPX weight format |
| ROCmFP4 | `Q4_0_ROCMFP4`, `Q4_0_ROCMFP4_FAST` | promoted 4-bit ROCm family baseline |
| ROCmFP6 | `Q6_0_ROCMFPX` | middle quality/size ROCmFPX weight format |
| ROCmFP8 | `Q8_0_ROCMFPX` | high-quality ROCmFPX reference format |

Agent-specific versions are also available:

| Family name | Agent preset | Role |
|---|---|---|
| ROCmFP3 Agent | `Q3_0_ROCMFPX_AGENT` | low-bit ROCmFPX with protected agent tensors |
| ROCmFP6 Agent | `Q6_0_ROCMFPX_AGENT` | middle ROCmFPX with protected agent tensors |
| ROCmFP8 Agent | `Q8_0_ROCMFPX_AGENT` | high-quality ROCmFPX with protected agent tensors |
| ROCmFP4 Agent | `Q4_0_ROCMFP4_COHERENT` | ROCmFP4 coherent agent-oriented preset |

ROCmFPX is not a K/V-cache-only compression trick. It is a set of actual GGUF
model-weight tensor formats with CPU reference paths plus ROCm/HIP and Vulkan
kernel coverage.

## Contributors And Credit

This work builds on `llama.cpp`; upstream authors and contributors retain credit
under the MIT license. See `AUTHORS`, `LICENSE`, and `THIRD_PARTY_NOTICES.md`.

ROCmFP4 and ROCmFPX experiment work in this repository is maintained by
`charlie12345` / `caf`.

Additional ROCmFPX contributors:

- `ciru-ai`: ROCmFPX FP3 Vulkan matvec/dequant speed path.
- Tom Turney / `PlunderStruck` / Aydan S.: TurboQuant `turbo3`/`turbo4`
  K/V-cache quantization paths for ROCm/HIP and Vulkan.

## Why It Is Different From Regular Quants

Most regular GGUF quants target broad size/quality tradeoffs. ROCmFPX is
AMD-oriented and keeps the ROCmFP4 discipline:

- 32-weight blocks for CPU, HIP, and Vulkan kernel compatibility
- finite unsigned UE4M3 scale bytes
- explicit integer-code-times-decoded-scale dequant math
- reconstruction-MSE scale selection where low-bit coherency needs it
- tensor-aware routing for low-bit coherency instead of applying one blunt type
  everywhere
- optional agent presets for JSON, tool calling, coding, and chat coherency

The agent presets do not invent a separate dequant kernel. They use the same
ROCmFPX math but protect the tensors that tend to break structured output:
token/output embeddings, attention Q/K/V/O, selected FFN-down, and selected
FFN-gate tensors.

## Detailed and Historical Benchmarks

These are additional pre-promotion comparisons from a Strix Halo / `gfx1151`
system. Treat them as historical local data, not a universal benchmark. All
rows within each table used the same model pair, backend, batch shape, K/V
cache, FlashAttention setting, and one test at a time.

### Qwen3.6 27B, Vanilla No-MTP

Model pair:

- Baseline: `Qwen3.6-27B-Q4_K_M.gguf`, `16.55 GB`
- ROCmFPX: `Qwen3.6-27B-VANILLA-NO-MTP-BF16-to-ROCmFP4-STRIX_LEAN.gguf`, `14.59 GB`
- Size delta: ROCmFP4 is `11.82%` smaller
- Test: `llama-bench`, `pp512 + tg128`, MTP/speculative decoding disabled

| Backend | Quant | Prompt fill tok/s | Decode tg128 tok/s |
|---|---|---:|---:|
| ROCm0 | Q4_K_M | 336.97 | 11.74 |
| ROCm0 | ROCmFP4 STRIX_LEAN | 328.03 | 13.53 |
| Vulkan0 | Q4_K_M | 352.04 | 12.89 |
| Vulkan0 | ROCmFP4 STRIX_LEAN | 376.98 | 14.27 |

On this 27B vanilla run, ROCmFP4 was slightly behind Q4_K_M for ROCm prompt
fill, but faster for decode on both ROCm and Vulkan. Vulkan ROCmFP4 also led
prompt fill.

### Qwen3.6 35B A3B Weight-Quant Comparison

Model pair:

- Baseline: `Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf`, `21.71 GB`
- ROCmFPX: `Qwen3.6-35B-A3B-MTP-BF16-to-ROCmFP4-STRIX_LEAN-ROCmFPXCLONE.gguf`, `19.05 GB`
- Size delta: ROCmFP4 is `12.28%` smaller
- Test: `llama-bench`, `pp512 + tg128`; this table measures the weight quants,
  not speculative MTP acceleration

| Backend | Quant | Prompt fill tok/s | Decode tg128 tok/s |
|---|---|---:|---:|
| ROCm0 | Q4_K_M | 1353.50 | 59.00 |
| ROCm0 | ROCmFP4 STRIX_LEAN | 1301.21 | 66.42 |
| Vulkan0 | Q4_K_M | 1065.83 | 70.57 |
| Vulkan0 | ROCmFP4 STRIX_LEAN | 1200.81 | 76.71 |

The same 35B A3B pair was also run through a 20-prompt Hermes-style agent
smoke:

| Backend | Quant | Prompt tok/s | Generation tok/s |
|---|---|---:|---:|
| ROCm0 | Q4_K_M | 699.7 | 31.9 |
| ROCm0 | ROCmFP4 STRIX_LEAN | 731.4 | 47.1 |
| Vulkan0 | Q4_K_M | 654.0 | 40.2 |
| Vulkan0 | ROCmFP4 STRIX_LEAN | 730.9 | 57.5 |

On this 35B A3B comparison, ROCmFP4 was smaller and faster on decode/generation
across ROCm and Vulkan. ROCm prompt fill was still slightly behind Q4_K_M in
`llama-bench`, while Vulkan prompt fill and Hermes-style prompts favored
ROCmFP4.

## Clone And Build

```bash
git clone https://github.com/charlie12345/ROCmFPX.git
cd ROCmFPX
```

Most users should stay on `main`. The preserved
`experimental-rocmfpx-branch` exists for history and rollback comparisons.

Pick the build script for your machine:

| Hardware | Build command | Output folder |
|---|---|---|
| Strix Halo / RDNA3.5 (`gfx1151`) | `env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh` | `build-strix-rocmfp4/` |
| RDNA2 / RX 6000 (`gfx1030` class) | `env JOBS=16 scripts/build-rdna2.sh` | `build-rdna2/` |
| RDNA3 / RX 7000 (`gfx1100` class) | `env JOBS=16 scripts/build-rdna3.sh` | `build-rdna3/` |
| RDNA4 / RX 9000 (`gfx1200` class) | `env JOBS=16 scripts/build-rdna4.sh` | `build-rdna4/` |
| RDNA4 / RX 9000 — self-contained (no system ROCm) | `env JOBS=16 scripts/build-rocmfp4-rocm714-local.sh` | `build-rdna4-rocm714/` |
| Vulkan-only / manual | use the Vulkan CMake path in `docs/BUILD-AMD-ARCHITECTURES.md` | custom |

For Strix Halo, the common runtime environment is:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export GGML_HIP_ENABLE_UNIFIED_MEMORY=1
```

Key binaries after build:

```text
build-strix-rocmfp4/bin/llama-quantize
build-strix-rocmfp4/bin/llama-cli
build-strix-rocmfp4/bin/llama-server
build-strix-rocmfp4/bin/llama-bench
build-strix-rocmfp4/bin/test-backend-ops
```

For RDNA2/RDNA3/RDNA4 builds, use the same binary names under that build
folder, for example `build-rdna3/bin/llama-quantize`.

The `build-rocmfp4-rocm714-local.sh` script (RDNA4 / RX 9000) downloads the
ROCm 7.14.0a20260624 toolchain automatically and bundles the required
runtime libraries alongside the binaries. The resulting build is
self-contained — no system-wide ROCm install or `LD_LIBRARY_PATH` needed
at runtime.

## Quantize Straight ROCmFPX Models

Use BF16 or F16 GGUF sources. The wrapper keeps split GGUFs split by default.

ROCmFP3:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q3_0_ROCMFPX.gguf \
  FORMAT=rocmfp3 PROFILE=straight scripts/quantize-rocmfpx-agent.sh
```

ROCmFP4:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q4_0_ROCMFP4.gguf \
  FORMAT=rocmfp4 PROFILE=straight scripts/quantize-rocmfpx-agent.sh
```

ROCmFP6:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q6_0_ROCMFPX.gguf \
  FORMAT=rocmfp6 PROFILE=straight scripts/quantize-rocmfpx-agent.sh
```

ROCmFP8:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q8_0_ROCMFPX.gguf \
  FORMAT=rocmfp8 PROFILE=straight scripts/quantize-rocmfpx-agent.sh
```

You can also call `llama-quantize` directly:

```bash
build-strix-rocmfp4/bin/llama-quantize source.gguf out-q3.gguf Q3_0_ROCMFPX
build-strix-rocmfp4/bin/llama-quantize source.gguf out-q4.gguf Q4_0_ROCMFP4
build-strix-rocmfp4/bin/llama-quantize source.gguf out-q6.gguf Q6_0_ROCMFPX
build-strix-rocmfp4/bin/llama-quantize source.gguf out-q8.gguf Q8_0_ROCMFPX
```

For low-bit ROCmFPX quants, pass an imatrix when you have one:

```bash
IMATRIX=/path/to/imatrix.gguf \
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q3_0_ROCMFPX.gguf \
  FORMAT=rocmfp3 PROFILE=straight scripts/quantize-rocmfpx-agent.sh
```

The wrapper forwards `IMATRIX` to `llama-quantize --imatrix`. ROCmFP3,
ROCmFP6, and ROCmFP8 use imatrix-weighted scale search; ROCmFP4 has its own
imatrix path.

## Quantize Agent ROCmFPX Models

Use agent mode when the model will be used for Hermes/OpenClaw-style workflows,
tool calling, JSON output, coding, or chat agents.

ROCmFP3 Agent:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q3_0_ROCMFPX_AGENT.gguf \
  FORMAT=rocmfp3 PROFILE=agent scripts/quantize-rocmfpx-agent.sh
```

ROCmFP6 Agent:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q6_0_ROCMFPX_AGENT.gguf \
  FORMAT=rocmfp6 PROFILE=agent scripts/quantize-rocmfpx-agent.sh
```

ROCmFP8 Agent:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q8_0_ROCMFPX_AGENT.gguf \
  FORMAT=rocmfp8 PROFILE=agent scripts/quantize-rocmfpx-agent.sh
```

ROCmFP4 Agent:

```bash
SRC=/path/to/model-BF16.gguf OUT=/path/to/model-Q4_0_ROCMFP4_COHERENT_AGENT.gguf \
  FORMAT=rocmfp4 PROFILE=agent scripts/quantize-rocmfpx-agent.sh
```

The wrapper maps `FORMAT` and `PROFILE` like this:

| FORMAT | PROFILE | Preset |
|---|---|---|
| `rocmfp3` | `straight` | `Q3_0_ROCMFPX` |
| `rocmfp3` | `agent` | `Q3_0_ROCMFPX_AGENT` |
| `rocmfp4` | `straight` | `Q4_0_ROCMFP4` |
| `rocmfp4` | `agent` | `Q4_0_ROCMFP4_COHERENT` |
| `rocmfp6` | `straight` | `Q6_0_ROCMFPX` |
| `rocmfp6` | `agent` | `Q6_0_ROCMFPX_AGENT` |
| `rocmfp8` | `straight` | `Q8_0_ROCMFPX` |
| `rocmfp8` | `agent` | `Q8_0_ROCMFPX_AGENT` |

## Convert An Existing NVFP4 Model To ROCmFP4

If you already have an **NVFP4** GGUF, you can re-map it onto the ROCmFP4 kernel
path without re-quantizing from BF16. This is the closest-matching conversion
ROCmFPX supports: NVFP4 and ROCmFP4 use the **same UE4M3 scale** and share **7 of
8 codebook levels** — only the top magnitude level differs (NVFP4 `12` vs ROCmFP4
`10`), so almost every weight maps over cleanly.

```bash
# Same 4.50 bpw as NVFP4 (closest quality match):
build-strix-rocmfp4/bin/llama-quantize --allow-requantize \
  model-NVFP4.gguf model-ROCMFP4.gguf Q4_0_ROCMFP4

# Smaller 4.25 bpw (speed-first layout, a little more loss):
build-strix-rocmfp4/bin/llama-quantize --allow-requantize \
  model-NVFP4.gguf model-ROCMFP4_FAST.gguf Q4_0_ROCMFP4_FAST
```

- `--allow-requantize` is **required**: NVFP4 GGUFs usually keep `output.weight` at
  a higher-precision type (e.g. `q6_K`), so the file has mixed source types.
- Example measured on a 9B NVFP4 model (wikitext-2, `gfx1151`): the 4.50 bpw target
  landed within noise of the NVFP4 source perplexity; the 4.25 bpw `FAST` target was
  ~5% higher perplexity for a ~10% smaller file. Numbers are model-dependent — always
  A/B against the NVFP4 source on your own prompts.
- To make *every* tensor ROCmFP4 (a uniform "even" file), use the `Q4_0_ROCMFP4_EVEN`
  / `Q4_0_ROCMFP4_FAST_EVEN` presets, which imply `--pure`.

## Faster Decode: MTP Speculative Decoding

If your model ships with an **MTP / NextN** draft head (many recent models do),
you can turn on self-speculative decoding for a real decode speedup — no separate
draft model needed. This is the most effective way to push decode throughput past
what the weight format alone can do, because accepted draft tokens produce several
tokens per weight read.

MTP helps **both dense and MoE** models here. On `gfx1151`, `-dev Vulkan0`
was the fastest backend in the validated Qwen3.6 and Qwable comparisons.

```bash
# General starting profile for a model with an embedded MTP/NextN head
build-strix-rocmfp4/bin/llama-cli \
  -m model-with-MTP.gguf -dev Vulkan0 -ngl 999 -fa on --jinja \
  --temp 0 \
  --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.6
```

- **Tune per model and workload.** `n6 / p0.60` is a useful starting point, but
  the validated Qwen3.6-35B-A3B profile was faster at `n4 / p0.55`. Very low
  `p_min` can waste work on rejected drafts; an overly high value can miss useful
  draft tokens.
- The speedup is **content-dependent**: structured / predictable output (code, lists,
  JSON) accepts more drafts and gains most; free-form creative text gains less.
- It is **lossless**: at greedy (`--temp 0`) the output matches non-speculative
  decoding token-for-token (the target model verifies every drafted token).
- See [Verified MTP Results](#verified-mtp-results--strix-halo-2026-07-12) for
  current Qwable and Qwen3.6 measurements, profiles, and branch-parity context.

**M-RoPE models (`qwen35` / `qwen35moe`, and any IMROPE/MROPE arch):** MTP now
works on these. They use 4-D M-RoPE positions, and the batch position check
previously rejected the MTP draft/verify batch every step (`for M-RoPE, it is
required that the position satisfies: X < Y`), so MTP silently fell back to
plain decode. The MTP hook batch is a hybrid (token id **plus** an injected
hidden-state row) and is allowed to reuse positions like an embedding batch, so
the strict check is now gated on `batch.token && !batch.embd`
(`src/llama-batch.cpp`). If you are on an older build and see that `X < Y` error
spamming during MTP, this is the fix. NEOX-RoPE MTP (e.g. Gemma4 assistants) was
never affected.

## What The Agent Preset Protects

The agent profile is a tensor-routing choice. It keeps the ROCmFPX block
formats but spends more bits on tensors that affect structured behavior:

- token and output embeddings
- attention Q/K/V/O tensors
- selected FFN-down tensors
- selective FFN-gate tensors
- bulk FFN-up tensors stay on the family quant where possible

This is why agent quants are slightly larger than straight quants. The goal is
to preserve JSON shape, tool-call shape, coding behavior, and chat coherency
without forcing the whole model to a generic high-bit quant.

## Run A Quantized Model

Simple ROCm run:

```bash
build-strix-rocmfp4/bin/llama-cli \
  -m /path/to/model-Q8_0_ROCMFPX_AGENT.gguf \
  -dev ROCm0 \
  -ngl 999 \
  -fa on \
  -c 8192 \
  -b 512 \
  -ub 512 \
  --jinja
```

OpenAI-compatible server:

```bash
build-strix-rocmfp4/bin/llama-server \
  -m /path/to/model-Q8_0_ROCMFPX_AGENT.gguf \
  --host 127.0.0.1 \
  --port 8138 \
  -dev ROCm0 \
  -ngl 999 \
  -fa on \
  -c 8192 \
  -b 512 \
  -ub 512 \
  --jinja \
  --reasoning off
```

## K/V Cache Rule

ROCmFPX model quants and K/V cache types are separate runtime controls.

The current guard promotes `-ctk q3_0_rocmfpx` to `q6_0_rocmfpx` because fp3 K
cache was below the observed tool-call and agent coherency floor. `q3_0_rocmfpx`
can still be used for V cache.

TurboQuant K/V cache support is already built into this tree as the `turbo3`
and `turbo4` runtime cache types, including CPU reference tests plus ROCm/HIP
and Vulkan paths. TurboQuant is not a ROCmFPX model-weight quant; use it with
`-ctk` and `-ctv` at runtime.

The recommended safe TurboQuant+ style policy is asymmetric K/V:

```bash
build-strix-rocmfp4/bin/llama-server \
  -m /path/to/model-Q6_0_ROCMFPX_AGENT.gguf \
  -dev Vulkan0 \
  -ngl 999 \
  -fa on \
  -ctk q8_0 \
  -ctv turbo4 \
  --jinja
```

For the ROCmFPX MTP server wrapper, use the preset script:

```bash
MODEL=/path/to/model-Q6_0_ROCMFPX_AGENT.gguf \
DEVICE=Vulkan0 \
scripts/run-rocmfpx-turboquant-asym-server.sh
```

This keeps K cache at `q8_0`, where attention quality and tool calls are more
sensitive, and uses `turbo4` for V cache, where compression is usually cheaper.
You can still run symmetric TurboQuant for sweeps with `-ctk turbo3 -ctv turbo3`
or `-ctk turbo4 -ctv turbo4`, but do not treat those as the default agentic
serving profile.

For symmetric TurboQuant experiments, first/last-layer K protection is available
as an opt-in compatibility knob:

```bash
LLAMA_KV_TURBO_BOUNDARY_LAYERS=2 \
build-strix-rocmfp4/bin/llama-server \
  -m /path/to/model.gguf \
  -ctk turbo4 \
  -ctv turbo4
```

With that flag, the first and last two model layers use `q8_0` for K cache
while the middle layers use the requested TurboQuant type. V boundary protection
is off by default; enable it only for experiments with
`LLAMA_KV_TURBO_BOUNDARY_V=1`.

Do not import the Python `turboquant_plus` research package into this C/C++ tree
as-is. The low-risk production findings are the asymmetric K/V policy and
documentation. QJL and turbo2 are intentionally not enabled here, and block-size
128 would require a GGML block-layout change and compatibility work.

## Test Agent Behavior

The agentic smoke harness checks chat, coding, JSON, tool-call JSON, coherency,
and streaming. It also refuses to start when ROCm reports an active KFD process,
so each run starts after VRAM/process cleanup.

```bash
MODEL=/path/to/model-Q8_0_ROCMFPX_AGENT.gguf \
BACKEND=ROCm0 \
ALIAS=rocmfpx-agent \
OUT_DIR=/tmp/rocmfpx-agentic-smoke \
scripts/check-rocmfpx-agentic-smoke.sh
```

## Code Layout

- `ggml/rocmfpx/` - ROCmFP3/ROCmFP6/ROCmFP8 reference formats
- `ggml/rocmfp4/` - ROCmFP4 reference path this family inherits from
- `scripts/quantize-rocmfpx-agent.sh` - simple straight-vs-agent quant wrapper
- `scripts/check-rocmfpx-agentic-smoke.sh` - OpenAI-compatible agent smoke test
- `docs/ROCmFPX-HANDOFF.md` - detailed handoff for reviewers and other agents
- `docs/ROCmFPX-EXPERIMENT.md` - experiment history, routing notes, and gates
- `docs/BUILD-AMD-ARCHITECTURES.md` - RDNA2/RDNA3/RDNA4/Strix build details

## License

This repository is based on `llama.cpp` and keeps the upstream MIT license. See
`LICENSE` for details. Bundled third-party notices are listed in
`THIRD_PARTY_NOTICES.md`.
