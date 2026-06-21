# ROCmFPX MTP Serving

This guide covers request-level MTP controls for ROCmFPX serving:

```text
speculative.n_max
speculative.n_min
speculative.p_min
```

These fields let each request lower the active draft policy without restarting
`llama-server`. The server startup value for `--spec-draft-n-max` remains the
allocation cap, so start the server at the highest draft depth you plan to test
and clamp individual requests downward.

## Build

Build `llama-server` from this ROCmFPX tree:

```bash
env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh llama-server llama-bench
```

The default Strix build output is:

```text
build-strix-rocmfp4/bin/llama-server
```

Set `BUILD_DIR` or `BIN` when using a different build directory.

## Start A ROCmFPX MTP Server

The helper script starts a single-slot OpenAI-compatible server with metrics,
MTP enabled, and request-level draft overrides available:

```bash
MODEL=/path/to/model.gguf \
PORT=18180 \
CTX_SIZE=32768 \
DEVICE=Vulkan0 \
SPEC_DRAFT_N_MAX=4 \
SPEC_DRAFT_N_MIN=0 \
SPEC_DRAFT_P_MIN=0.0 \
SPEC_DRAFT_P_SPLIT=0.10 \
scripts/run-rocmfpx-mtp-server.sh
```

Useful defaults:

```text
DEVICE=Vulkan0
BATCH_SIZE=2048
UBATCH_SIZE=512
CACHE_TYPE_K=f16
CACHE_TYPE_V=f16
CACHE_TYPE_K_DRAFT=f16
CACHE_TYPE_V_DRAFT=f16
THREADS=16
THREADS_BATCH=32
STRICT_BENCH=1
```

`STRICT_BENCH=1` disables prompt-cache reuse and sets slot prompt similarity to
zero so benchmark rows are easier to compare. For interactive serving, set
`STRICT_BENCH=0` if you want normal prompt-cache behavior.

## Per-Request Overrides

Use the request keys on `/completion`:

```bash
curl -sS http://127.0.0.1:18180/completion \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Write a concise technical note about ROCmFPX MTP serving.",
    "n_predict": 512,
    "temperature": 0,
    "ignore_eos": true,
    "speculative.n_max": 2,
    "speculative.n_min": 0,
    "speculative.p_min": 0.0
  }'
```

The response `generation_settings` should echo the effective values. If
`speculative.n_max` is higher than the server cap, it is clamped to the cap.
`speculative.n_min` is clamped to `0..n_max`, and `speculative.p_min` is
clamped to `0.0..1.0`.

OpenAI chat-compatible requests use the same keys in the top-level payload:

```bash
curl -sS http://127.0.0.1:18180/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rocmfpx-mtp",
    "messages": [
      {"role": "user", "content": "Summarize the request-level MTP knobs."}
    ],
    "max_tokens": 512,
    "temperature": 0,
    "speculative.n_max": 2,
    "speculative.n_min": 0,
    "speculative.p_min": 0.0
  }'
```

## Dynamic Drafting

Use `scripts/rocmfpx-draft-profile.py` when a client wants to select draft
depth from prompt length before sending a request:

```bash
scripts/rocmfpx-draft-profile.py --profile fp3-mtp --prompt-tokens 4096 --pretty
scripts/rocmfpx-draft-profile.py \
  --profile dense-coder \
  --base-url http://127.0.0.1:18180 \
  --prompt-file /path/to/prompt.txt \
  --pretty
```

The helper emits only request JSON fields. It does not change the server cap,
so the server still needs to start with a high enough `--spec-draft-n-max`.
Available profiles are `fp3-mtp`, `fp4-general`, and `dense-coder`.

## Suggested Starting Points

These are starting points for sweeps, not universal defaults:

| Model path | Startup settings | Request settings |
|---|---|---|
| FP3 MTP speed sweep | `n_max=4`, `p_split=0.10`, target/draft KV `f16` | short context: `n_max=4`, `p_min=0.75`; long context: `n_max=2`, `p_min=0.0` |
| ROCmFP4 MTP general | `n_max=4`, `p_split=0.10`, target/draft KV `f16` | start with `n_max=4`, `p_min=0.0` and sweep `p_min=0.25` |
| ROCmFP4 dense coder | `n_max=6`, `p_split=0.20`, target KV `q8_0`, draft KV `f16` | start with `n_max=6`, `p_min=0.0` |

Example dense-coder server:

```bash
MODEL=/path/to/coder-rocmfp4-agent.gguf \
ALIAS=rocmfpx-coder-mtp \
PORT=18180 \
CTX_SIZE=32768 \
DEVICE=Vulkan0 \
SPEC_DRAFT_DEVICE=Vulkan0 \
BATCH_SIZE=2048 \
UBATCH_SIZE=512 \
CACHE_TYPE_K=q8_0 \
CACHE_TYPE_V=q8_0 \
CACHE_TYPE_K_DRAFT=f16 \
CACHE_TYPE_V_DRAFT=f16 \
SPEC_DRAFT_N_MAX=6 \
SPEC_DRAFT_N_MIN=0 \
SPEC_DRAFT_P_MIN=0.0 \
SPEC_DRAFT_P_SPLIT=0.20 \
STRICT_BENCH=1 \
scripts/run-rocmfpx-mtp-server.sh
```

Use this request policy with that cap:

```json
{
  "speculative.n_max": 6,
  "speculative.n_min": 0,
  "speculative.p_min": 0.0
}
```

## Validation

Before reporting a serving result, record:

- model path and alias
- server binary path and commit
- backend device
- context allocation
- prompt tokens and generated tokens
- target and draft KV cache types
- batch and ubatch
- startup MTP cap and per-request speculative fields
- prompt-cache setting
- decode tok/s, prompt tok/s, TTFP, total time
- draft accepted and draft generated counters

For quick server checks:

```bash
curl -sS http://127.0.0.1:18180/health
curl -sS http://127.0.0.1:18180/props | jq '.default_generation_settings'
curl -sS http://127.0.0.1:18180/metrics | head
```

Use served API rows or a CLI guard with draft counters for headline MTP speed.
Do not use standalone `llama-bench` TG as the headline for MTP serving.
