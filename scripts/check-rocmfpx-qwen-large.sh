#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
QUANTIZE_BIN="${QUANTIZE_BIN:-$BUILD_DIR/bin/llama-quantize}"
MODEL_SRC="${MODEL_SRC:-}"
MODEL_LEAN="${MODEL_LEAN:-}"
MODEL_AGENT="${MODEL_AGENT:-}"
BACKEND="${BACKEND:-ROCm0}"
SKIP_MISSING_MODEL="${SKIP_MISSING_MODEL:-1}"
OUT_DIR="${OUT_DIR:-/tmp/rocmfpx-qwen-large}"
PRESET_LEAN="${PRESET_LEAN:-Q3_0_ROCMFPX}"
PRESET_AGENT="${PRESET_AGENT:-Q3_0_ROCMFPX_AGENT}"
RUN_SWEEP="${RUN_SWEEP:-1}"

cd "$ROOT"

if [[ -z "$MODEL_SRC" || ! -f "$MODEL_SRC" ]]; then
    if [[ "$SKIP_MISSING_MODEL" == "1" ]]; then
        python3 - <<PY
import json
print(json.dumps({
    "status": "skip",
    "reason": "set MODEL_SRC to a Qwen3 4B/8B+ BF16 or F16 GGUF fixture",
}))
PY
        exit 0
    fi
    echo "missing MODEL_SRC for large-model gate" >&2
    exit 1
fi

if [[ ! -x "$QUANTIZE_BIN" ]]; then
    echo "missing llama-quantize: $QUANTIZE_BIN" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
base="$(basename "$MODEL_SRC" .gguf)"

if [[ -z "$MODEL_LEAN" ]]; then
    MODEL_LEAN="$OUT_DIR/${base}-${PRESET_LEAN}.gguf"
fi
if [[ -z "$MODEL_AGENT" ]]; then
    MODEL_AGENT="$OUT_DIR/${base}-${PRESET_AGENT}.gguf"
fi

if [[ ! -f "$MODEL_LEAN" ]]; then
    "$QUANTIZE_BIN" "$MODEL_SRC" "$MODEL_LEAN" "$PRESET_LEAN"
fi
if [[ ! -f "$MODEL_AGENT" ]]; then
    "$QUANTIZE_BIN" "$MODEL_SRC" "$MODEL_AGENT" "$PRESET_AGENT"
fi

MODEL="$MODEL_LEAN" BACKEND="$BACKEND" scripts/check-rocmfpx-qwen-coherency.sh
MODEL="$MODEL_LEAN" BACKEND="$BACKEND" scripts/check-rocmfpx-agent-json.sh
MODEL="$MODEL_AGENT" BACKEND="$BACKEND" scripts/check-rocmfpx-agent-json.sh
MODEL="$MODEL_AGENT" BACKEND="$BACKEND" scripts/check-rocmfpx-tool-calling.sh

if [[ "$RUN_SWEEP" == "1" ]]; then
    MODEL_SRC="$MODEL_SRC" OUT_DIR="$OUT_DIR/sweep" RUN_AGENT_JSON=1 \
        PRESETS="Q3_K_M Q3_0_ROCMFPX Q3_0_ROCMFPX_AGENT Q6_K Q6_0_ROCMFPX Q6_0_ROCMFPX_AGENT Q8_0 Q8_0_ROCMFPX Q8_0_ROCMFPX_AGENT" \
        scripts/sweep-rocmfpx-agent-routing.sh
fi

python3 - <<PY
import json
print(json.dumps({
    "status": "pass",
    "model_src": "$MODEL_SRC",
    "model_lean": "$MODEL_LEAN",
    "model_agent": "$MODEL_AGENT",
    "backend": "$BACKEND",
}, indent=2))
PY
