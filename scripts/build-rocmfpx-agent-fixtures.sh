#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
QUANTIZE_BIN="${QUANTIZE_BIN:-$BUILD_DIR/bin/llama-quantize}"
MODEL_SRC="${MODEL_SRC:-/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q4_K_M.gguf}"
HERMES_SRC="${HERMES_SRC:-}"
OPENCLAW_SRC="${OPENCLAW_SRC:-}"
HERMES_OUT_DIR="${HERMES_OUT_DIR:-/home/caf/strix-fp4/models/rocmfpx-hermes-tests}"
OPENCLAW_OUT_DIR="${OPENCLAW_OUT_DIR:-/home/caf/strix-fp4/models/rocmfpx-openclaw-tests}"
PRESET="${PRESET:-Q3_0_ROCMFPX_AGENT}"
BUILD_PROXY="${BUILD_PROXY:-1}"
ALLOW_REQUANTIZE="${ALLOW_REQUANTIZE:-1}"

cd "$ROOT"

if [[ ! -x "$QUANTIZE_BIN" ]]; then
    echo "missing llama-quantize: $QUANTIZE_BIN" >&2
    exit 1
fi

quantize_one() {
    local src="$1"
    local out="$2"
    mkdir -p "$(dirname "$out")"
    if [[ -f "$out" ]]; then
        echo "exists: $out"
        return 0
    fi
    local -a quant_args=()
    if [[ "$ALLOW_REQUANTIZE" == "1" ]]; then
        quant_args+=(--allow-requantize)
    fi
    echo "quantizing $src -> $out ($PRESET)"
    "$QUANTIZE_BIN" "${quant_args[@]}" "$src" "$out" "$PRESET"
}

mkdir -p "$HERMES_OUT_DIR" "$OPENCLAW_OUT_DIR"

if [[ -n "$HERMES_SRC" && -f "$HERMES_SRC" ]]; then
    quantize_one "$HERMES_SRC" "$HERMES_OUT_DIR/hermes-${PRESET}.gguf"
elif [[ "$BUILD_PROXY" == "1" && -f "$MODEL_SRC" ]]; then
    quantize_one "$MODEL_SRC" "$HERMES_OUT_DIR/hermes-${PRESET}.gguf"
    echo "NOTE: Hermes fixture is a Qwen proxy from $MODEL_SRC until HERMES_SRC is set"
fi

if [[ -n "$OPENCLAW_SRC" && -f "$OPENCLAW_SRC" ]]; then
    quantize_one "$OPENCLAW_SRC" "$OPENCLAW_OUT_DIR/openclaw-${PRESET}.gguf"
elif [[ "$BUILD_PROXY" == "1" && -f "$MODEL_SRC" ]]; then
    quantize_one "$MODEL_SRC" "$OPENCLAW_OUT_DIR/openclaw-${PRESET}.gguf"
    echo "NOTE: OpenClaw fixture is a Qwen proxy from $MODEL_SRC until OPENCLAW_SRC is set"
fi

python3 - <<PY
import json
print(json.dumps({
    "status": "pass",
    "hermes": "$HERMES_OUT_DIR/hermes-${PRESET}.gguf",
    "openclaw": "$OPENCLAW_OUT_DIR/openclaw-${PRESET}.gguf",
    "preset": "$PRESET",
    "proxy": ${BUILD_PROXY},
}, indent=2))
PY
