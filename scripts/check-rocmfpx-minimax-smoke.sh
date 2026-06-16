#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
QUANTIZE_BIN="${QUANTIZE_BIN:-$BUILD_DIR/bin/llama-quantize}"
MODEL_SRC="${MODEL_SRC:-}"
MODEL_OUT="${MODEL_OUT:-/home/caf/strix-fp4/models/minimax-m3-rocmfpx/MiniMax-M3-Q3_0_ROCMFPX.gguf}"
PRESET="${PRESET:-Q3_0_ROCMFPX}"
SKIP_MISSING_MODEL="${SKIP_MISSING_MODEL:-1}"
RUN_COHERENCY="${RUN_COHERENCY:-0}"
BACKEND="${BACKEND:-ROCm0}"

cd "$ROOT"

if [[ -z "$MODEL_SRC" || ! -f "$MODEL_SRC" ]]; then
    if [[ "$SKIP_MISSING_MODEL" == "1" ]]; then
        python3 - <<PY
import json
print(json.dumps({
    "status": "skip",
    "reason": "set MODEL_SRC to MiniMax-M3 BF16/F16/Q4_K source on external NVMe or local path",
}))
PY
        exit 0
    fi
    echo "missing MODEL_SRC for MiniMax smoke" >&2
    exit 1
fi

if [[ ! -x "$QUANTIZE_BIN" ]]; then
    echo "missing llama-quantize: $QUANTIZE_BIN" >&2
    exit 1
fi

mkdir -p "$(dirname "$MODEL_OUT")"
if [[ ! -f "$MODEL_OUT" ]]; then
    "$QUANTIZE_BIN" "$MODEL_SRC" "$MODEL_OUT" "$PRESET"
fi

if [[ "$RUN_COHERENCY" == "1" ]]; then
    MODEL="$MODEL_OUT" BACKEND="$BACKEND" scripts/check-rocmfpx-qwen-coherency.sh
fi

python3 - <<PY
import json
print(json.dumps({
    "status": "pass",
    "model_src": "$MODEL_SRC",
    "model_out": "$MODEL_OUT",
    "preset": "$PRESET",
}, indent=2))
PY
