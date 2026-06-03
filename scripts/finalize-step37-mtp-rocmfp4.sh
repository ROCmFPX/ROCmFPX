#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-rocmfp4-deepseek-clean-upstream}"
CONVERT_SESSION="${CONVERT_SESSION:-step37-mtp-convert}"
SOURCE="${SOURCE:-/mnt/ai-models/rocmfp4-quants/Step-3.7-flash-BF16-MTP.gguf}"
MODEL_DIR="${MODEL_DIR:-/home/caf/strix-fp4/models/Step-3.7-Flash-MTP-GGUF}"
OUTPUT="${OUTPUT:-$MODEL_DIR/Step-3.7-flash-BF16-MTP-to-ROCmFP4-STRIX_LEAN.gguf}"
QUANT_TYPE="${QUANT_TYPE:-Q4_0_ROCMFP4_STRIX_LEAN}"
MIN_FREE_BYTES="${MIN_FREE_BYTES:-130000000000}"

cd "$ROOT"

if tmux has-session -t "$CONVERT_SESSION" 2>/dev/null; then
    echo "Conversion is still running in tmux session: $CONVERT_SESSION" >&2
    echo "Monitor it with: tmux attach -t $CONVERT_SESSION" >&2
    exit 1
fi

if [[ ! -s "$SOURCE" ]]; then
    echo "Missing completed BF16+MTP source: $SOURCE" >&2
    exit 1
fi

if [[ -e "$OUTPUT" ]]; then
    echo "Refusing to overwrite existing ROCmFP4 output: $OUTPUT" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"

available_bytes="$(df --output=avail -B1 "$MODEL_DIR" | tail -n 1 | tr -d ' ')"
if (( available_bytes < MIN_FREE_BYTES )); then
    echo "Insufficient free space in $MODEL_DIR" >&2
    echo "Available bytes: $available_bytes; required minimum: $MIN_FREE_BYTES" >&2
    exit 1
fi

echo "== Validate completed BF16+MTP GGUF header =="
"$SCRIPT_DIR/check-step37-mtp-gguf-metadata.sh" "$SOURCE"

echo
echo "== Quantize directly to the internal NVMe =="
"$BUILD_DIR/bin/llama-quantize" "$SOURCE" "$OUTPUT" "$QUANT_TYPE"

echo
echo "== Validate internal-drive ROCmFP4 GGUF header =="
"$SCRIPT_DIR/check-step37-mtp-gguf-metadata.sh" "$OUTPUT"

echo
echo "Created: $OUTPUT"
echo "Keep the old internal no-MTP file until StepFun MTP runtime validation passes."
