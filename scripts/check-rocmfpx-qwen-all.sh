#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-LEAN.gguf}"
BACKEND="${BACKEND:-ROCm0}"

cd "$ROOT"

scripts/check-rocmfpx-reference.sh
MODEL="$MODEL" BACKEND="$BACKEND" scripts/check-rocmfpx-qwen-coherency.sh
MODEL="$MODEL" BACKEND="$BACKEND" scripts/check-rocmfpx-qwen-bench.sh
MODEL="$MODEL" BACKEND="$BACKEND" scripts/check-rocmfpx-qwen-strict-json.sh
MODEL="$MODEL" BACKEND="$BACKEND" scripts/check-rocmfpx-agent-json.sh
MODEL="$MODEL" BACKEND="$BACKEND" scripts/check-rocmfpx-tool-calling.sh

echo "ROCmFPX Qwen validation passed for ${MODEL} on ${BACKEND}"
