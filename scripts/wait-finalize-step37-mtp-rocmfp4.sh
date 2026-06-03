#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONVERT_SESSION="${CONVERT_SESSION:-step37-mtp-convert}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SOURCE="${SOURCE:-/mnt/ai-models/rocmfp4-quants/Step-3.7-flash-BF16-MTP.gguf}"

cd "$ROOT"

while tmux has-session -t "$CONVERT_SESSION" 2>/dev/null; do
    if [[ -e "$SOURCE" ]]; then
        stat -c "Waiting for conversion: size=%s modified=%y" "$SOURCE"
    else
        echo "Waiting for conversion output: $SOURCE"
    fi
    sleep "$POLL_SECONDS"
done

echo "Conversion session ended. Starting guarded NVMe finalization."
"$SCRIPT_DIR/finalize-step37-mtp-rocmfp4.sh"

echo
echo "Starting first StepFun ROCmFP4 native-MTP smoke."
"$SCRIPT_DIR/run-step37-mtp-rocmfp4-smoke.sh"
