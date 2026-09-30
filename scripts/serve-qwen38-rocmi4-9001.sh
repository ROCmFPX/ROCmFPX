#!/usr/bin/env bash
# Serve Qwen3.8-27B Q4_0_ROCMI4 on port 9001 inside a ROCm 10 container.
# W4A4 HIP build. No speculative decoding / MTP. 80k context.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-rocm-dev:10.0.0}"
NAME="${NAME:-rocmfpx-qwen38-27b-rocmi4}"
HOST_IP="${HOST_IP:-192.168.1.200}"
PORT="${PORT:-9001}"
MODEL="${MODEL:-/home/ghazni/models/rocmfpx/cafonez/Qwen3.8-27B-ROCmI4/Qwen3.8-27B-Q4_0_ROCMI4.gguf}"
BIN_DIR="${BIN_DIR:-$ROOT/build-rdna3-w4a4/bin}"
ALIAS="${ALIAS:-qwen38-27b-rocmi4-w4a4}"
REL_BIN="${BIN_DIR#"$ROOT"/}"

if [[ ! -x "$BIN_DIR/llama-server" ]]; then
  echo "Missing $BIN_DIR/llama-server — build first (scripts/build-rdna3-w4a4.sh)" >&2
  exit 1
fi

docker rm -f "$NAME" 2>/dev/null || true

exec docker run -d --name "$NAME" --restart unless-stopped \
  --device=/dev/kfd --device=/dev/dri \
  --group-add 44 --group-add 993 \
  -p "${HOST_IP}:${PORT}:9001" \
  -v "$ROOT:/workspace:ro" \
  -v /home/ghazni/models:/home/ghazni/models:ro \
  -e ROCM_PATH=/opt/rocm \
  -e LD_LIBRARY_PATH="/workspace/${REL_BIN}:/opt/rocm/lib" \
  -w /workspace \
  "$IMAGE" \
  "/workspace/${REL_BIN}/llama-server" \
    -m "$MODEL" \
    --host 0.0.0.0 --port 9001 \
    --alias "$ALIAS" \
    -dev ROCm0 -ngl 999 -np 1 \
    -c 80000 -b 512 -ub 256 -t 16 -fa on \
    -ctk f16 -ctv f16 --jinja
