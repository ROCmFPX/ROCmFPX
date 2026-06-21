#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPT_DIR/rocmfpx-lib.sh"
rocmfpx_setup_env

BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
BIN="${BIN:-$BUILD_DIR/bin/llama-server}"
MODEL="${MODEL:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18180}"
ALIAS="${ALIAS:-rocmfpx-mtp}"
DEVICE="${DEVICE:-Vulkan0}"
SPEC_DRAFT_DEVICE="${SPEC_DRAFT_DEVICE:-$DEVICE}"
CTX_SIZE="${CTX_SIZE:-32768}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
THREADS="${THREADS:-16}"
THREADS_BATCH="${THREADS_BATCH:-32}"
SPEC_DRAFT_THREADS="${SPEC_DRAFT_THREADS:-$THREADS}"
SPEC_DRAFT_THREADS_BATCH="${SPEC_DRAFT_THREADS_BATCH:-$THREADS_BATCH}"
CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${CACHE_TYPE_V:-$CACHE_TYPE_K}"
CACHE_TYPE_K_DRAFT="${CACHE_TYPE_K_DRAFT:-$CACHE_TYPE_K}"
CACHE_TYPE_V_DRAFT="${CACHE_TYPE_V_DRAFT:-$CACHE_TYPE_V}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-4}"
SPEC_DRAFT_N_MIN="${SPEC_DRAFT_N_MIN:-0}"
SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-0.0}"
SPEC_DRAFT_P_SPLIT="${SPEC_DRAFT_P_SPLIT:-0.10}"
CACHE_RAM="${CACHE_RAM:-0}"
STRICT_BENCH="${STRICT_BENCH:-1}"
NO_MMPROJ="${NO_MMPROJ:-1}"

if [[ -z "$MODEL" ]]; then
    echo "MODEL must point to a ROCmFPX/ROCmFP4 GGUF" >&2
    exit 2
fi

rocmfpx_require_binary "$BIN"
SKIP_MISSING_MODEL=0 rocmfpx_require_model "$MODEL"

cache_args=(--cache-ram "$CACHE_RAM")
if [[ "$STRICT_BENCH" == "1" ]]; then
    cache_args=(--no-cache-prompt --cache-ram "$CACHE_RAM" --slot-prompt-similarity 0.0)
fi

mmproj_args=()
if [[ "$NO_MMPROJ" == "1" ]]; then
    mmproj_args=(--no-mmproj)
fi

cd "$ROOT"

exec "$BIN" \
    -m "$MODEL" \
    --alias "$ALIAS" \
    --host "$HOST" \
    --port "$PORT" \
    --jinja \
    -c "$CTX_SIZE" \
    --reasoning off \
    --reasoning-format none \
    --reasoning-budget -1 \
    --no-context-shift \
    -dev "$DEVICE" \
    -ngl 999 \
    -fa on \
    -b "$BATCH_SIZE" \
    -ub "$UBATCH_SIZE" \
    -t "$THREADS" \
    -tb "$THREADS_BATCH" \
    -ctk "$CACHE_TYPE_K" \
    -ctv "$CACHE_TYPE_V" \
    --temp 0 \
    --top-p 0.95 \
    --top-k 20 \
    --seed 123 \
    --parallel 1 \
    "${mmproj_args[@]}" \
    --metrics \
    --no-webui \
    "${cache_args[@]}" \
    --spec-type draft-mtp \
    --spec-draft-device "$SPEC_DRAFT_DEVICE" \
    --spec-draft-ngl all \
    --spec-draft-threads "$SPEC_DRAFT_THREADS" \
    --spec-draft-threads-batch "$SPEC_DRAFT_THREADS_BATCH" \
    --spec-draft-type-k "$CACHE_TYPE_K_DRAFT" \
    --spec-draft-type-v "$CACHE_TYPE_V_DRAFT" \
    --spec-draft-n-max "$SPEC_DRAFT_N_MAX" \
    --spec-draft-n-min "$SPEC_DRAFT_N_MIN" \
    --spec-draft-p-min "$SPEC_DRAFT_P_MIN" \
    --spec-draft-p-split "$SPEC_DRAFT_P_SPLIT" \
    --no-spec-draft-backend-sampling \
    --spec-draft-poll 1 \
    --spec-draft-poll-batch 1 \
    "$@"
