#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE="${ROOT_DIR}/scripts/run-deepseek-v4-mtp-rocmfp4-smoke.sh"
MEMORY_MAX="${MEMORY_MAX:-108G}"
MEMORY_SWAP_MAX="${MEMORY_SWAP_MAX:-0}"

if ! command -v systemd-run >/dev/null 2>&1; then
    printf 'missing required containment tool: systemd-run\n' >&2
    exit 1
fi

exec systemd-run \
    --user \
    --pipe \
    --wait \
    --collect \
    -p "MemoryMax=${MEMORY_MAX}" \
    -p "MemorySwapMax=${MEMORY_SWAP_MAX}" \
    /usr/bin/env \
        "CONTEXT=${CONTEXT:-128}" \
        "N_PREDICT=${N_PREDICT:-8}" \
        "N_GPU_LAYERS=${N_GPU_LAYERS:-999}" \
        "SPEC_DRAFT_N_GPU_LAYERS=${SPEC_DRAFT_N_GPU_LAYERS:-all}" \
        "CACHE_TYPE_K=${CACHE_TYPE_K:-f16}" \
        "CACHE_TYPE_V=${CACHE_TYPE_V:-f16}" \
        "SPEC_DRAFT_TYPE_K=${SPEC_DRAFT_TYPE_K:-f16}" \
        "SPEC_DRAFT_TYPE_V=${SPEC_DRAFT_TYPE_V:-f16}" \
        "TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-900}" \
        "MIN_AVAILABLE_MEMORY_GIB=${MIN_AVAILABLE_MEMORY_GIB:-110}" \
        "MAX_TMP_USED_GIB=${MAX_TMP_USED_GIB:-2}" \
        "${SMOKE}" \
        "$@"
