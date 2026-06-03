#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE="${ROOT_DIR}/scripts/run-deepseek-v4-mtp-rocmfp4-capped-smoke.sh"
MODEL="${MODEL:-/home/caf/strix-fp4/models/DeepSeek-V4-Flash-180B-GGUF/DeepSeek-V4-Flash-180B-MTP-MXFP4-to-ROCmFP4-STRIX_LEAN.gguf}"
BACKEND="${BACKEND:-ROCm0}"
CONTEXTS="${CONTEXTS:-32768 65536 131072 200000 262144 524288 1048576}"
N_PREDICT="${N_PREDICT:-2}"

if [[ ! -x "${SMOKE}" ]]; then
    printf 'missing capped smoke runner: %s\n' "${SMOKE}" >&2
    exit 1
fi

printf 'DeepSeek V4 ROCmFP4 native-MTP context ladder: backend=%s model=%s\n' "${BACKEND}" "${MODEL}"
printf 'contexts: %s\n' "${CONTEXTS}"

for context in ${CONTEXTS}; do
    printf '\n=== context=%s ===\n' "${context}"
    free -h

    if CONTEXT="${context}" N_PREDICT="${N_PREDICT}" "${SMOKE}" "${MODEL}" "${BACKEND}"; then
        printf 'PASS: context=%s\n' "${context}"
    else
        status="$?"
        printf 'STOP: context=%s failed with status=%s\n' "${context}" "${status}" >&2
        exit "${status}"
    fi

    sleep 5
done

printf '\nPASS: all requested DeepSeek V4 context rungs completed\n'
