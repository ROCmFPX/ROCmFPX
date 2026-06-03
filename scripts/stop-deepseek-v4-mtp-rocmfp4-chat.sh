#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-/home/caf/strix-fp4/llama.cpp-deepseek-v4-rocmfp4-modern-port/build-deepseek4-rocmfp4/bin/llama-cli}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/DeepSeek-V4-Flash-180B-GGUF/DeepSeek-V4-Flash-180B-MTP-MXFP4-to-ROCmFP4-STRIX_LEAN.gguf}"
WAIT_SECONDS="${WAIT_SECONDS:-10}"
PATTERN="${BIN} -m ${MODEL}"

mapfile -t pids < <(pgrep -f -- "${PATTERN}" || true)

if (( ${#pids[@]} == 0 )); then
    printf 'No DeepSeek V4 ROCmFP4 terminal chat worker is running.\n'
    exit 0
fi

printf 'Stopping DeepSeek V4 ROCmFP4 terminal chat worker(s): %s\n' "${pids[*]}"
kill -TERM "${pids[@]}"

survivors=("${pids[@]}")
for (( second = 0; second < WAIT_SECONDS; ++second )); do
    survivors=()
    for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            survivors+=("${pid}")
        fi
    done

    if (( ${#survivors[@]} == 0 )); then
        printf 'DeepSeek V4 ROCmFP4 terminal chat stopped cleanly.\n'
        exit 0
    fi

    sleep 1
done

printf 'Force-stopping stuck DeepSeek V4 ROCmFP4 terminal chat worker(s): %s\n' "${survivors[*]}" >&2
kill -KILL "${survivors[@]}"
