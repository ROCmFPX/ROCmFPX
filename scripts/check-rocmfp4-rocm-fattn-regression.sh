#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BIN="${BIN:-$ROOT/build-strix-rocmfp4/bin/test-backend-ops}"
MAX_DUAL_US="${MAX_DUAL_US:-88.0}"
MAX_FAST_US="${MAX_FAST_US:-84.0}"
MAX_QWEN_DUAL_US="${MAX_QWEN_DUAL_US:-240.0}"
MAX_QWEN_FAST_US="${MAX_QWEN_FAST_US:-215.0}"

cd "$ROOT"

output="$(
    env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
        "$BIN" perf \
            -b ROCm0 \
            -o FLASH_ATTN_EXT \
            -p rocmfp4 \
            --output console
)"

printf "%s\n" "$output"

extract_us() {
    local shape="$1"
    local type="$2"
    printf "%s\n" "$output" |
        sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g' |
        grep -F "$shape" |
        sed -n "/type_K=${type},type_V=${type}/s/.* \([0-9.][0-9.]*\) us\/run.*/\1/p" |
        tail -n 1
}

check_case() {
    local label="$1"
    local shape="$2"
    local type="$3"
    local max_us="$4"
    local got

    got="$(extract_us "$shape" "$type")"
    if [[ -z "$got" ]]; then
        echo "FAIL: could not parse ${label} ROCmFP4 FlashAttention timing" >&2
        exit 1
    fi

    awk -v label="$label" -v got="$got" -v max="$max_us" 'BEGIN {
        if (got + 0 > max + 0) {
            printf("FAIL: %s ROCmFP4 FlashAttention %.2f us/run exceeds max %.2f\n", label, got, max) > "/dev/stderr";
            exit 1;
        }
        printf("PASS: %s ROCmFP4 FlashAttention %.2f us/run <= %.2f\n", label, got, max);
    }'
}

shape_default='hsk=64,hsv=64,nh=8,nr23=[8,1],kv=7680,nb=1'
shape_qwen='hsk=128,hsv=128,nh=8,nr23=[12,1],kv=7680,nb=1'

check_case "dual-scale 64d" "$shape_default" q4_0_rocmfp4 "$MAX_DUAL_US"
check_case "FAST 64d" "$shape_default" q4_0_rocmfp4_fast "$MAX_FAST_US"
check_case "dual-scale Qwen-style 128d" "$shape_qwen" q4_0_rocmfp4 "$MAX_QWEN_DUAL_US"
check_case "FAST Qwen-style 128d" "$shape_qwen" q4_0_rocmfp4_fast "$MAX_QWEN_FAST_US"
