#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-fp2-vulkan-q81}"
BACKEND="${BACKEND:-Vulkan0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
OUTPUT_DIR="${OUTPUT_DIR:-$BUILD_DIR/rocmfp2-q81-gates}"
TEST_BIN="$BUILD_DIR/bin/test-backend-ops"

if [[ ! -x "$TEST_BIN" ]]; then
    printf 'error: test binary is not executable: %s\n' "$TEST_BIN" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

run_gate() {
    local op="$1"
    local n="$2"
    local mode="$3"
    local params
    local log="$OUTPUT_DIR/${op,,}-n${n}-${mode}.log"

    if [[ "$op" == "MUL_MAT" ]]; then
        params="type_a=q2_0_rocmfpx,type_b=f32,m=64,n=${n},k=2048,bs=\\[1,1\\],nr=\\[1,1\\]"
    else
        params="type_a=q2_0_rocmfpx,type_b=f32,n_mats=8,n_used=2,b=0,m=64,n=${n},k=2048"
    fi

    printf 'running %s n=%s mode=%s on %s\n' "$op" "$n" "$mode" "$BACKEND"
    if [[ "$mode" == "forced-mmvq" ]]; then
        env -u GGML_VK_DISABLE_MMVQ GGML_VK_FORCE_MMVQ=1 \
            timeout --kill-after=15s "$TIMEOUT_SEC" \
            "$TEST_BIN" test -b "$BACKEND" -o "$op" -p "$params" 2>&1 |
            tee "$log"
    else
        env -u GGML_VK_FORCE_MMVQ GGML_VK_DISABLE_MMVQ=1 \
            timeout --kill-after=15s "$TIMEOUT_SEC" \
            "$TEST_BIN" test -b "$BACKEND" -o "$op" -p "$params" 2>&1 |
            tee "$log"
    fi

    if ! grep -Fq "1/1 tests passed" "$log"; then
        printf 'error: expected exactly one passing test; see %s\n' "$log" >&2
        exit 1
    fi
}

for op in MUL_MAT MUL_MAT_ID; do
    for n in 1 2 4 8; do
        run_gate "$op" "$n" forced-mmvq
        run_gate "$op" "$n" f32-fallback
    done
done

printf 'all 16 ROCmFP2 Vulkan Q8_1/fallback gates passed; logs: %s\n' "$OUTPUT_DIR"
