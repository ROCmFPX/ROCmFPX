#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BIN="${BIN:-$ROOT/build-strix-rocmfp4/bin/test-backend-ops}"
MAX_FAST_US="${MAX_FAST_US:-50.0}"
MAX_DUAL_US="${MAX_DUAL_US:-56.0}"
MAX_FAST_N2_US="${MAX_FAST_N2_US:-64.0}"
MAX_DUAL_N2_US="${MAX_DUAL_N2_US:-65.0}"
MAX_FAST_N4_US="${MAX_FAST_N4_US:-100.0}"
MAX_DUAL_N4_US="${MAX_DUAL_N4_US:-94.0}"
MAX_FAST_N8_US="${MAX_FAST_N8_US:-178.0}"
MAX_DUAL_N8_US="${MAX_DUAL_N8_US:-170.0}"

cd "$ROOT"

run_mul_mat_perf() {
    local type="$1"
    local ncols="${2:-1}"
    env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
        "$BIN" perf \
            -b ROCm0 \
            -o MUL_MAT \
            -p "type_a=${type},type_b=f32,m=4096,n=${ncols},k=14336" \
            --output console
}

extract_us() {
    sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g' | sed -n 's/.* \([0-9.][0-9.]*\) us\/run.*/\1/p' | tail -n 1
}

check_case() {
    local label="$1"
    local type="$2"
    local ncols="$3"
    local max_us="$4"

    local output us
    output="$(run_mul_mat_perf "$type" "$ncols")"
    printf "%s\n" "$output"

    us="$(printf "%s\n" "$output" | extract_us)"
    if [[ -z "$us" ]]; then
        echo "FAIL: could not parse ${label} ROCm MUL_MAT timing" >&2
        exit 1
    fi

    awk -v label="$label" -v got="$us" -v max="$max_us" 'BEGIN {
        if (got + 0 > max + 0) {
            printf("FAIL: %s ROCm MUL_MAT %.2f us/run exceeds max %.2f\n", label, got, max) > "/dev/stderr";
            exit 1;
        }
        printf("PASS: %s ROCm MUL_MAT %.2f us/run <= %.2f\n", label, got, max);
    }'
}

check_case "q4_0_rocmfp4_fast n=1" q4_0_rocmfp4_fast 1 "$MAX_FAST_US"
check_case "q4_0_rocmfp4 n=1"      q4_0_rocmfp4      1 "$MAX_DUAL_US"
check_case "q4_0_rocmfp4_fast n=2" q4_0_rocmfp4_fast 2 "$MAX_FAST_N2_US"
check_case "q4_0_rocmfp4 n=2"      q4_0_rocmfp4      2 "$MAX_DUAL_N2_US"
check_case "q4_0_rocmfp4_fast n=4" q4_0_rocmfp4_fast 4 "$MAX_FAST_N4_US"
check_case "q4_0_rocmfp4 n=4"      q4_0_rocmfp4      4 "$MAX_DUAL_N4_US"
check_case "q4_0_rocmfp4_fast n=8" q4_0_rocmfp4_fast 8 "$MAX_FAST_N8_US"
check_case "q4_0_rocmfp4 n=8"      q4_0_rocmfp4      8 "$MAX_DUAL_N8_US"
