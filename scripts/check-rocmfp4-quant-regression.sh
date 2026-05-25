#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
SIZE="${SIZE:-65536}"
ITERS="${ITERS:-20}"

# Conservative ceilings for the accepted Strix Halo CPU quantizer path.
# These are intentionally above the best observed values so normal run-to-run
# noise does not fail the guard, while obvious regressions are caught.
MAX_Q4_CYCLES="${MAX_Q4_CYCLES:-5450}"
MAX_FAST_CYCLES="${MAX_FAST_CYCLES:-4700}"
MAX_Q4_IMATRIX_CYCLES="${MAX_Q4_IMATRIX_CYCLES:-7350}"
MAX_FAST_IMATRIX_CYCLES="${MAX_FAST_IMATRIX_CYCLES:-6300}"
MAX_Q4_DEQUANT_CYCLES="${MAX_Q4_DEQUANT_CYCLES:-42}"
MAX_FAST_DEQUANT_CYCLES="${MAX_FAST_DEQUANT_CYCLES:-42}"
MAX_Q4_VEC_DOT_CYCLES="${MAX_Q4_VEC_DOT_CYCLES:-36}"
MAX_FAST_VEC_DOT_CYCLES="${MAX_FAST_VEC_DOT_CYCLES:-32}"

cd "$ROOT"

"$BUILD_DIR/bin/test-quantize-fns" >/tmp/rocmfp4-quant-fns.log

run_perf() {
    local type="$1"
    shift
    "$BUILD_DIR/bin/test-quantize-perf" --type "$type" -i "$ITERS" --size "$SIZE" "$@"
}

extract_cycles() {
    awk '
        /quantize_row_q$/ { in_q = 1; next }
        in_q && /min cycles\/32 vals/ {
            print $NF;
            exit;
        }
    '
}

extract_op_cycles() {
    local op="$1"
    awk -v op="$op" '
        $0 ~ "  " op "$" { in_op = 1; next }
        in_op && /min cycles\/32 vals/ {
            print $NF;
            exit;
        }
    '
}

q4_output="$(run_perf q4_0_rocmfp4)"
fast_output="$(run_perf q4_0_rocmfp4_fast)"
q4_imatrix_output="$(run_perf q4_0_rocmfp4 --op quantize_row_q --imatrix)"
fast_imatrix_output="$(run_perf q4_0_rocmfp4_fast --op quantize_row_q --imatrix)"

printf "%s\n" "$q4_output"
printf "%s\n" "$fast_output"
printf "%s\n" "$q4_imatrix_output"
printf "%s\n" "$fast_imatrix_output"

q4_cycles="$(printf "%s\n" "$q4_output" | extract_cycles)"
fast_cycles="$(printf "%s\n" "$fast_output" | extract_cycles)"
q4_dequant_cycles="$(printf "%s\n" "$q4_output" | extract_op_cycles dequantize_row_q)"
fast_dequant_cycles="$(printf "%s\n" "$fast_output" | extract_op_cycles dequantize_row_q)"
q4_vec_dot_cycles="$(printf "%s\n" "$q4_output" | extract_op_cycles vec_dot_q)"
fast_vec_dot_cycles="$(printf "%s\n" "$fast_output" | extract_op_cycles vec_dot_q)"
q4_imatrix_cycles="$(printf "%s\n" "$q4_imatrix_output" | extract_cycles)"
fast_imatrix_cycles="$(printf "%s\n" "$fast_imatrix_output" | extract_cycles)"

if [[ -z "$q4_cycles" || -z "$fast_cycles" ||
      -z "$q4_dequant_cycles" || -z "$fast_dequant_cycles" ||
      -z "$q4_vec_dot_cycles" || -z "$fast_vec_dot_cycles" ||
      -z "$q4_imatrix_cycles" || -z "$fast_imatrix_cycles" ]]; then
    echo "FAIL: could not parse ROCmFP4 quantizer cycles" >&2
    exit 1
fi

check_max() {
    local label="$1"
    local got="$2"
    local max="$3"

    awk -v label="$label" -v got="$got" -v max="$max" 'BEGIN {
        if (got + 0 > max + 0) {
            printf("FAIL: %s %.2f cycles/32 exceeds max %.2f\n", label, got, max) > "/dev/stderr";
            exit 1;
        }
        printf("PASS: %s %.2f cycles/32 <= %.2f\n", label, got, max);
    }'
}

check_max "q4_0_rocmfp4" "$q4_cycles" "$MAX_Q4_CYCLES"
check_max "q4_0_rocmfp4_fast" "$fast_cycles" "$MAX_FAST_CYCLES"
check_max "q4_0_rocmfp4 dequant" "$q4_dequant_cycles" "$MAX_Q4_DEQUANT_CYCLES"
check_max "q4_0_rocmfp4_fast dequant" "$fast_dequant_cycles" "$MAX_FAST_DEQUANT_CYCLES"
check_max "q4_0_rocmfp4 vec_dot" "$q4_vec_dot_cycles" "$MAX_Q4_VEC_DOT_CYCLES"
check_max "q4_0_rocmfp4_fast vec_dot" "$fast_vec_dot_cycles" "$MAX_FAST_VEC_DOT_CYCLES"
check_max "q4_0_rocmfp4 imatrix" "$q4_imatrix_cycles" "$MAX_Q4_IMATRIX_CYCLES"
check_max "q4_0_rocmfp4_fast imatrix" "$fast_imatrix_cycles" "$MAX_FAST_IMATRIX_CYCLES"
