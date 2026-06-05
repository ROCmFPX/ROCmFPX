#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BIN="${BIN:-$ROOT/build-strix-rocmfp4/bin/test-backend-ops}"

# Conservative ceilings for the large Vulkan0 CPY perf shape. Runtime
# quantization is slower on RADV Vulkan than ROCm, but these bounds catch
# fallback and also protect the ROCmFP4 lower-scale pruning speedup.
MAX_F32_TO_Q4_US="${MAX_F32_TO_Q4_US:-11500.0}"
MAX_F16_TO_Q4_US="${MAX_F16_TO_Q4_US:-3000.0}"
MAX_BF16_TO_Q4_US="${MAX_BF16_TO_Q4_US:-3100.0}"
MAX_Q4_TO_F32_US="${MAX_Q4_TO_F32_US:-700.0}"
MAX_F32_TO_FAST_US="${MAX_F32_TO_FAST_US:-12200.0}"
MAX_F16_TO_FAST_US="${MAX_F16_TO_FAST_US:-3600.0}"
MAX_BF16_TO_FAST_US="${MAX_BF16_TO_FAST_US:-3700.0}"
MAX_FAST_TO_F32_US="${MAX_FAST_TO_F32_US:-700.0}"

cd "$ROOT"

echo "== ROCmFP4 Vulkan CPY correctness =="
env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
    "$BIN" test \
        -b Vulkan0 \
        -o CPY \
        -p "q4_0_rocmfp4" \
        --output console

echo
echo "== ROCmFP4 Vulkan CPY performance =="
tmp_output="$(mktemp)"
trap 'rm -f "$tmp_output"' EXIT
env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}" \
    "$BIN" perf \
        -b Vulkan0 \
        -o CPY \
        -p "q4_0_rocmfp4" \
        --output console | tee "$tmp_output"

output="$(cat "$tmp_output")"

extract_us() {
    local pattern="$1"
    printf "%s\n" "$output" |
        sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g' |
        sed -n "/${pattern}/s/.* \([0-9.][0-9.]*\) us\/run.*/\1/p" |
        tail -n 1
}

f32_to_q4_us="$(extract_us 'type_src=f32,type_dst=q4_0_rocmfp4,')"
f16_to_q4_us="$(extract_us 'type_src=f16,type_dst=q4_0_rocmfp4,')"
bf16_to_q4_us="$(extract_us 'type_src=bf16,type_dst=q4_0_rocmfp4,')"
q4_to_f32_us="$(extract_us 'type_src=q4_0_rocmfp4,type_dst=f32,')"
f32_to_fast_us="$(extract_us 'type_src=f32,type_dst=q4_0_rocmfp4_fast,')"
f16_to_fast_us="$(extract_us 'type_src=f16,type_dst=q4_0_rocmfp4_fast,')"
bf16_to_fast_us="$(extract_us 'type_src=bf16,type_dst=q4_0_rocmfp4_fast,')"
fast_to_f32_us="$(extract_us 'type_src=q4_0_rocmfp4_fast,type_dst=f32,')"

if [[ -z "$f32_to_q4_us" || -z "$f16_to_q4_us" || -z "$bf16_to_q4_us" || -z "$q4_to_f32_us" ||
      -z "$f32_to_fast_us" || -z "$f16_to_fast_us" || -z "$bf16_to_fast_us" || -z "$fast_to_f32_us" ]]; then
    echo "FAIL: could not parse ROCmFP4 Vulkan CPY timings" >&2
    exit 1
fi

check_max() {
    local label="$1"
    local got="$2"
    local max="$3"

    awk -v label="$label" -v got="$got" -v max="$max" 'BEGIN {
        if (got + 0 > max + 0) {
            printf("FAIL: %s %.2f us/run exceeds max %.2f\n", label, got, max) > "/dev/stderr";
            exit 1;
        }
        printf("PASS: %s %.2f us/run <= %.2f\n", label, got, max);
    }'
}

check_max "F32->Q4_0_ROCMFP4 Vulkan CPY" "$f32_to_q4_us" "$MAX_F32_TO_Q4_US"
check_max "F16->Q4_0_ROCMFP4 Vulkan CPY" "$f16_to_q4_us" "$MAX_F16_TO_Q4_US"
check_max "BF16->Q4_0_ROCMFP4 Vulkan CPY" "$bf16_to_q4_us" "$MAX_BF16_TO_Q4_US"
check_max "Q4_0_ROCMFP4->F32 Vulkan CPY" "$q4_to_f32_us" "$MAX_Q4_TO_F32_US"
check_max "F32->Q4_0_ROCMFP4_FAST Vulkan CPY" "$f32_to_fast_us" "$MAX_F32_TO_FAST_US"
check_max "F16->Q4_0_ROCMFP4_FAST Vulkan CPY" "$f16_to_fast_us" "$MAX_F16_TO_FAST_US"
check_max "BF16->Q4_0_ROCMFP4_FAST Vulkan CPY" "$bf16_to_fast_us" "$MAX_BF16_TO_FAST_US"
check_max "Q4_0_ROCMFP4_FAST->F32 Vulkan CPY" "$fast_to_f32_us" "$MAX_FAST_TO_F32_US"
