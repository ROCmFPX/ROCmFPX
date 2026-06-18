#!/usr/bin/env bash
# Regression gate: fp3 KV cache (-ctk/-ctv q3_0_rocmfpx) must not produce garbage.
# Requires flash-attn on (quantized V cache). fp3 K routes to FA tile kernel (vec KQ drifts).
#
# Re-run after changes to ggml/rocmfpx/, set-rows.cu, cpy-utils.cuh, or fattn-*.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
HERMES_MODEL="${HERMES_MODEL:-$ROOT/../models/rocmfpx-hermes-tests/hermes-Q3_0_ROCMFPX_AGENT.gguf}"
MODEL_COHERENT="${MODEL_COHERENT:-$ROOT/../models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-LEAN.gguf}"
BACKEND="${BACKEND:-ROCm0}"
SKIP_MISSING_MODEL="${SKIP_MISSING_MODEL:-1}"

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
export GGML_HIP_ENABLE_UNIFIED_MEMORY="${GGML_HIP_ENABLE_UNIFIED_MEMORY:-1}"

cd "$ROOT"

if [[ ! -f "$HERMES_MODEL" ]]; then
    if [[ "$SKIP_MISSING_MODEL" == "1" ]]; then
        python3 - <<PY
import json
print(json.dumps({"status": "skip", "reason": "missing HERMES_MODEL", "model": "$HERMES_MODEL"}))
PY
        exit 0
    fi
    echo "missing HERMES_MODEL: $HERMES_MODEL" >&2
    exit 1
fi

failures=0

run_case() {
    local label="$1"
    local extra_args="$2"
    local script="$3"
    local model="$4"
    shift 4
    local extra_env=("$@")

    echo "=== fp3 KV case: $label ==="
    if ! env "${extra_env[@]}" MODEL="$model" BACKEND="$BACKEND" LLAMA_COMPLETION_ARGS="$extra_args" bash "$script"; then
        echo "FAIL: $label" >&2
        failures=$((failures + 1))
    fi
}

run_case "hermes-fp3kv-fa" "-ctk q3_0_rocmfpx -ctv q3_0_rocmfpx --flash-attn on" \
    "$SCRIPT_DIR/check-rocmfpx-hermes-smoke.sh" "$HERMES_MODEL"

if [[ -f "$MODEL_COHERENT" ]]; then
    run_case "qwen-coherency-fp3kv" "-ctk q3_0_rocmfpx -ctv q3_0_rocmfpx --flash-attn on" \
        "$SCRIPT_DIR/check-rocmfpx-qwen-coherency.sh" "$MODEL_COHERENT" \
        COHERENCY_PROBES="coding,summary"
fi

if [[ -x "$ROOT/build-strix-rocmfp4/bin/test-backend-ops" ]]; then
    echo "=== fp3 KV backend-ops SET_ROWS subset ==="
    setrows_out="$(mktemp)"
    if ! "$ROOT/build-strix-rocmfp4/bin/test-backend-ops" test -b "$BACKEND" -o SET_ROWS >"$setrows_out" 2>&1; then
        echo "FAIL: test-backend-ops SET_ROWS exited non-zero" >&2
        failures=$((failures + 1))
    elif ! rg -q "type=q3_0_rocmfpx.*OK" "$setrows_out"; then
        echo "FAIL: no passing q3_0_rocmfpx SET_ROWS cases on $BACKEND" >&2
        failures=$((failures + 1))
    fi
    rm -f "$setrows_out"

    echo "=== fp3 KV backend-ops FLASH_ATTN subset ==="
    fa_out="$(mktemp)"
    if ! "$ROOT/build-strix-rocmfp4/bin/test-backend-ops" test -b "$BACKEND" -o FLASH_ATTN_EXT >"$fa_out" 2>&1; then
        echo "FAIL: test-backend-ops FLASH_ATTN_EXT exited non-zero" >&2
        failures=$((failures + 1))
    elif ! rg -q "type_K=q3_0_rocmfpx,type_V=q3_0_rocmfpx.*OK" "$fa_out"; then
        echo "FAIL: no passing q3_0_rocmfpx FLASH_ATTN cases on $BACKEND" >&2
        failures=$((failures + 1))
    fi
    rm -f "$fa_out"
fi

if [[ "$failures" -gt 0 ]]; then
    python3 - <<PY
import json
print(json.dumps({"status": "fail", "failures": $failures, "hermes_model": "$HERMES_MODEL"}))
PY
    exit 1
fi

python3 - <<PY
import json
print(json.dumps({
    "status": "pass",
    "hermes_model": "$HERMES_MODEL",
    "backend": "$BACKEND",
    "note": "fp3 KV cache regression passed (hermes FA + coherency coding/summary + backend ops)",
}))
PY