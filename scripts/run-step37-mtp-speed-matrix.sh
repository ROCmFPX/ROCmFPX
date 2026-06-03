#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE="${SMOKE:-$SCRIPT_DIR/run-step37-mtp-rocmfp4-smoke.sh}"
HANDOFF_DIR="${HANDOFF_DIR:-/home/caf/strix-fp4/ROCMFP4-HANDOFF}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-$HANDOFF_DIR/results/step37-speed-matrix-$RUN_ID}"
MATRIX_MODE="${MATRIX_MODE:-context}"
NATIVE_CTX="${NATIVE_CTX:-262144}"
MAX_SWAP_USED_MIB="${MAX_SWAP_USED_MIB:-64}"
MIN_AVAILABLE_GIB="${MIN_AVAILABLE_GIB:-116}"
N_PREDICT="${N_PREDICT:-64}"
PROMPT="${PROMPT:-List the integers from one through sixty-four in order, separated by spaces.}"
CONTEXT_KV="${CONTEXT_KV:-q4_0}"

mem_available_kib() {
    awk '/^MemAvailable:/ { print $2 }' /proc/meminfo
}

swap_used_kib() {
    awk '
        /^SwapTotal:/ { total = $2 }
        /^SwapFree:/  { free = $2 }
        END { print total - free }
    ' /proc/meminfo
}

run_case() {
    local backend="$1"
    local ctx="$2"
    local nmax="$3"
    local main_kv="$4"
    local draft_kv="$5"
    local backend_sampling="$6"
    local p_split="${7:-0.10}"
    local p_min="${8:-0.0}"
    local n_min="${9:-0}"
    local name="${backend,,}-ctx${ctx}-n${nmax}-main${main_kv}-draft${draft_kv}-bs${backend_sampling}-split${p_split}-pmin${p_min}-nmin${n_min}"
    local log="$RUN_DIR/$name.log"
    local available_kib
    local rc
    local prompt_tps
    local generation_tps
    local acceptance

    available_kib="$(mem_available_kib)"
    if (( available_kib < MIN_AVAILABLE_GIB * 1024 * 1024 )); then
        echo "Stopping before $name: only $((available_kib / 1024 / 1024)) GiB is available." >&2
        exit 1
    fi

    echo "Running $name ..."
    set +e
    BACKEND="$backend" \
    CTX_SIZE="$ctx" \
    SPEC_DRAFT_N_MAX="$nmax" \
    CACHE_TYPE_K="$main_kv" \
    CACHE_TYPE_V="$main_kv" \
    SPEC_DRAFT_TYPE_K="$draft_kv" \
    SPEC_DRAFT_TYPE_V="$draft_kv" \
    SPEC_DRAFT_BACKEND_SAMPLING="$backend_sampling" \
    SPEC_DRAFT_P_SPLIT="$p_split" \
    SPEC_DRAFT_P_MIN="$p_min" \
    SPEC_DRAFT_N_MIN="$n_min" \
    N_PREDICT="$N_PREDICT" \
    PROMPT="$PROMPT" \
    "$SMOKE" > "$log" 2>&1
    rc=$?
    set -e

    if (( rc != 0 )); then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tFAIL\t\t\t\t%s\n" \
            "$backend" "$ctx" "$nmax" "$main_kv" "$draft_kv" "$backend_sampling" \
            "$p_split" "$p_min" "$n_min" "$log" >> "$summary"
        echo "FAIL $name; continuing after guarded failure."
        return
    fi

    prompt_tps="$(sed -n 's|.*Prompt: \([0-9.]*\) t/s.*|\1|p' "$log" | tail -n 1)"
    generation_tps="$(sed -n 's|.*Generation: \([0-9.]*\) t/s.*|\1|p' "$log" | tail -n 1)"
    acceptance="$(sed -n 's|.*draft acceptance = \([0-9.]*\) ( *\([0-9]*\) accepted / *\([0-9]*\) generated).*|\1 (\2/\3)|p' "$log" | tail -n 1)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tPASS\t%s\t%s\t%s\t%s\n" \
        "$backend" "$ctx" "$nmax" "$main_kv" "$draft_kv" "$backend_sampling" \
        "$p_split" "$p_min" "$n_min" "$prompt_tps" "$generation_tps" "$acceptance" "$log" >> "$summary"
    echo "PASS $name: prompt=$prompt_tps t/s decode=$generation_tps t/s acceptance=$acceptance"
}

swap_kib="$(swap_used_kib)"
if (( swap_kib > MAX_SWAP_USED_MIB * 1024 )); then
    echo "Refusing to start: swap still has $((swap_kib / 1024)) MiB in use." >&2
    echo "Run: sudo swapoff -a && sudo swapon -a" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
summary="$RUN_DIR/summary.tsv"
printf "backend\tcontext\tn_max\tmain_kv\tdraft_kv\tbackend_sampling\tp_split\tp_min\tn_min\tresult\tprompt_tps\tgeneration_tps\tacceptance\tlog\n" > "$summary"

case "$MATRIX_MODE" in
    context)
        for backend in Vulkan0 ROCm0; do
            for ctx in 32768 65536 131072 "$NATIVE_CTX"; do
                run_case "$backend" "$ctx" 2 "$CONTEXT_KV" "$CONTEXT_KV" 1
            done
        done
        ;;
    native-tune)
        for backend in Vulkan0 ROCm0; do
            for nmax in 1 2 3; do
                for kv in q4_0 q8_0; do
                    run_case "$backend" "$NATIVE_CTX" "$nmax" "$kv" "$kv" 1
                done
            done
        done
        for nmax in 1 2 3; do
            for kv in q4_0 q8_0; do
                run_case ROCm0 "$NATIVE_CTX" "$nmax" "$kv" "$kv" 0
            done
        done
        ;;
    native-fine)
        for backend in Vulkan0 ROCm0; do
            run_case "$backend" "$NATIVE_CTX" 2 q8_0 q4_0 1
            run_case "$backend" "$NATIVE_CTX" 2 q4_0 q8_0 1
            run_case "$backend" "$NATIVE_CTX" 2 q8_0 q8_0 1 0.05
            run_case "$backend" "$NATIVE_CTX" 2 q8_0 q8_0 1 0.20
            run_case "$backend" "$NATIVE_CTX" 2 q8_0 q8_0 1 0.10 0.10
            run_case "$backend" "$NATIVE_CTX" 2 q8_0 q8_0 1 0.10 0.20
            run_case "$backend" "$NATIVE_CTX" 2 q8_0 q8_0 1 0.10 0.0 1
        done
        run_case Vulkan0 "$NATIVE_CTX" 2 q8_0 q8_0 0
        ;;
    *)
        echo "Unknown MATRIX_MODE: $MATRIX_MODE" >&2
        exit 1
        ;;
esac

echo
echo "Completed StepFun speed matrix:"
column -t -s $'\t' "$summary"
