#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SMOKE="${SMOKE:-$SCRIPT_DIR/run-step37-mtp-rocmfp4-smoke.sh}"
HANDOFF_DIR="${HANDOFF_DIR:-/home/caf/strix-fp4/ROCMFP4-HANDOFF}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-$HANDOFF_DIR/results/step37-context-ladder-$RUN_ID}"
CTX_SIZES="${CTX_SIZES:-2048 4096 8192 16384 32768 65536 131072 262144}"
NATIVE_CTX="${NATIVE_CTX:-262144}"
BACKEND="${BACKEND:-Vulkan0}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-2}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q4_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q4_0}"
SPEC_DRAFT_TYPE_K="${SPEC_DRAFT_TYPE_K:-q4_0}"
SPEC_DRAFT_TYPE_V="${SPEC_DRAFT_TYPE_V:-q4_0}"
MIN_AVAILABLE_GIB="${MIN_AVAILABLE_GIB:-116}"
MAX_SWAP_USED_MIB="${MAX_SWAP_USED_MIB:-64}"

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

available_kib="$(mem_available_kib)"
swap_kib="$(swap_used_kib)"

if (( swap_kib > MAX_SWAP_USED_MIB * 1024 )); then
    echo "Refusing to start: swap still has $((swap_kib / 1024)) MiB in use." >&2
    echo "Run: sudo swapoff -a && sudo swapon -a" >&2
    exit 1
fi

if (( available_kib < MIN_AVAILABLE_GIB * 1024 * 1024 )); then
    echo "Refusing to start: only $((available_kib / 1024 / 1024)) GiB is available." >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
summary="$RUN_DIR/summary.tsv"
printf "context\tbackend\tmain_kv\tdraft_kv\tresult\tprompt_tps\tgeneration_tps\tlog\n" > "$summary"

echo "StepFun 3.7 Flash native-MTP context allocation ladder"
echo "Logs: $RUN_DIR"

for ctx in $CTX_SIZES; do
    if (( ctx > NATIVE_CTX )); then
        echo "Skipping context $ctx: above native context $NATIVE_CTX."
        continue
    fi

    available_kib="$(mem_available_kib)"
    if (( available_kib < MIN_AVAILABLE_GIB * 1024 * 1024 )); then
        echo "Stopping before context $ctx: only $((available_kib / 1024 / 1024)) GiB is available." >&2
        exit 1
    fi

    log="$RUN_DIR/ctx-$ctx.log"
    echo
    echo "Testing context $ctx on $BACKEND ..."

    set +e
    BACKEND="$BACKEND" \
    CTX_SIZE="$ctx" \
    SPEC_DRAFT_N_MAX="$SPEC_DRAFT_N_MAX" \
    CACHE_TYPE_K="$CACHE_TYPE_K" \
    CACHE_TYPE_V="$CACHE_TYPE_V" \
    SPEC_DRAFT_TYPE_K="$SPEC_DRAFT_TYPE_K" \
    SPEC_DRAFT_TYPE_V="$SPEC_DRAFT_TYPE_V" \
    "$SMOKE" 2>&1 | tee "$log"
    rc="${PIPESTATUS[0]}"
    set -e

    if (( rc != 0 )); then
        printf "%s\t%s\t%s/%s\t%s/%s\tFAIL\t\t\t%s\n" \
            "$ctx" "$BACKEND" "$CACHE_TYPE_K" "$CACHE_TYPE_V" \
            "$SPEC_DRAFT_TYPE_K" "$SPEC_DRAFT_TYPE_V" "$log" >> "$summary"
        echo "Stopping at context $ctx after a guarded failure. See $log" >&2
        exit "$rc"
    fi

    prompt_tps="$(sed -n 's|.*Prompt: \([0-9.]*\) t/s.*|\1|p' "$log" | tail -n 1)"
    generation_tps="$(sed -n 's|.*Generation: \([0-9.]*\) t/s.*|\1|p' "$log" | tail -n 1)"
    printf "%s\t%s\t%s/%s\t%s/%s\tPASS\t%s\t%s\t%s\n" \
        "$ctx" "$BACKEND" "$CACHE_TYPE_K" "$CACHE_TYPE_V" \
        "$SPEC_DRAFT_TYPE_K" "$SPEC_DRAFT_TYPE_V" \
        "$prompt_tps" "$generation_tps" "$log" >> "$summary"
done

echo
echo "Completed StepFun context allocation ladder:"
column -t -s $'\t' "$summary"
