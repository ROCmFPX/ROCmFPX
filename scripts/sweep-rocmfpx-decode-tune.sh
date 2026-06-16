#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_ROOT="${BUILD_ROOT:-$ROOT}"
BACKEND="${BACKEND:-ROCm0}"
MODEL_LEAN="${MODEL_LEAN:-/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-LEAN.gguf}"
MODEL_AGENT="${MODEL_AGENT:-}"
PROFILES="${PROFILES:-rocmfpx-strix-nwarps1 rocmfpx-strix-nwarps2 rocmfpx-strix-rpb2}"
SKIP_MISSING_MODEL="${SKIP_MISSING_MODEL:-1}"
OUT_DIR="${OUT_DIR:-/tmp/rocmfpx-decode-tune}"

cd "$ROOT"

if [[ ! -f "$MODEL_LEAN" ]]; then
    if [[ "$SKIP_MISSING_MODEL" == "1" ]]; then
        python3 - <<PY
import json
print(json.dumps({"status": "skip", "reason": "missing MODEL_LEAN", "model": "$MODEL_LEAN"}))
PY
        exit 0
    fi
    echo "missing MODEL_LEAN: $MODEL_LEAN" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
# shellcheck source=scripts/rocmfp4-decode-tune-flags.sh
source "$ROOT/scripts/rocmfp4-decode-tune-flags.sh"

bench_model() {
    local label="$1"
    local model="$2"
    local profile="$3"
    local build_dir="$OUT_DIR/build-${profile}-${label}"
    local bench_bin="$build_dir/bin/llama-bench"
    local log="$OUT_DIR/${profile}-${label}.bench.log"

    if [[ ! -x "$bench_bin" ]]; then
        ROCMFPX_DECODE_TUNE="$profile" BUILD_DIR="$build_dir" \
            "$ROOT/scripts/build-strix-rocmfp4-mtp.sh" llama-bench >/dev/null
    fi

    timeout --kill-after=20s 3m "$bench_bin" \
        -m "$model" -dev "$BACKEND" -ngl 99 -p 16 -n 16 -r 2 >"$log" 2>&1 || true
}

for profile in $PROFILES; do
    bench_model lean "$MODEL_LEAN" "$profile"
    if [[ -n "$MODEL_AGENT" && -f "$MODEL_AGENT" ]]; then
        bench_model agent "$MODEL_AGENT" "$profile"
    fi
done

python3 - "$OUT_DIR" "$MODEL_LEAN" "$MODEL_AGENT" <<'PY'
import json
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
lean = sys.argv[2]
agent = sys.argv[3]
results = []
for log in sorted(out.glob("*.bench.log")):
    text = log.read_text(encoding="utf-8", errors="replace")
    tg = re.search(r"tg\d+.*?\|\s*([\d.]+)\s+±", text)
    pp = re.search(r"pp\d+.*?\|\s*([\d.]+)\s+±", text)
    stem = log.stem
    profile, label = stem.rsplit("-", 1)
    results.append({
        "profile": profile,
        "label": label,
        "pp16_tps": float(pp.group(1)) if pp else None,
        "tg16_tps": float(tg.group(1)) if tg else None,
        "ok": tg is not None,
    })

print(json.dumps({
    "status": "pass",
    "model_lean": lean,
    "model_agent": agent or None,
    "results": results,
}, indent=2, sort_keys=True))
PY
