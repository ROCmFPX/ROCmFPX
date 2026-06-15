#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BIN="${BIN:-$ROOT/build-strix-rocmfp4/bin/llama-bench}"
MODEL="${MODEL:-/home/caf/strix-fp4/models/rocmfpx-bf16-tests/Qwen3-0.6B-Q3_0_ROCMFPX_COHERENT-LEAN.gguf}"
BACKEND="${BACKEND:-ROCm0}"
MIN_TG_TPS="${MIN_TG_TPS:-50}"

cd "$ROOT"

tmp_out="$(mktemp)"
trap 'rm -f "$tmp_out"' EXIT

timeout --kill-after=20s 3m "$BIN" \
    -m "$MODEL" \
    -dev "$BACKEND" \
    -ngl 99 \
    -p 16 \
    -n 16 \
    -r 3 >"$tmp_out" 2>&1

cat "$tmp_out"

python3 - "$MIN_TG_TPS" "$tmp_out" <<'PY'
import re
import sys

min_tg = float(sys.argv[1])
with open(sys.argv[2], "r", encoding="utf-8") as f:
    text = f.read()

pp = re.search(r"pp\d+.*?\|\s*([\d.]+)\s+±", text)
tg = re.search(r"tg\d+.*?\|\s*([\d.]+)\s+±", text)
if not pp or not tg:
    raise SystemExit("bench parse failed: could not find pp/tg throughput")

pp_tps = float(pp.group(1))
tg_tps = float(tg.group(1))
print(f"parsed pp16={pp_tps:.2f} t/s tg16={tg_tps:.2f} t/s")

if tg_tps < min_tg:
    raise SystemExit(f"decode speed too low: tg16={tg_tps:.2f} < min={min_tg}")
PY

echo "Bench gate passed for ${MODEL} on ${BACKEND}"