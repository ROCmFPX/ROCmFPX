#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BIN="${BIN:-$ROOT/build-strix-rocmfp4/bin/llama-quantize}"
MODEL_SRC="${MODEL_SRC:-}"
SKIP_MISSING_MODEL="${SKIP_MISSING_MODEL:-1}"
OUT_DIR="${OUT_DIR:-/tmp/rocmfpx-moe-routing}"

cd "$ROOT"

if [[ ! -x "$BIN" ]]; then
    echo "missing llama-quantize binary: $BIN" >&2
    exit 1
fi

if [[ -z "$MODEL_SRC" || ! -f "$MODEL_SRC" ]]; then
    if [[ "$SKIP_MISSING_MODEL" == "1" ]]; then
        python3 - <<PY
import json
print(json.dumps({"status": "skip", "reason": "set MODEL_SRC to a MiniMax/Mixtral/MoE BF16 or F16 GGUF fixture"}))
PY
        exit 0
    fi
    echo "missing MODEL_SRC MoE fixture" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
tmp_json="$OUT_DIR/moe-routing-results.json"

python3 - "$tmp_json" <<'PY'
import json, sys
path = sys.argv[1]
path and open(path, "w", encoding="utf-8").write(json.dumps({"status": "running", "results": []}))
PY

run_dry() {
    local preset="$1"
    local log="$OUT_DIR/${preset}.log"
    "$BIN" --dry-run "$MODEL_SRC" "$OUT_DIR/${preset}.gguf" "$preset" >"$log" 2>&1
}

for preset in Q3_0_ROCMFPX Q3_0_ROCMFPX_AGENT Q6_0_ROCMFPX Q6_0_ROCMFPX_AGENT Q8_0_ROCMFPX Q8_0_ROCMFPX_AGENT; do
    run_dry "$preset"
done

python3 - "$OUT_DIR" "$MODEL_SRC" <<'PY'
import json
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
src = sys.argv[2]
results = []
failed = False
for log in sorted(out.glob("Q*_ROCMFPX*.log")):
    text = log.read_text(encoding="utf-8", errors="replace")
    size = re.search(r"quant size\s+=\s+([0-9.]+ MiB) \(([0-9.]+ BPW)\)", text)
    has_expert = bool(re.search(r"ffn_(gate|up|down)_exps|expert|n_expert", text, re.I))
    ok = "error" not in text.lower() and "failed" not in text.lower()
    failed = failed or not ok
    results.append({
        "preset": log.stem,
        "ok": ok,
        "size": size.group(1) if size else None,
        "bpw": size.group(2) if size else None,
        "expert_evidence": has_expert,
    })

print(json.dumps({"status": "fail" if failed else "pass", "source": src, "results": results}, indent=2, sort_keys=True))
raise SystemExit(1 if failed else 0)
PY
