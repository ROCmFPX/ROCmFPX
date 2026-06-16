#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SKIP_MISSING_MODEL="${SKIP_MISSING_MODEL:-1}"

cd "$ROOT"

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

run_one() {
    local name="$1"
    local script="$2"
    local status=skip
    if [[ ! -f "$script" ]]; then
        echo "$name skip missing-script" >>"$tmp"
        return 0
    fi
    local out
    if out="$(SKIP_MISSING_MODEL="$SKIP_MISSING_MODEL" bash "$script" 2>&1)"; then
        if [[ "$out" == *"SKIP:"* || "$out" == *'"status": "skip'* ]]; then
            status=skip
        else
            status=pass
        fi
    else
        if [[ "$out" == *"SKIP:"* || "$out" == *'"status": "skip'* ]]; then
            status=skip
        else
            status=fail
        fi
    fi
    echo "$name $status" >>"$tmp"
}

overall=pass
run_one mtp "$ROOT/scripts/check-rocmfpx-mtp-smoke.sh"
run_one eagle3 "$ROOT/scripts/check-rocmfpx-eagle3-smoke.sh"
run_one speculative "$ROOT/scripts/check-rocmfpx-speculative-smoke.sh"

while read -r name status; do
    if [[ "$status" == fail ]]; then
        overall=fail
    fi
done <"$tmp"

python3 - "$tmp" "$overall" <<'PY'
import json, pathlib, sys
path, overall = sys.argv[1], sys.argv[2]
results = []
for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
    name, status = line.split()
    results.append({"name": name, "status": status})
print(json.dumps({"status": overall, "results": results}, indent=2, sort_keys=True))
raise SystemExit(0 if overall == "pass" else 1)
PY
