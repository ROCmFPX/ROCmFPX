#!/usr/bin/env bash
# Drive the focused test-backend-ops gates that cover ROCmFPX tensor types.
#
# Splits CPU / ROCm / Vulkan into independent runs so a Vulkan timeout does
# not mask a CPU pass. Emits a single JSON report.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
TEST_BIN="${TEST_BIN:-$BUILD_DIR/bin/test-backend-ops}"
OPS="${OPS:-MUL_MAT,GET_ROWS,CPY,SET_ROWS}"
BACKENDS="${BACKENDS:-CPU ROCm0 Vulkan0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-240}"
OUT_DIR="${OUT_DIR:-/tmp/rocmfpx-backend-ops}"
SKIP_VULKAN="${SKIP_VULKAN:-0}"

cd "$ROOT"

if [[ ! -x "$TEST_BIN" ]]; then
    echo "missing test-backend-ops: $TEST_BIN" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

run_backend() {
    local backend="$1"
    local log="$OUT_DIR/${backend}.log"
    local timeout_s="$TIMEOUT_SEC"
    if [[ "$backend" == CPU ]]; then
        timeout_s=120
    fi
    echo "=== backend-ops: $backend ===" | tee "$log"
    if timeout --kill-after=20s "${timeout_s}s" "$TEST_BIN" test -o "$OPS" -b "$backend" \
            >>"$log" 2>&1; then
        echo "{\"backend\":\"$backend\",\"ok\":true}" >"$OUT_DIR/${backend}.json"
    else
        local code=$?
        echo "{\"backend\":\"$backend\",\"ok\":false,\"exit\":$code}" >"$OUT_DIR/${backend}.json"
    fi
}

if [[ "$SKIP_VULKAN" == "1" ]]; then
    BACKENDS="${BACKENDS//Vulkan0/}"
fi

for backend in $BACKENDS; do
    run_backend "$backend"
done

python3 - "$OUT_DIR" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
results = []
overall = "pass"
for path in sorted(out.glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    results.append(data)
    if not data["ok"]:
        overall = "fail"
print(json.dumps({"status": overall, "results": results}, indent=2, sort_keys=True))
raise SystemExit(0 if overall == "pass" else 1)
PY
