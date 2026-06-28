#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
FIXTURE_DIR="$ROOT/tests/fixtures/rocmfpx-ranked-policy"
TMP_DIR="${TMPDIR:-/tmp}/rocmfpx-ranked-policy-check.$$"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$TMP_DIR"

(
    cd "$ROOT"
    python3 scripts/rocmfpx-ranked-policy.py \
        --rank-csv tests/fixtures/rocmfpx-ranked-policy/attention-rank.sample.csv \
        --leave-count 2 \
        --output "$TMP_DIR/leave2.tensor-type.txt"
)

diff -u "$FIXTURE_DIR/leave2.tensor-type.expected.txt" "$TMP_DIR/leave2.tensor-type.txt"

echo "ROCmFPX ranked policy check passed"
