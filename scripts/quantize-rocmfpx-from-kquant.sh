#!/usr/bin/env bash
# Requantize an existing K-quant / Q8 GGUF into ROCmFPX presets without BF16 source.
#
# Quality ladder (best -> worst source for ROCmFPX Q3 LEAN):
#   BF16/F16 > Q8_0/Q8_K > Q6_K > Q4_K_M > Q3_K_M / prior ROCmFPX
#
# Q4_K_M is the practical minimum for Q3_0_ROCMFPX on small models; Q3_K_M and
# double-ROCmFPX chains (Q8_0_ROCMFPX -> Q3) often fail coherency probes.
#
# For ROCmFP4 (promoted branch), requantize with PRESET=Q4_0_ROCMFP4_FAST or
# Q4_0_ROCMFP4_COHERENT from Q4_K_M / Q6_K / Q8_0 — same ladder applies.
# Use Q6_0_ROCMFPX (not Q3) when the source is Q8-class and you need KV headroom.
#
# Usage:
#   SRC=/path/to/model-Q4_K_M.gguf OUT=/path/to/out.gguf PRESET=Q3_0_ROCMFPX \
#     scripts/quantize-rocmfpx-from-kquant.sh
#
# With importance matrix (recommended for Q3 from Q8):
#   IMATRIX=/path/to/imatrix.gguf SRC=... OUT=... PRESET=Q3_0_ROCMFPX \
#     scripts/quantize-rocmfpx-from-kquant.sh
#
# Optional post-quant coherency gate:
#   RUN_COHERENCY=1 SRC=... OUT=... scripts/quantize-rocmfpx-from-kquant.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
QUANTIZE_BIN="${QUANTIZE_BIN:-$BUILD_DIR/bin/llama-quantize}"
SRC="${SRC:-}"
OUT="${OUT:-}"
PRESET="${PRESET:-Q3_0_ROCMFPX}"
IMATRIX="${IMATRIX:-}"
ALLOW_REQUANTIZE="${ALLOW_REQUANTIZE:-1}"
RUN_COHERENCY="${RUN_COHERENCY:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<EOF
Usage: $(basename "$0")

Required:
  SRC   Input GGUF (Q4_K_M, Q6_K, Q8_0, Q8_0_ROCMFPX, etc.)
  OUT   Output GGUF path

Optional:
  PRESET=Q3_0_ROCMFPX | Q6_0_ROCMFPX | Q8_0_ROCMFPX | *_AGENT variants
  IMATRIX=path         Importance matrix for better Q3/Q6 from Q8 sources
  RUN_COHERENCY=1      Run check-rocmfpx-qwen-coherency.sh after quantize
  DRY_RUN=1            Print planned command only

Examples:
  SRC=model-Q4_K_M.gguf OUT=model-Q3_0_ROCMFPX.gguf PRESET=Q3_0_ROCMFPX $0
  SRC=model-Q8_0.gguf IMATRIX=imatrix.gguf OUT=model-Q6_0_ROCMFPX.gguf PRESET=Q6_0_ROCMFPX $0
EOF
}

if [[ -z "$SRC" || -z "$OUT" ]]; then
    usage >&2
    exit 2
fi

if [[ ! -x "$QUANTIZE_BIN" ]]; then
    echo "missing llama-quantize: $QUANTIZE_BIN" >&2
    exit 1
fi

if [[ ! -f "$SRC" ]]; then
    echo "missing source: $SRC" >&2
    exit 1
fi

src_base="$(basename "$SRC")"
warn=""
if [[ "$src_base" == *[Rr][Oo][Cc][Mm][Ff][Pp][Xx]* && "$PRESET" == *ROCMFPX* ]]; then
    warn="source is already ROCmFPX; double requant often fails coherency — prefer Q4_K_M/Q6_K/Q8_0 stock quants"
elif [[ "$src_base" == *Q3_K* ]]; then
    warn="Q3_K source is below practical floor for Q3_0_ROCMFPX; expect coherency failures"
fi

recommended_preset() {
    local src="$1"
    local want="$2"
    if [[ "$src" == *Q8* || "$src" == *q8* ]]; then
        if [[ "$want" == Q3_0_ROCMFPX* ]]; then
            echo "Q6_0_ROCMFPX"
            return
        fi
    fi
    echo "$want"
}

if [[ -z "${PRESET_FORCE:-}" ]]; then
    rec="$(recommended_preset "$src_base" "$PRESET")"
    if [[ "$rec" != "$PRESET" ]]; then
        echo "NOTE: for Q8-class source targeting Q3, recommended preset is $rec (use PRESET_FORCE=$PRESET to override)"
        PRESET="$rec"
    fi
fi

quant_args=()
if [[ "$ALLOW_REQUANTIZE" == "1" ]]; then
    quant_args+=(--allow-requantize)
fi
if [[ -n "$IMATRIX" ]]; then
    if [[ ! -f "$IMATRIX" ]]; then
        echo "missing IMATRIX: $IMATRIX" >&2
        exit 1
    fi
    quant_args+=(--imatrix "$IMATRIX")
fi

mkdir -p "$(dirname "$OUT")"

echo "Source:  $SRC"
echo "Output:  $OUT"
echo "Preset:  $PRESET"
[[ -n "$warn" ]] && echo "WARN:    $warn"
[[ -n "$IMATRIX" ]] && echo "Imatrix: $IMATRIX"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY RUN: $QUANTIZE_BIN ${quant_args[*]} \"$SRC\" \"$OUT\" $PRESET"
    exit 0
fi

"$QUANTIZE_BIN" "${quant_args[@]}" "$SRC" "$OUT" "$PRESET"

python3 - "$SRC" "$OUT" "$PRESET" "$warn" <<'PY'
import json, os, sys
src, out, preset, warn = sys.argv[1:5]
print(json.dumps({
    "status": "pass",
    "source": src,
    "output": out,
    "preset": preset,
    "warn": warn or None,
    "allow_requantize": os.environ.get("ALLOW_REQUANTIZE", "1") == "1",
    "imatrix": os.environ.get("IMATRIX") or None,
}, indent=2))
PY

if [[ "$RUN_COHERENCY" == "1" ]]; then
    echo "Running coherency gate on $OUT ..."
    MODEL="$OUT" "$SCRIPT_DIR/check-rocmfpx-qwen-coherency.sh"
fi