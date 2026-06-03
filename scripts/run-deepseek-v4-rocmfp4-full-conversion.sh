#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${1:-/mnt/ai-models/gguf-sources/DeepSeek-V4-Flash-180B}"
OUTPUT_DIR="${2:-/mnt/ai-models/rocmfp4-quants/DeepSeek-V4-Flash-180B}"
INCLUDE_MTP="${3:-0}"
ARTIFACT_SUFFIX=""
CONVERT_ARGS=()
if [[ "${INCLUDE_MTP}" == "1" ]]; then
    ARTIFACT_SUFFIX="-MTP"
    CONVERT_ARGS+=(--deepseek4-include-mtp --use-temp-file)
fi
CONVERSION_TMPDIR="${CONVERSION_TMPDIR:-${OUTPUT_DIR}/.tmp}"
INTERMEDIATE="${OUTPUT_DIR}/DeepSeek-V4-Flash-180B${ARTIFACT_SUFFIX}-MXFP4-F16.gguf"
QUANTIZED="${OUTPUT_DIR}/DeepSeek-V4-Flash-180B${ARTIFACT_SUFFIX}-MXFP4-to-ROCmFP4-STRIX_LEAN.gguf"
INTERMEDIATE_PARTIAL="${INTERMEDIATE}.partial"
QUANTIZED_PARTIAL="${QUANTIZED}.partial"
QUANTIZER="${ROOT_DIR}/build-deepseek4-rocmfp4/bin/llama-quantize"

mkdir -p "${OUTPUT_DIR}"
if [[ "${INCLUDE_MTP}" == "1" ]]; then
    mkdir -p "${CONVERSION_TMPDIR}"
fi
cd "${ROOT_DIR}"

if [[ ! -x "${QUANTIZER}" ]]; then
    printf 'missing quantizer: %s\n' "${QUANTIZER}" >&2
    printf 'build it with: cmake --build build-deepseek4-rocmfp4 --target llama-quantize -j 12\n' >&2
    exit 1
fi

for path in "${INTERMEDIATE}" "${QUANTIZED}" "${INTERMEDIATE_PARTIAL}" "${QUANTIZED_PARTIAL}"; do
    if [[ -e "${path}" ]]; then
        printf 'refusing to overwrite existing artifact: %s\n' "${path}" >&2
        exit 1
    fi
done

TMPDIR="${CONVERSION_TMPDIR}" python3 convert_hf_to_gguf.py \
    "${SOURCE_DIR}" \
    --outtype f16 \
    "${CONVERT_ARGS[@]}" \
    --outfile "${INTERMEDIATE_PARTIAL}"
mv "${INTERMEDIATE_PARTIAL}" "${INTERMEDIATE}"

"${QUANTIZER}" --dry-run "${INTERMEDIATE}" Q4_0_ROCMFP4
"${QUANTIZER}" "${INTERMEDIATE}" "${QUANTIZED_PARTIAL}" Q4_0_ROCMFP4

PYTHONPATH="${ROOT_DIR}/gguf-py" python3 - "${QUANTIZED_PARTIAL}" "${INCLUDE_MTP}" <<'PY'
from collections import Counter
from pathlib import Path
import sys

from gguf import GGUFReader

path = Path(sys.argv[1])
include_mtp = sys.argv[2] == "1"
reader = GGUFReader(path)
tensors = {tensor.name: tensor for tensor in reader.tensors}

expected = {
    "blk.0.ffn_down_exps.weight": "MXFP4",
    "blk.0.ffn_gate_exps.weight": "MXFP4",
    "blk.0.ffn_up_exps.weight": "MXFP4",
    "blk.0.ffn_gate_tid2eid.weight": "I32",
    "blk.0.attn_kv.weight": "Q4_0_ROCMFP4",
    "blk.2.attn_compressor_kv.weight": "Q4_0_ROCMFP4",
    "blk.2.indexer_compressor_kv.weight": "Q4_0_ROCMFP4",
    "blk.3.attn_compressor_kv.weight": "Q4_0_ROCMFP4",
    "token_embd.weight": "Q4_0_ROCMFP4",
}
if include_mtp:
    expected.update({
        "blk.43.nextn.e_proj.weight": "Q4_0_ROCMFP4",
        "blk.43.nextn.h_proj.weight": "Q4_0_ROCMFP4",
        "blk.43.nextn.enorm.weight": "F32",
        "blk.43.nextn.hnorm.weight": "F32",
        "blk.43.nextn.shared_head_norm.weight": "F32",
        "blk.43.nextn.hc_head_base.weight": "F32",
        "blk.43.nextn.hc_head_fn.weight": "Q4_0_ROCMFP4",
        "blk.43.nextn.hc_head_scale.weight": "F32",
        "blk.43.ffn_down_exps.weight": "MXFP4",
        "blk.43.attn_kv.weight": "Q4_0_ROCMFP4",
    })

for name, tensor_type in expected.items():
    actual = tensors[name].tensor_type.name
    if actual != tensor_type:
        raise SystemExit(f"{name}: expected {tensor_type}, got {actual}")

counts = Counter(tensor.tensor_type.name for tensor in reader.tensors)
print(f"validated {path}")
print(f"tensor count: {len(reader.tensors)}")
print(f"tensor types: {dict(sorted(counts.items()))}")
PY

mv "${QUANTIZED_PARTIAL}" "${QUANTIZED}"
printf 'full DeepSeek V4 ROCmFP4 artifact: %s\n' "${QUANTIZED}"
