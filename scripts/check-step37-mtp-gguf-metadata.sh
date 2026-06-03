#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MODEL="${1:-/mnt/ai-models/rocmfp4-quants/Step-3.7-flash-BF16-MTP.gguf}"

if [[ ! -s "$MODEL" ]]; then
    echo "Missing StepFun GGUF: $MODEL" >&2
    exit 1
fi

PYTHONPATH="$ROOT/gguf-py${PYTHONPATH:+:$PYTHONPATH}" python3 - "$MODEL" <<'PY'
import sys

import gguf

model = sys.argv[1]
reader = gguf.GGUFReader(model)


def scalar(key: str) -> int:
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f"missing required GGUF metadata: {key}")
    return int(field.parts[-1][0])


def text(key: str) -> str:
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f"missing required GGUF metadata: {key}")
    return bytes(field.parts[-1]).decode("utf-8")


expected = {
    "step35.block_count": 48,
    "step35.nextn_predict_layers": 3,
    "tokenizer.ggml.bos_token_id": 0,
    "tokenizer.ggml.eos_token_id": 128007,
    "tokenizer.ggml.padding_token_id": 1,
}

for key, value in expected.items():
    actual = scalar(key)
    if actual != value:
        raise ValueError(f"{key}: expected {value}, got {actual}")
    print(f"{key} = {actual}")

tokenizer_pre = text("tokenizer.ggml.pre")
if tokenizer_pre not in {"deepseek-v3", "step35"}:
    raise ValueError(
        "tokenizer.ggml.pre: expected deepseek-v3 or the early experimental "
        f"step35 alias, got {tokenizer_pre}"
    )
print(f"tokenizer.ggml.pre = {tokenizer_pre}")
if tokenizer_pre == "step35":
    print("warning: using early experimental step35 tokenizer alias")

tokens = reader.get_field("tokenizer.ggml.tokens")
if tokens is None:
    raise ValueError("missing required GGUF metadata: tokenizer.ggml.tokens")
token_count = len(tokens.data)
if token_count != 128896:
    raise ValueError(f"tokenizer.ggml.tokens: expected 128896 entries, got {token_count}")
print(f"tokenizer.ggml.tokens = {token_count} entries")
print(f"StepFun MTP GGUF metadata passed: {model}")
PY
