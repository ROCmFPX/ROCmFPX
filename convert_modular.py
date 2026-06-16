#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper for the modular converter path. "
            "Forwards to convert_hf_to_gguf.py so model-specific options stay in one place."
        ),
    )
    parser.add_argument("model_dir", type=Path, help="directory containing the Hugging Face model")
    parser.add_argument("output_gguf", type=Path, help="GGUF file to write")
    parser.add_argument("outtype", nargs="?", default="bf16", help="output type, e.g. auto, f32, f16, bf16, q8_0")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="additional convert_hf_to_gguf.py arguments; prefix with -- to separate them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    sys.path.insert(0, str(ROOT))
    sys.path.insert(1, str(ROOT / "gguf-py"))

    from convert_hf_to_gguf import main as convert_main

    sys.argv = [
        "convert_hf_to_gguf.py",
        "--outfile",
        str(args.output_gguf),
        "--outtype",
        args.outtype,
        *extra_args,
        str(args.model_dir),
    ]
    convert_main()


if __name__ == "__main__":
    main()
