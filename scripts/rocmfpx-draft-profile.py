#!/usr/bin/env python3
"""Choose ROCmFPX request-level speculative settings from prompt length."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


SHORT_CONTEXT_LIMIT = 49152
PROFILES = ("fp3-mtp", "fp4-general", "dense-coder")


def tokenize_count(base_url: str, content: str, api_key: str | None) -> int:
    payload = json.dumps({
        "content": content,
        "add_special": False,
        "parse_special": True,
        "with_pieces": False,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/tokenize",
        data=payload,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.loads(response.read().decode("utf-8"))
    tokens = data.get("tokens")
    if not isinstance(tokens, list):
        raise SystemExit("/tokenize response did not include a tokens array")
    return len(tokens)


def choose_profile(prompt_tokens: int, profile: str) -> dict[str, float | int]:
    if profile == "dense-coder":
        return {
            "speculative.n_max": 6,
            "speculative.n_min": 0,
            "speculative.p_min": 0.0,
        }
    if profile == "fp4-general":
        return {
            "speculative.n_max": 4,
            "speculative.n_min": 0,
            "speculative.p_min": 0.0,
        }
    if profile == "fp3-mtp" and prompt_tokens < SHORT_CONTEXT_LIMIT:
        return {
            "speculative.n_max": 4,
            "speculative.n_min": 0,
            "speculative.p_min": 0.75,
        }
    return {
        "speculative.n_max": 2,
        "speculative.n_min": 0,
        "speculative.p_min": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit request JSON for the current ROCmFPX dynamic draft policy."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt-tokens", type=int, help="Known prompt token count")
    source.add_argument("--prompt-file", help="Prompt file to tokenize through llama-server")
    parser.add_argument("--base-url", help="llama-server URL, required with --prompt-file")
    parser.add_argument("--api-key", help="Bearer token for llama-server")
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default="fp3-mtp",
        help="Draft policy family to emit",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument("--with-token-count", action="store_true", help="Include prompt_tokens in the output JSON")
    args = parser.parse_args()

    if args.prompt_tokens is not None:
        prompt_tokens = args.prompt_tokens
    else:
        if not args.base_url:
            parser.error("--base-url is required with --prompt-file")
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        prompt_tokens = tokenize_count(args.base_url, prompt, args.api_key)

    if prompt_tokens < 0:
        parser.error("--prompt-tokens must be non-negative")

    result = choose_profile(prompt_tokens, args.profile)
    if args.with_token_count:
        result = {"prompt_tokens": prompt_tokens, "profile": args.profile, **result}

    if args.pretty:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
