#!/usr/bin/env python3
"""Adaptive request wrapper for ROCmFPX dynamic drafting.

This client keeps llama-server simple: the server starts with a safe speculative
cap, and this wrapper injects per-request speculative settings based on prompt
length plus optional feedback from prior draft acceptance.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROFILE_PATH = SCRIPT_DIR / "rocmfpx-draft-profile.py"
PROFILE_SPEC = importlib.util.spec_from_file_location("rocmfpx_draft_profile", PROFILE_PATH)
if PROFILE_SPEC is None or PROFILE_SPEC.loader is None:
    raise RuntimeError(f"failed to load {PROFILE_PATH}")
rocmfpx_draft_profile = importlib.util.module_from_spec(PROFILE_SPEC)
PROFILE_SPEC.loader.exec_module(rocmfpx_draft_profile)

choose_profile = rocmfpx_draft_profile.choose_profile
tokenize_count = rocmfpx_draft_profile.tokenize_count


DEFAULT_STATE = {
    "requests": 0,
    "draft_n": 0,
    "draft_n_accepted": 0,
    "acceptance_ema": None,
    "throughput_ema": None,
    "last_n_max": None,
    "last_p_min": None,
    "n_max_stats": {},
    "last_update": None,
}


def load_json_arg(value: str | None, path: str | None) -> dict[str, Any]:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    if value:
        return json.loads(value)
    return {}


def load_state(path: str | None) -> dict[str, Any]:
    if not path:
        return copy.deepcopy(DEFAULT_STATE)
    state_path = Path(path)
    if not state_path.exists():
        return copy.deepcopy(DEFAULT_STATE)
    data = json.loads(state_path.read_text(encoding="utf-8"))
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(data)
    return state


def save_state(path: str | None, state: dict[str, Any]) -> None:
    if not path:
        return
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_text(payload: dict[str, Any], endpoint: str) -> str:
    if endpoint.endswith("/v1/chat/completions") or "messages" in payload:
        parts: list[str] = []
        for message in payload.get("messages", []):
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(payload.get("prompt", ""))


def infer_prompt_tokens(args: argparse.Namespace, payload: dict[str, Any]) -> int:
    if args.prompt_tokens is not None:
        return args.prompt_tokens
    text = extract_text(payload, args.endpoint)
    if not text:
        return 0
    if args.no_tokenize:
        return max(1, len(text) // 4)
    return tokenize_count(args.base_url, text, args.api_key)


def adapt_policy(policy: dict[str, Any], state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = dict(policy)
    base_n_max = int(result.get("speculative.n_max", 0))
    n_max = base_n_max
    if n_max <= 0:
        return result

    # Prefer the fastest nearby n_max seen for this workload, then apply the
    # global acceptance guard below. This keeps the policy adaptive without
    # letting one outlier request swing all future calls.
    stats = state.get("n_max_stats")
    if isinstance(stats, dict):
        best_n_max = n_max
        best_tps = -1.0
        for key, value in stats.items():
            if not isinstance(value, dict):
                continue
            try:
                candidate_n_max = int(key)
            except ValueError:
                continue
            if candidate_n_max < args.min_n_max or candidate_n_max > args.max_n_max:
                continue
            if abs(candidate_n_max - base_n_max) > args.max_profile_shift:
                continue
            tps = value.get("throughput_ema")
            acceptance = value.get("acceptance_ema")
            count = value.get("count", 0)
            if (
                isinstance(tps, (int, float))
                and isinstance(acceptance, (int, float))
                and isinstance(count, int)
                and count >= args.min_stats_count
                and float(acceptance) >= args.low_acceptance
                and float(tps) > best_tps
            ):
                best_n_max = candidate_n_max
                best_tps = float(tps)
        n_max = best_n_max

    acceptance = state.get("acceptance_ema")
    if not isinstance(acceptance, (int, float)):
        result["speculative.n_max"] = n_max
        result["speculative.n_min"] = min(int(result.get("speculative.n_min", 0)), n_max)
        return result

    if acceptance < args.low_acceptance:
        n_max = max(args.min_n_max, n_max - 1)
        result["speculative.p_min"] = min(1.0, max(float(result.get("speculative.p_min", 0.0)), 0.25))
        result["speculative.p_split"] = min(float(result.get("speculative.p_split", 0.10)), 0.10)
    elif acceptance > args.high_acceptance:
        n_max = min(args.max_n_max, n_max + 1)
        result["speculative.p_min"] = max(0.0, min(float(result.get("speculative.p_min", 0.0)), 0.25))

    result["speculative.n_max"] = n_max
    result["speculative.n_min"] = min(int(result.get("speculative.n_min", 0)), n_max)
    return result


def find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_key(value, key)
            if found is not None:
                return found
    return None


def ema(old: Any, sample: float, alpha: float) -> float:
    if isinstance(old, (int, float)):
        return alpha * sample + (1.0 - alpha) * float(old)
    return sample


def update_state_from_response(state: dict[str, Any], response: dict[str, Any], alpha: float) -> dict[str, Any]:
    draft_n = find_key(response, "draft_n")
    draft_n_accepted = find_key(response, "draft_n_accepted")
    if not isinstance(draft_n, (int, float)) or not isinstance(draft_n_accepted, (int, float)) or draft_n <= 0:
        return state

    accepted = max(0.0, min(float(draft_n_accepted), float(draft_n)))
    rate = accepted / float(draft_n)
    rate = ema(state.get("acceptance_ema"), rate, alpha)

    throughput = find_key(response, "predicted_per_second")
    if isinstance(throughput, (int, float)):
        state["throughput_ema"] = ema(state.get("throughput_ema"), float(throughput), alpha)

    n_max = find_key(response, "speculative.n_max")
    p_min = find_key(response, "speculative.p_min")
    if isinstance(n_max, (int, float)):
        n_max_int = int(n_max)
        state["last_n_max"] = n_max_int
        stats = state.get("n_max_stats")
        if not isinstance(stats, dict):
            stats = {}
        bucket = stats.get(str(n_max_int))
        if not isinstance(bucket, dict):
            bucket = {"count": 0}
        bucket["count"] = int(bucket.get("count", 0)) + 1
        bucket["acceptance_ema"] = ema(bucket.get("acceptance_ema"), accepted / float(draft_n), alpha)
        if isinstance(throughput, (int, float)):
            bucket["throughput_ema"] = ema(bucket.get("throughput_ema"), float(throughput), alpha)
        stats[str(n_max_int)] = bucket
        state["n_max_stats"] = stats
    if isinstance(p_min, (int, float)):
        state["last_p_min"] = float(p_min)

    state["requests"] = int(state.get("requests", 0)) + 1
    state["draft_n"] = int(state.get("draft_n", 0)) + int(draft_n)
    state["draft_n_accepted"] = int(state.get("draft_n_accepted", 0)) + int(draft_n_accepted)
    state["acceptance_ema"] = rate
    state["last_update"] = int(time.time())
    return state


THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


def strip_thinking_text(text: str) -> str:
    text = THINK_RE.sub("", text)
    return text.lstrip()


def strip_thinking_response(obj: Any) -> Any:
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for key, value in obj.items():
            if key in {"reasoning", "reasoning_content"}:
                continue
            if key in {"content", "text"} and isinstance(value, str):
                cleaned[key] = strip_thinking_text(value)
            else:
                cleaned[key] = strip_thinking_response(value)
        return cleaned
    if isinstance(obj, list):
        return [strip_thinking_response(value) for value in obj]
    return obj


def send_request(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(
        args.base_url.rstrip("/") + args.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {body}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a request with ROCmFPX Dynamic Drafting fields.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18180", help="llama-server base URL")
    parser.add_argument("--endpoint", default="/completion", help="API endpoint, e.g. /completion or /v1/chat/completions")
    parser.add_argument("--api-key", help="Bearer token for llama-server")
    parser.add_argument("--json", help="Request payload JSON")
    parser.add_argument("--json-file", help="Request payload JSON file")
    parser.add_argument("--prompt-tokens", type=int, help="Known prompt token count")
    parser.add_argument("--profile", default="fp3-mtp", choices=("fp3-mtp", "fp4-general", "dense-coder"))
    parser.add_argument("--state-file", help="Persist draft acceptance feedback here")
    parser.add_argument("--no-tokenize", action="store_true", help="Estimate token count instead of calling /tokenize")
    parser.add_argument("--dry-run", action="store_true", help="Print adjusted payload and do not send")
    parser.add_argument("--strip-thinking", action=argparse.BooleanOptionalAction, default=True,
                        help="Strip <think>...</think> blocks and reasoning fields from responses")
    parser.add_argument("--chat-reasoning-format", default="deepseek",
                        choices=("none", "auto", "deepseek", "deepseek-legacy"),
                        help="Reasoning parser format to request for OpenAI-compatible chat calls")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--min-n-max", type=int, default=1)
    parser.add_argument("--max-n-max", type=int, default=8)
    parser.add_argument("--max-profile-shift", type=int, default=2)
    parser.add_argument("--min-stats-count", type=int, default=1)
    parser.add_argument("--low-acceptance", type=float, default=0.45)
    parser.add_argument("--high-acceptance", type=float, default=0.80)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    args = parser.parse_args()

    payload = load_json_arg(args.json, args.json_file)
    state = load_state(args.state_file)
    prompt_tokens = infer_prompt_tokens(args, payload)
    policy = choose_profile(prompt_tokens, args.profile)
    policy = adapt_policy(policy, state, args)

    adjusted = dict(payload)
    adjusted.update(policy)
    if args.endpoint.endswith("/v1/chat/completions") or "messages" in adjusted:
        adjusted.setdefault("reasoning_format", args.chat_reasoning_format)

    if args.dry_run:
        result: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "profile": args.profile,
            "state": state,
            "request": adjusted,
        }
    else:
        result = send_request(args, adjusted)
        state = update_state_from_response(state, result, args.ema_alpha)
        save_state(args.state_file, state)
        if args.strip_thinking:
            result = strip_thinking_response(result)

    if args.pretty:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
