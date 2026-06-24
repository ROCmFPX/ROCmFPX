#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPT_DIR/rocmfpx-lib.sh"
rocmfpx_setup_env

BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-rocmfp4}"
BIN="${BIN:-$BUILD_DIR/bin/llama-server}"
MODEL="${MODEL:-}"
DEVICE="${DEVICE:-Vulkan0}"
REQUIRE_MTP="${REQUIRE_MTP:-0}"
REQUIRE_PROFILE="${REQUIRE_PROFILE:-0}"
WRAPPER_OUT="${WRAPPER_OUT:-}"

if [[ -z "$MODEL" ]]; then
    echo "MODEL must point to a GGUF" >&2
    exit 2
fi

rocmfpx_require_binary "$BIN"
SKIP_MISSING_MODEL=0 rocmfpx_require_model "$MODEL"

caps_json="$("$SCRIPT_DIR/rocmfpx-model-capabilities.py" "$MODEL")"

python3 - "$caps_json" "$BIN" "$DEVICE" "$REQUIRE_MTP" "$REQUIRE_PROFILE" "$WRAPPER_OUT" <<'PY'
import json
import os
import pathlib
import shlex
import sys

caps = json.loads(sys.argv[1])
binary = pathlib.Path(sys.argv[2])
device = sys.argv[3]
require_mtp = sys.argv[4] == "1"
require_profile = sys.argv[5] == "1"
wrapper_out = sys.argv[6]
profile = caps["serving_profile"]

errors = []
warnings = []

if require_mtp and not caps["supports_mtp"]:
    errors.append("REQUIRE_MTP=1 but the model does not expose MTP metadata/tensors")

if require_profile and profile["name"].startswith("generic-"):
    errors.append("REQUIRE_PROFILE=1 but no model-specific ROCmFPX serving profile is known")

args = profile.get("server_args", [])
launch_command = [str(binary), "-m", caps["model"], *args]
if args and device not in args and "-dev" in args:
    warnings.append(f"requested DEVICE={device}, but recommended profile args use {args[args.index('-dev') + 1]}")

if not binary.name.startswith("llama-server"):
    warnings.append(f"binary name is unusual for serving: {binary}")

wrapper_path = None
if wrapper_out and not errors:
    wrapper = pathlib.Path(wrapper_out)
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"exec {shlex.join(launch_command)} \"$@\"\n",
        encoding="utf-8",
    )
    os.chmod(wrapper, 0o755)
    wrapper_path = str(wrapper)

result = {
    "status": "fail" if errors else "pass",
    "model": caps["model"],
    "model_kind": caps["model_kind"],
    "binary": str(binary),
    "device": device,
    "supports_mtp": caps["supports_mtp"],
    "supports_diffusion": caps["supports_diffusion"],
    "supports_dflash": caps["supports_dflash"],
    "is_qat": caps["is_qat"],
    "is_agent": caps["is_agent"],
    "serving_profile": profile,
    "launch_command": launch_command,
    "wrapper_out": wrapper_path,
    "warnings": warnings,
    "errors": errors,
}

print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(1 if errors else 0)
PY
