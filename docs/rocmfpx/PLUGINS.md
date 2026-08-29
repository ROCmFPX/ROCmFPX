# Plugin and sidecar ABI

ROCmFPX plugin ABI v1 is declared in `include/rocmfpx-plugin.h`. A shared
library exports `rocmfpx_plugin_query`, returns a size-versioned descriptor,
and may advertise backend, sidecar, KV-checkpoint, repack-cache, and
expert-streaming capabilities. The host validates ABI version and structure
size, calls `on_load`, and unloads plugins in reverse order.

Plugins load only from the explicit `ROCMFPX_PLUGIN_PATH` environment variable.
It is a colon-separated list on Unix and a semicolon-separated list on Windows;
entries can be shared libraries or directories. Directories are scanned once,
non-recursively and in sorted order. Canonical paths prevent duplicate loads.
The current working directory is never searched implicitly.

Minimal descriptor:

```c
#define ROCMFPX_PLUGIN_BUILD
#include "rocmfpx-plugin.h"

static const struct rocmfpx_plugin_v1 plugin = {
    ROCMFPX_PLUGIN_ABI_VERSION, sizeof(plugin), "example", "0.1.0",
    ROCMFPX_PLUGIN_CAP_SIDECAR, 0, 0,
};

const struct rocmfpx_plugin_v1 * rocmfpx_plugin_query(
        uint32_t abi, const struct rocmfpx_plugin_host_v1 * host) {
    return abi == ROCMFPX_PLUGIN_ABI_VERSION && host ? &plugin : 0;
}
```

Capability bits are discovery contracts in ABI v1. A plugin performs its own
registration or sidecar initialization in `on_load`; host callbacks for data
exchange will be added only in an ABI-compatible structure extension. Plugins
are native code and must be treated as trusted.

The SSD lanes have distinct purposes: KV checkpoints persist resumable prompt
state, repack caches store device-specific transformed weights keyed by model
and kernel identity, and expert streaming supplies bounded MoE expert windows.
They must use atomic publication, verify model/tensor hashes, and never treat a
cache as the canonical GGUF.
