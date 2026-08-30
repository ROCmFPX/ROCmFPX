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

The host exposes three backend-registry callbacks to v1 plugins:

- `backend_load(path)` registers a standard ggml backend shared library and
  returns its device count.
- `backend_count()` and `backend_name(index)` enumerate the active backend
  registry after registration.
- `plugin_path` is the canonical path of the plugin being queried, so relative
  sidecar assets can be resolved beside the library.

The host object passed to `rocmfpx_plugin_query` is temporary. A plugin that
needs a service later must copy the function pointer or string value during the
query; it must not retain the host-structure pointer.

Minimal descriptor:

```c
#define ROCMFPX_PLUGIN_BUILD
#include "rocmfpx-plugin.h"

static const struct rocmfpx_plugin_v1 plugin = {
    ROCMFPX_PLUGIN_ABI_VERSION, sizeof(plugin), "example", "0.1.0",
    ROCMFPX_PLUGIN_CAP_SIDECAR, 0, 0, 0, 0,
};

const struct rocmfpx_plugin_v1 * rocmfpx_plugin_query(
        uint32_t abi, const struct rocmfpx_plugin_host_v1 * host) {
    return abi == ROCMFPX_PLUGIN_ABI_VERSION && host ? &plugin : 0;
}
```

New v1 descriptors may also provide `on_model_open` and `on_model_close`.
`on_model_open` receives a non-owning model identifier, canonical model path,
architecture, file type, byte and element counts, plus flags for models with a
PLE lane or SSM layers. The strings and structure are valid only for the
duration of the callback. `on_model_close` is delivered in reverse plugin order
before the model is destroyed. Plugins should key sidecar state by `model_id`
and release it during close.

Older v1 plugins remain loadable because both host and plugin structures are
size-versioned. Capability bits are discovery contracts: a backend plugin
registers its ggml backend through `backend_load`, while a sidecar can prepare
model-scoped state in `on_model_open`. Plugins are native code and must be
treated as trusted.

Call `rocmfpx_plugins_shutdown()` explicitly only after all plugin-visible
models have been released. Otherwise ROCmFPX keeps plugins resident until the
llama library or process exits. This is deliberate: some llama.cpp tools call
`llama_backend_free()` before their stack-owned models are destroyed, and the
model-close callbacks must remain callable throughout that teardown.

Build and run a plugin on Linux:

```bash
cc -shared -fPIC -I/path/to/ROCmFPX/include plugin.c -o plugin.so
ROCMFPX_PLUGIN_PATH=/absolute/path/to/plugin.so ./llama-cli -m model.gguf ...
```

Use an explicit absolute path in production. A directory may be mounted
read-only into a container and assigned to `ROCMFPX_PLUGIN_PATH`.

The SSD lanes have distinct purposes: KV checkpoints persist resumable prompt
state, repack caches store device-specific transformed weights keyed by model
and kernel identity, and expert streaming supplies bounded MoE expert windows.
They must use atomic publication, verify model/tensor hashes, and never treat a
cache as the canonical GGUF.
