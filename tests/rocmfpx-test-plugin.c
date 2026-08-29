#include "rocmfpx-plugin.h"

static int loaded;

static int on_load(void) {
    loaded = 1;
    return 0;
}

static void on_unload(void) {
    loaded = 0;
}

static const struct rocmfpx_plugin_v1 plugin = {
    ROCMFPX_PLUGIN_ABI_VERSION,
    sizeof(struct rocmfpx_plugin_v1),
    "rocmfpx-test-plugin",
    "1.0.0",
    ROCMFPX_PLUGIN_CAP_SIDECAR | ROCMFPX_PLUGIN_CAP_REPACK_CACHE,
    on_load,
    on_unload,
};

const struct rocmfpx_plugin_v1 * rocmfpx_plugin_query(
        uint32_t host_abi_version, const struct rocmfpx_plugin_host_v1 * host) {
    if (host_abi_version != ROCMFPX_PLUGIN_ABI_VERSION || !host ||
        host->abi_version != ROCMFPX_PLUGIN_ABI_VERSION || !host->log) {
        return 0;
    }
    return &plugin;
}
