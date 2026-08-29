#include "rocmfpx-plugin.h"

#include <cstring>
#include <cstdio>

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "requirement failed at line %d: %s\n", __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(int argc, char ** argv) {
    REQUIRE(argc == 2);
    REQUIRE(rocmfpx_plugins_count() == 0);
    REQUIRE(rocmfpx_plugins_load(argv[1]) == 1);
    REQUIRE(rocmfpx_plugins_load(argv[1]) == 0);
    REQUIRE(rocmfpx_plugins_count() == 1);

    const rocmfpx_plugin_v1 * plugin = rocmfpx_plugin_at(0);
    REQUIRE(plugin != nullptr);
    REQUIRE(std::strcmp(plugin->name, "rocmfpx-test-plugin") == 0);
    REQUIRE((plugin->capabilities & ROCMFPX_PLUGIN_CAP_SIDECAR) != 0);
    REQUIRE((plugin->capabilities & ROCMFPX_PLUGIN_CAP_REPACK_CACHE) != 0);
    REQUIRE(rocmfpx_plugin_at(1) == nullptr);

    rocmfpx_plugins_shutdown();
    REQUIRE(rocmfpx_plugins_count() == 0);
    return 0;
}
