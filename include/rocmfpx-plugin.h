#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(LLAMA_SHARED)
#    if defined(LLAMA_BUILD)
#      define ROCMFPX_API __declspec(dllexport)
#    else
#      define ROCMFPX_API __declspec(dllimport)
#    endif
#  else
#    define ROCMFPX_API
#  endif
#  if defined(ROCMFPX_PLUGIN_BUILD)
#    define ROCMFPX_PLUGIN_EXPORT __declspec(dllexport)
#  else
#    define ROCMFPX_PLUGIN_EXPORT
#  endif
#else
#  define ROCMFPX_API __attribute__((visibility("default")))
#  define ROCMFPX_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#define ROCMFPX_PLUGIN_ABI_VERSION 1u

enum rocmfpx_plugin_capability {
    ROCMFPX_PLUGIN_CAP_BACKEND          = 1ull << 0,
    ROCMFPX_PLUGIN_CAP_SIDECAR          = 1ull << 1,
    ROCMFPX_PLUGIN_CAP_KV_CHECKPOINT    = 1ull << 2,
    ROCMFPX_PLUGIN_CAP_REPACK_CACHE     = 1ull << 3,
    ROCMFPX_PLUGIN_CAP_EXPERT_STREAMING = 1ull << 4,
};

enum rocmfpx_plugin_log_level {
    ROCMFPX_PLUGIN_LOG_ERROR = 0,
    ROCMFPX_PLUGIN_LOG_WARN  = 1,
    ROCMFPX_PLUGIN_LOG_INFO  = 2,
    ROCMFPX_PLUGIN_LOG_DEBUG = 3,
};

typedef void (*rocmfpx_plugin_log_fn)(int level, const char * message);

struct rocmfpx_plugin_host_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    rocmfpx_plugin_log_fn log;
};

struct rocmfpx_plugin_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char * name;
    const char * version;
    uint64_t capabilities;
    int  (*on_load)(void);
    void (*on_unload)(void);
};

typedef const struct rocmfpx_plugin_v1 * (*rocmfpx_plugin_query_fn)(
        uint32_t host_abi_version, const struct rocmfpx_plugin_host_v1 * host);

// Every v1 shared library exports this exact symbol.
ROCMFPX_PLUGIN_EXPORT const struct rocmfpx_plugin_v1 * rocmfpx_plugin_query(
        uint32_t host_abi_version, const struct rocmfpx_plugin_host_v1 * host);

// Load a platform-separated list of files or directories. A directory is
// scanned non-recursively for shared libraries. Duplicate canonical paths are
// ignored. Returns the number of newly loaded plugins, or a negative value when
// the path list itself is invalid.
ROCMFPX_API int rocmfpx_plugins_load(const char * path_list);
ROCMFPX_API size_t rocmfpx_plugins_count(void);
ROCMFPX_API const struct rocmfpx_plugin_v1 * rocmfpx_plugin_at(size_t index);
ROCMFPX_API void rocmfpx_plugins_shutdown(void);

#if defined(__cplusplus)
}
#endif
