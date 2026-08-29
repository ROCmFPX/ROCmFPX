#include "rocmfpx-plugin.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace {

struct loaded_plugin {
    std::string path;
    const rocmfpx_plugin_v1 * descriptor;
#if defined(_WIN32)
    HMODULE handle;
#else
    void * handle;
#endif
};

std::mutex g_mutex;
std::vector<loaded_plugin> g_plugins;
std::unordered_set<std::string> g_paths;

void host_log(int level, const char * message) {
    const char * label = level == ROCMFPX_PLUGIN_LOG_ERROR ? "error" :
                         level == ROCMFPX_PLUGIN_LOG_WARN  ? "warn"  :
                         level == ROCMFPX_PLUGIN_LOG_DEBUG ? "debug" : "info";
    fprintf(stderr, "rocmfpx-plugin: %s: %s\n", label, message ? message : "");
}

bool is_library(const std::filesystem::path & path) {
#if defined(_WIN32)
    return path.extension() == ".dll";
#elif defined(__APPLE__)
    return path.extension() == ".dylib" || path.extension() == ".so";
#else
    return path.extension() == ".so";
#endif
}

int load_one(const std::filesystem::path & input) {
    std::error_code ec;
    const auto path = std::filesystem::weakly_canonical(input, ec);
    if (ec || !std::filesystem::is_regular_file(path, ec) || !is_library(path)) {
        return 0;
    }
    const std::string key = path.string();
    if (g_paths.count(key)) {
        return 0;
    }

#if defined(_WIN32)
    HMODULE handle = LoadLibraryW(path.wstring().c_str());
    auto query = handle ? reinterpret_cast<rocmfpx_plugin_query_fn>(GetProcAddress(handle, "rocmfpx_plugin_query")) : nullptr;
#else
    void * handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    auto query = handle ? reinterpret_cast<rocmfpx_plugin_query_fn>(dlsym(handle, "rocmfpx_plugin_query")) : nullptr;
#endif
    if (!handle || !query) {
        host_log(ROCMFPX_PLUGIN_LOG_WARN, ("cannot load " + key + " or missing rocmfpx_plugin_query").c_str());
        if (handle) {
#if defined(_WIN32)
            FreeLibrary(handle);
#else
            dlclose(handle);
#endif
        }
        return 0;
    }

    const rocmfpx_plugin_host_v1 host = {
        ROCMFPX_PLUGIN_ABI_VERSION, sizeof(rocmfpx_plugin_host_v1), host_log,
    };
    const rocmfpx_plugin_v1 * plugin = query(ROCMFPX_PLUGIN_ABI_VERSION, &host);
    if (!plugin || plugin->abi_version != ROCMFPX_PLUGIN_ABI_VERSION ||
        plugin->struct_size < sizeof(rocmfpx_plugin_v1) || !plugin->name ||
        (plugin->on_load && plugin->on_load() != 0)) {
        host_log(ROCMFPX_PLUGIN_LOG_WARN, ("rejected incompatible plugin " + key).c_str());
#if defined(_WIN32)
        FreeLibrary(handle);
#else
        dlclose(handle);
#endif
        return 0;
    }

    g_paths.insert(key);
    g_plugins.push_back({ key, plugin, handle });
    host_log(ROCMFPX_PLUGIN_LOG_INFO, ("loaded " + std::string(plugin->name) + " " +
             (plugin->version ? plugin->version : "unknown") + " from " + key).c_str());
    return 1;
}

} // namespace

int rocmfpx_plugins_load(const char * path_list) {
    if (!path_list || !*path_list) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    int loaded = 0;
#if defined(_WIN32)
    const char separator = ';';
#else
    const char separator = ':';
#endif
    std::string paths(path_list);
    size_t begin = 0;
    while (begin <= paths.size()) {
        const size_t end = paths.find(separator, begin);
        const std::string item = paths.substr(begin, end == std::string::npos ? end : end - begin);
        if (!item.empty()) {
            std::error_code ec;
            const std::filesystem::path path(item);
            if (std::filesystem::is_directory(path, ec)) {
                std::vector<std::filesystem::path> candidates;
                for (const auto & entry : std::filesystem::directory_iterator(path, ec)) {
                    if (!ec && entry.is_regular_file() && is_library(entry.path())) {
                        candidates.push_back(entry.path());
                    }
                }
                std::sort(candidates.begin(), candidates.end());
                for (const auto & candidate : candidates) {
                    loaded += load_one(candidate);
                }
            } else {
                loaded += load_one(path);
            }
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return loaded;
}

size_t rocmfpx_plugins_count(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_plugins.size();
}

const rocmfpx_plugin_v1 * rocmfpx_plugin_at(size_t index) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return index < g_plugins.size() ? g_plugins[index].descriptor : nullptr;
}

void rocmfpx_plugins_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto it = g_plugins.rbegin(); it != g_plugins.rend(); ++it) {
        if (it->descriptor->on_unload) {
            it->descriptor->on_unload();
        }
#if defined(_WIN32)
        FreeLibrary(it->handle);
#else
        dlclose(it->handle);
#endif
    }
    g_plugins.clear();
    g_paths.clear();
}
