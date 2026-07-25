#!/usr/bin/env bash
# Shared HIP architecture helpers for the ROCmFPX build scripts. Sourced, not executed.
#
# Building for the wrong AMDGPU target does not fail the build: the binary links,
# loads the model, and then segfaults or hangs with no diagnostic (issues #18, #37).
# These helpers pick the right target when the machine can tell us, and check the
# emitted code objects afterwards so a mismatch is loud instead of silent.

# Print every AMDGPU target the machine reports, one per line.
rocmfpx_detect_hip_archs() {
    local out=""

    if command -v rocm_agent_enumerator >/dev/null 2>&1; then
        out="$(rocm_agent_enumerator 2>/dev/null || true)"
    fi
    if [[ -z "${out//[[:space:]]/}" ]] && command -v offload-arch >/dev/null 2>&1; then
        out="$(offload-arch 2>/dev/null || true)"
    fi
    if [[ -z "${out//[[:space:]]/}" ]] && command -v rocminfo >/dev/null 2>&1; then
        out="$(rocminfo 2>/dev/null || true)"
    fi

    # gfx000 is the placeholder the enumerator prints for the CPU agent.
    printf '%s\n' "${out}" \
        | tr ' \t' '\n\n' \
        | grep -oE '^gfx[0-9a-f]{3,4}[a-z]*$' \
        | grep -vx 'gfx000' \
        | sort -u
}

# rocmfpx_select_hip_arch <fallback> <family regex>
#
# An explicit CMAKE_HIP_ARCHITECTURES always wins. Otherwise use the installed GPU
# when it belongs to this script's family, so a card that needs a sibling target
# does not silently get the family default. Falls back to the historical value when
# nothing matching is visible, which keeps offline and cross-compile builds working.
rocmfpx_select_hip_arch() {
    local fallback="$1" family="$2" arch

    if [[ -n "${CMAKE_HIP_ARCHITECTURES:-}" ]]; then
        printf '%s\n' "${CMAKE_HIP_ARCHITECTURES}"
        return 0
    fi

    while read -r arch; do
        [[ -z "${arch}" ]] && continue
        if [[ "${arch}" =~ ${family} ]]; then
            if [[ "${arch}" != "${fallback}" ]]; then
                printf 'Detected %s; building for it instead of the default %s.\n' "${arch}" "${fallback}" >&2
                printf 'Set CMAKE_HIP_ARCHITECTURES to build for a different target.\n' >&2
            fi
            printf '%s\n' "${arch}"
            return 0
        fi
    done < <(rocmfpx_detect_hip_archs)

    printf '%s\n' "${fallback}"
}

# rocmfpx_reset_stale_build_dir <build dir> <arch>
#
# CMake records the new target in CMakeCache.txt when the architecture changes, but
# object files compiled for the old one can survive in the tree, which is how a cache
# reading gfx1201 ended up shipping gfx1200 code objects. Changing target requires a
# full recompile anyway, so start clean. Set ROCMFPX_KEEP_STALE_BUILD=1 to opt out.
rocmfpx_reset_stale_build_dir() {
    local build_dir="$1" arch="$2" cached

    [[ -f "${build_dir}/CMakeCache.txt" ]] || return 0

    cached="$(sed -n 's/^CMAKE_HIP_ARCHITECTURES:[^=]*=//p' "${build_dir}/CMakeCache.txt" | head -1)"
    [[ -n "${cached}" && "${cached}" != "${arch}" ]] || return 0

    if [[ "${ROCMFPX_KEEP_STALE_BUILD:-0}" == "1" ]]; then
        printf 'Warning: %s was configured for %s, now building %s.\n' "${build_dir}" "${cached}" "${arch}" >&2
        printf 'ROCMFPX_KEEP_STALE_BUILD=1 is set, so stale code objects may survive.\n' >&2
        return 0
    fi

    printf '%s was configured for %s, now building %s.\n' "${build_dir}" "${cached}" "${arch}"
    printf 'Removing it so no code objects for the old target survive.\n'
    rm -rf "${build_dir}"
}

# rocmfpx_verify_hip_arch <build dir> <expected arch>
#
# The code objects are what actually run, so check them rather than the cache.
rocmfpx_verify_hip_arch() {
    local build_dir="$1" expected="$2" probe found want missing

    command -v strings >/dev/null 2>&1 || return 0

    for probe in "${build_dir}"/bin/libggml-hip.so "${build_dir}"/bin/llama-cli "${build_dir}"/bin/llama-bench; do
        [[ -f "${probe}" ]] || continue

        found="$(strings "${probe}" 2>/dev/null \
            | grep -oE 'gfx[0-9a-f]{3,4}[a-z]*' \
            | grep -vx 'gfx000' \
            | sort -u | tr '\n' ' ')"
        [[ -z "${found//[[:space:]]/}" ]] && continue

        # CMAKE_HIP_ARCHITECTURES may be a list; every requested target must be present.
        missing=""
        for want in ${expected//[;,]/ }; do
            [[ -z "${want}" ]] && continue
            [[ " ${found}" == *" ${want} "* ]] || missing+=" ${want}"
        done

        if [[ -n "${missing//[[:space:]]/}" ]]; then
            printf 'ERROR: %s holds code objects for [%s], missing [%s] of the requested %s.\n' \
                "${probe}" "${found% }" "${missing# }" "${expected}" >&2
            printf 'Delete %s and build again.\n' "${build_dir}" >&2
            return 1
        fi

        printf 'Verified %s code objects in %s\n' "${expected}" "${probe##*/}"
        return 0
    done

    return 0
}
