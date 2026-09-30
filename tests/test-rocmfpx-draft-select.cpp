// SPDX-License-Identifier: MIT
#include "../src/rocmfpx-draft-select.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

static void require(bool value, const char * message) {
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

// Deliberately slow complete-sort oracle. The compatibility contract keeps
// [0, count/2), reserves a valid last token, and fills by descending score.
// Score ties prefer lower token IDs; signed zero compares equal. IEEE NaNs
// retain the established total ordering (positive above +inf, negative below -inf).
static bool ahead(const std::vector<float> & values, int a, int b) {
    const float x = values[a], y = values[b];
    if (std::isnan(x) || std::isnan(y)) {
        const int cx = std::isnan(x) ? (std::signbit(x) ? -1 : 1) : 0;
        const int cy = std::isnan(y) ? (std::signbit(y) ? -1 : 1) : 0;
        if (cx != cy) return cx > cy;
        uint32_t ax, by;
        std::memcpy(&ax, &x, 4); std::memcpy(&by, &y, 4);
        if (ax != by) return std::signbit(x) ? ax < by : ax > by;
    } else if (x != y) {
        return x > y;
    }
    return a < b;
}

static std::vector<int32_t> oracle(const std::vector<float> & values, int count, int last) {
    std::vector<int32_t> result(count/2), rest;
    std::iota(result.begin(), result.end(), 0);
    if (last >= count/2 && last < int(values.size())) result.push_back(last);
    for (int id = count/2; id < int(values.size()); ++id) if (id != last) rest.push_back(id);
    std::sort(rest.begin(), rest.end(), [&](int a, int b) { return ahead(values, a, b); });
    result.insert(result.end(), rest.begin(), rest.begin() + count - result.size());
    std::sort(result.begin(), result.end());
    return result;
}

int main() {
    rocmfpx::draft_selector selector;
    const std::vector<float> literal = {-8, -7, -6, -5, 40, 30, 20, 10};
    require(selector.select(literal.data(), 8, 4, 7) == std::vector<int32_t>({0,1,4,7}), "retain coverage, strongest score, and last token");
    const std::vector<float> ties(12, 0.0f);
    require(selector.select(ties.data(), 12, 6, 11) == std::vector<int32_t>({0,1,2,3,4,11}), "ties use lowest token IDs");
    std::mt19937 rng(19271);
    std::normal_distribution<float> normal(0, 6);
    int checked = 2;
    for (int n : {3, 17, 257, 8193, 248320}) {
        for (int mode = 0; mode < 9; ++mode) {
            std::vector<float> values(n);
            for (int i = 0; i < n; ++i) {
                float v = normal(rng);
                if (mode == 1) v = std::round(v);
                if (mode == 2) v = i % 2 ? 0.0f : -0.0f;
                if (mode == 3) v = float(i);
                if (mode == 4) v = float(-i);
                if (mode == 5) { uint32_t bits = rng(); std::memcpy(&v, &bits, 4); }
                if (mode == 6) v = i % 2 ? INFINITY : -INFINITY;
                if (mode == 7) v = i % 127 == 0 ? 20 : -20;
                if (mode == 8) v = (i > n/2 && i < n/2 + n/16) ? 50 : -50;
                values[i] = v;
            }
            for (int count : {1, std::min(n, 16), std::min(n, 16384), n}) {
                for (int last : {-1, 0, n/2, n-1, n+1}) {
                    const auto got = selector.select(values.data(), n, count, last);
                    require(got == oracle(values, count, last), "selection agrees with full-sort oracle");
                    ++checked;
                }
            }
        }
    }
    require(selector.select(nullptr, 0, 0, -1).empty(), "empty vocabulary is safe");
    bool rejected = false;
    try { selector.select(literal.data(), 8, 9, -1); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "reject count beyond vocabulary");
    std::printf("PASS: %d selector cases plus empty and invalid-input checks\n", checked);
}
