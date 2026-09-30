// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace rocmfpx {
class draft_selector {
public:
    const std::vector<int32_t> & select(const float * scores, int vocabulary, int budget, int previous) {
        if (vocabulary < 0 || budget < 0 || budget > vocabulary || (vocabulary && !scores)) {
            throw std::invalid_argument("invalid draft vocabulary selection");
        }
        result_.clear();
        if (!budget) return result_;
        const int permanent = budget / 2;
        const bool reserve_previous = previous >= permanent && previous < vocabulary;
        const int wanted = budget - permanent - int(reserve_previous);
        result_.resize(permanent);
        std::iota(result_.begin(), result_.end(), 0);
        if (reserve_previous) result_.push_back(previous);
        if (!wanted) return result_;

        // A sampled cutoff is only a work-reduction hint. Partitioning below
        // is exact; an underestimated candidate pool triggers a complete scan.
        // Unlike a radix selector, no full-vocabulary key array or digit passes
        // are required. Scratch buffers persist between decode iterations.
        samples_.clear();
        const int population = vocabulary - permanent;
        const int sample_count = std::min(population, 1024);
        for (int i = 0; i < sample_count; ++i) {
            const int token = permanent + int((int64_t(i) * population + population / 2) / sample_count);
            if (token != previous) samples_.push_back(rank(scores[token], token));
        }
        uint64_t cutoff = 0;
        if (!samples_.empty()) {
            const size_t target = std::min(samples_.size() - 1,
                (size_t(wanted) * samples_.size() * 3 / (size_t(population) * 2)) + 8);
            std::nth_element(samples_.begin(), samples_.begin() + target, samples_.end(), std::greater<uint64_t>());
            cutoff = samples_[target];
        }
        candidates_.clear();
        collect(scores, permanent, vocabulary, previous, cutoff);
        if (candidates_.size() < size_t(wanted)) {
            candidates_.clear();
            for (int token = permanent; token < vocabulary; ++token) {
                if (token != previous) candidates_.push_back(rank(scores[token], token));
            }
        }
        if (candidates_.size() > size_t(wanted)) {
            std::nth_element(candidates_.begin(), candidates_.begin() + wanted, candidates_.end(), std::greater<uint64_t>());
        }
        // Emit in token order without sorting thousands of selected IDs.
        membership_.assign((population + 63) / 64, 0);
        if (reserve_previous) {
            result_.pop_back();
            const int local = previous - permanent;
            membership_[local / 64] |= UINT64_C(1) << (local % 64);
        }
        for (int i = 0; i < wanted; ++i) {
            const int local = int32_t(~uint32_t(candidates_[i])) - permanent;
            membership_[local / 64] |= UINT64_C(1) << (local % 64);
        }
        for (size_t word = 0; word < membership_.size(); ++word) {
            uint64_t bits = membership_[word];
            while (bits) {
                result_.push_back(permanent + int(word * 64) + trailing_zeroes(bits));
                bits &= bits - 1;
            }
        }
        return result_;
    }
private:
    void collect(const float * scores, int first, int end, int previous, uint64_t cutoff) {
        int token = first;
#if defined(__SSE2__)
        // Screen four scores at a time, materializing keys only for survivors.
        // Integer ordering also handles signed NaNs without an unordered-float
        // comparison. This path needs only the x86-64 baseline ISA.
        const __m128i zero = _mm_setzero_si128();
        const __m128i sign = _mm_set1_epi32(INT32_MIN);
        const __m128i threshold = _mm_set1_epi32(int32_t(uint32_t(cutoff >> 32) ^ UINT32_C(0x80000000)));
        for (; token + 4 <= end; token += 4) {
            __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(scores + token));
            raw = _mm_andnot_si128(_mm_cmpeq_epi32(_mm_slli_epi32(raw, 1), zero), raw);
            const __m128i ordered = _mm_xor_si128(raw, _mm_or_si128(_mm_srai_epi32(raw, 31), sign));
            const __m128i signed_order = _mm_xor_si128(ordered, sign);
            const __m128i eligible = _mm_or_si128(_mm_cmpgt_epi32(signed_order, threshold), _mm_cmpeq_epi32(signed_order, threshold));
            unsigned mask = unsigned(_mm_movemask_ps(_mm_castsi128_ps(eligible)));
            while (mask) {
                const int id = token + trailing_zeroes(mask);
                const uint64_t key = rank(scores[id], id);
                if (id != previous && key >= cutoff) candidates_.push_back(key);
                mask &= mask - 1;
            }
        }
#endif
        for (; token < end; ++token) {
            const uint64_t key = rank(scores[token], token);
            if (token != previous && key >= cutoff) candidates_.push_back(key);
        }
    }
    static uint64_t rank(float value, int token) {
        // Monotonic IEEE encoding plus token ID gives a deterministic total
        // order, including NaN payloads. Normalize the two zero encodings.
        uint32_t raw = 0;
        if (value != 0) std::memcpy(&raw, &value, sizeof(raw));
        const uint32_t ordered = raw ^ ((raw >> 31) ? UINT32_MAX : UINT32_C(0x80000000));
        return (uint64_t(ordered) << 32) | uint32_t(~uint32_t(token));
    }
    static int trailing_zeroes(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_ctzll(value);
#else
        int n = 0;
        while (!(value & 1)) { value >>= 1; ++n; }
        return n;
#endif
    }
    std::vector<int32_t> result_;
    std::vector<uint64_t> samples_;
    std::vector<uint64_t> candidates_;
    std::vector<uint64_t> membership_;
};
}
