// Host oracle for ROCmI4 W4A4-MMVQ packing and V_DOT8_I32_IU4.
//
// Exact MMVQ unpacks signed nibbles and DP4As against Q8 activations.
// W4A4-MMVQ packs activations in the same Q4_0 nibble layout as the weights
// (lo = elem j, hi = elem j+16) and dots packed dwords with DOT8. When the
// activations are already on the IU4 grid, the integer accumulators must match.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_fails = 0;

#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
        ++g_fails; \
    } \
} while (0)

static int8_t nibble_i4(uint32_t x, int i) {
    const int n = (int) ((x >> (4 * i)) & 0xFu);
    return (int8_t) ((n ^ 8) - 8);
}

static int unpack_even_odd_dp4a(uint32_t a, uint32_t b) {
    int sum = 0;
    for (int byte = 0; byte < 4; ++byte) {
        const int8_t ae = nibble_i4(a, byte * 2);
        const int8_t ao = nibble_i4(a, byte * 2 + 1);
        const int8_t be = nibble_i4(b, byte * 2);
        const int8_t bo = nibble_i4(b, byte * 2 + 1);
        sum += ae * be + ao * bo;
    }
    return sum;
}

static int dot8_iu4_ref(uint32_t a, uint32_t b) {
    int sum = 0;
    for (int i = 0; i < 8; ++i) {
        sum += nibble_i4(a, i) * nibble_i4(b, i);
    }
    return sum;
}

static uint8_t pack_q4_0_byte(int8_t lo, int8_t hi) {
    return (uint8_t) ((lo & 0x0F) | ((hi & 0x0F) << 4));
}

static uint32_t load_dword(const uint8_t * qs, int i32) {
    return (uint32_t) qs[4 * i32 + 0]
         | ((uint32_t) qs[4 * i32 + 1] << 8)
         | ((uint32_t) qs[4 * i32 + 2] << 16)
         | ((uint32_t) qs[4 * i32 + 3] << 24);
}

static int8_t quant_iu4(float x, float d) {
    if (!(d > 0.0f)) {
        return 0;
    }
    int q = (int) std::round((double) (x / d));
    if (q < -8) {
        q = -8;
    }
    if (q > 7) {
        q = 7;
    }
    return (int8_t) q;
}

static uint32_t xorshift32(uint32_t & s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

static int8_t rand_i4(uint32_t & s) {
    return (int8_t) ((int) (xorshift32(s) % 16u) - 8);
}

static void test_dot8_matches_nibble_sum() {
    uint32_t s = 0xC0FFEEu;
    for (int n = 0; n < 4096; ++n) {
        const uint32_t a = xorshift32(s);
        const uint32_t b = xorshift32(s);
        const int d8 = dot8_iu4_ref(a, b);
        const int dp = unpack_even_odd_dp4a(a, b);
        CHECK(d8 == dp, "DOT8 %d != unpack+DP4A %d (a=%08x b=%08x)", d8, dp, a, b);
    }
}

static void test_packed_block_matches_q8_path() {
    constexpr int QK = 32;
    uint32_t s = 0xA5A5A5A5u;
    for (int trial = 0; trial < 256; ++trial) {
        int8_t w[QK];
        int8_t a[QK];
        uint8_t wqs[QK / 2];
        uint8_t aqs[QK / 2];
        int8_t a_q8[QK];
        for (int i = 0; i < QK; ++i) {
            w[i] = rand_i4(s);
            a[i] = rand_i4(s);
            a_q8[i] = a[i];
        }
        for (int j = 0; j < QK / 2; ++j) {
            wqs[j] = pack_q4_0_byte(w[j], w[j + 16]);
            aqs[j] = pack_q4_0_byte(a[j], a[j + 16]);
        }

        int sum_dot8 = 0;
        int sum_q8 = 0;
        int sum_scalar = 0;
        for (int i32 = 0; i32 < 4; ++i32) {
            const uint32_t wd = load_dword(wqs, i32);
            const uint32_t ad = load_dword(aqs, i32);
            sum_dot8 += dot8_iu4_ref(wd, ad);

            // Exact MMVQ: even nibbles vs a[i32*4 + 0..3], odd vs a[16 + i32*4 + 0..3]
            for (int k = 0; k < 4; ++k) {
                const int8_t we = nibble_i4(wd, k * 2);
                const int8_t wo = nibble_i4(wd, k * 2 + 1);
                sum_q8 += we * a_q8[i32 * 4 + k];
                sum_q8 += wo * a_q8[16 + i32 * 4 + k];
            }
        }
        for (int i = 0; i < QK; ++i) {
            sum_scalar += w[i] * a[i];
        }
        CHECK(sum_dot8 == sum_scalar, "DOT8 block %d != scalar %d", sum_dot8, sum_scalar);
        CHECK(sum_dot8 == sum_q8, "DOT8 block %d != Q8 path %d", sum_dot8, sum_q8);
    }
}

static void test_float_quant_pack_roundtrip() {
    constexpr int QK = 32;
    uint32_t s = 1u;
    double max_rel = 0.0;
    for (int trial = 0; trial < 128; ++trial) {
        float x[QK];
        float amax = 0.0f;
        for (int i = 0; i < QK; ++i) {
            const float u = (float) ((int) (xorshift32(s) % 2001u) - 1000) / 250.0f;
            x[i] = u;
            amax = std::fmax(amax, std::fabs(u));
        }
        const float d = amax > 0.0f ? amax / 7.0f : 0.0f;
        int8_t q[QK];
        uint8_t qs[QK / 2];
        for (int i = 0; i < QK; ++i) {
            q[i] = quant_iu4(x[i], d);
        }
        for (int j = 0; j < QK / 2; ++j) {
            qs[j] = pack_q4_0_byte(q[j], q[j + 16]);
        }
        for (int j = 0; j < QK / 2; ++j) {
            CHECK(nibble_i4(qs[j], 0) == q[j], "lo nibble mismatch");
            CHECK(nibble_i4(qs[j], 1) == q[j + 16], "hi nibble mismatch");
        }
        double num = 0.0;
        double den = 0.0;
        for (int i = 0; i < QK; ++i) {
            const double dq = (double) q[i] * (double) d;
            const double e = dq - (double) x[i];
            num += e * e;
            den += (double) x[i] * (double) x[i];
        }
        const double rel = den > 0.0 ? num / den : 0.0;
        if (rel > max_rel) {
            max_rel = rel;
        }
    }
    CHECK(max_rel < 0.05, "IU4 act quant NMSE %g too high", max_rel);
}

int main() {
    test_dot8_matches_nibble_sum();
    test_packed_block_matches_q8_path();
    test_float_quant_pack_roundtrip();
    if (g_fails) {
        std::fprintf(stderr, "%d checks failed\n", g_fails);
        return 1;
    }
    std::printf("ok: IU4 pack + DOT8 oracle\n");
    return 0;
}
