// Probe for docs/rocmfpx/plans/rdna3-gfx1100-tg-pp-acceleration.md
// TG-1: measures what the 17-byte ROCmI4 block layout costs vs 16-byte aligned.
// Build: hipcc -O3 --offload-arch=gfx1100 -o layoutprobe layoutprobe.hip
// Block = { uint8 qs[16]; uint8 e; } with no padding, so the 16-byte qs field
// starts at 17k and is never 16-byte aligned except at k=0. MMVQ walks a row
// with stride 17. This compares that walk against a 16-byte-aligned layout with
// the same useful bytes.
// Variants:
//   packed17   16B uint4 load at 17k+1 per block              (old 840.1 ref)
//   mis4       faithful MMVQ loads: four misaligned 4B loads at 17k+4j plus
//              the 1B scale at 17k+16
//   coop       workgroup stages a 256-block round (4352 B) with 272 aligned
//              uint4 loads into LDS, each thread realigns its block with
//              funnel shifts                             (rewrite candidate)
//   aligned16  16-byte blocks, aligned loads               (bandwidth ref)
// Every packed-buffer kernel prints two rates: nblk*16/t (old table metric)
// and nblk*17/t (true bytes touched, mis4/coop also read the scale byte).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define BLOCKS_PER_ROW 512        // 512 * 32 = 16384 weights per row
#define NROWS 1

// A: 17-byte blocks, read 16 bytes per block at offset 17k+1 (MMVQ pattern)
__global__ void __launch_bounds__(256) k_packed(const uint8_t * __restrict__ buf, size_t nblk, unsigned * out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0,0,0,0);
    for (; i < nblk; i += stride) {
        uint4 v;
        __builtin_memcpy(&v, buf + i*17 + 1, 16);   // misaligned by 1 byte
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 0xDEADBEEFu && acc.y == 0xFEEDFACEu && acc.z == 0x12345678u && acc.w == 0x9ABCDEF0u)
        out[threadIdx.x & 7] = acc.x;
}

// B: same useful bytes, 16-byte aligned blocks
__global__ void __launch_bounds__(256) k_aligned(const uint8_t * __restrict__ buf, size_t nblk, unsigned * out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0,0,0,0);
    for (; i < nblk; i += stride) {
        uint4 v;
        __builtin_memcpy(&v, buf + i*16, 16);       // aligned
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 0xDEADBEEFu && acc.y == 0xFEEDFACEu && acc.z == 0x12345678u && acc.w == 0x9ABCDEF0u)
        out[threadIdx.x & 7] = acc.x;
}

// C: 17-byte blocks, the real MMVQ load shape: four misaligned 4B loads of qs
// at 17k+4j (same idiom as rocmfp4_get_qs_i32) plus the 1B scale at 17k+16
__global__ void __launch_bounds__(256) k_mis4(const uint8_t * __restrict__ buf, size_t nblk, unsigned * out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0,0,0,0);
    for (; i < nblk; i += stride) {
        const unsigned * q = (const unsigned *) (buf + i*17);
        acc.x += q[0]; acc.y += q[1]; acc.z += q[2]; acc.w += q[3];
        acc.x += buf[i*17 + 16];       // scale byte
    }
    if (acc.x == 0xDEADBEEFu && acc.y == 0xFEEDFACEu && acc.z == 0x12345678u && acc.w == 0x9ABCDEF0u)
        out[threadIdx.x & 7] = acc.x;
}

// D: 17-byte blocks, cooperative aligned staging. One round = 256 blocks
// = 4352 B = 272 aligned uint4 chunks. Phase 1 stages the span into LDS
// (every thread one chunk, 16 threads a second), phase 2 has thread t rebuild
// block t from the 5 LDS dwords covering bytes 17t..17t+16.
__global__ void __launch_bounds__(256) k_coop(const uint8_t * __restrict__ buf, size_t nblk, unsigned * out) {
    __shared__ __align__(16) uint8_t stage[256*17];
    uint4 acc = make_uint4(0,0,0,0);
    for (size_t r = blockIdx.x; r*256 < nblk; r += gridDim.x) {
        const uint4 * base = (const uint4 *) (buf + r*256*17);   // r*4352 stays 16B aligned
        ((uint4 *) stage)[threadIdx.x] = base[threadIdx.x];
        if (threadIdx.x < 16)
            ((uint4 *) stage)[256 + threadIdx.x] = base[256 + threadIdx.x];
        __syncthreads();
        const unsigned * s = (const unsigned *) stage;
        const unsigned po = 17u*threadIdx.x;         // block start byte in the span
        const unsigned d0 = po >> 2;
        const unsigned sh = (po & 3u) * 8u;
        unsigned v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            v[j] = sh ? ((s[d0 + j] >> sh) | (s[d0 + j + 1] << (32 - sh))) : s[d0 + j];
        acc.x += v[0]; acc.y += v[1]; acc.z += v[2]; acc.w += v[3];
        acc.x += (s[(po + 16u) >> 2] >> (((po + 16u) & 3u) * 8u)) & 0xFFu;   // scale byte
        __syncthreads();               // stage is overwritten next round
    }
    if (acc.x == 0xDEADBEEFu && acc.y == 0xFEEDFACEu && acc.z == 0x12345678u && acc.w == 0x9ABCDEF0u)
        out[threadIdx.x & 7] = acc.x;
}

int main(void) {
    const size_t nblk = getenv("NBLK") ? strtoull(getenv("NBLK"), 0, 10) : (size_t)250000000;
    const size_t sz_p = nblk * 17 + 4352, sz_a = nblk * 16;   // pad: coop rounds up to whole 256-block spans
    uint8_t *bp, *ba; unsigned *out;
    hipMalloc(&bp, sz_p); hipMalloc(&ba, sz_a); hipMalloc(&out, 4096);
    hipMemset(bp, 1, sz_p); hipMemset(ba, 1, sz_a);
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);

    auto bench = [&](const char * name, void (*k)(const uint8_t*, size_t, unsigned*), uint8_t * buf, double touched) {
        int blocks = 1536, threads = 256;
        k<<<blocks, threads>>>(buf, nblk, out); hipDeviceSynchronize();
        float best = 1e30f;
        for (int r = 0; r < 3; r++) {
            hipEventRecord(e0); k<<<blocks, threads>>>(buf, nblk, out); hipEventRecord(e1); hipEventSynchronize(e1);
            float t; hipEventElapsedTime(&t, e0, e1); if (t < best) best = t;
        }
        const double useful = (double)nblk*16;
        printf("%-9s nblk=%zu useful=%.2f GB : %7.1f GB/s x16", name, nblk, useful/1e9, useful/(best*1e-3)/1e9);
        if (touched) printf(" | %7.1f GB/s x17", touched/(best*1e-3)/1e9);
        printf("  (%.3f ms)\n", best);
    };
    const double t17 = (double)nblk*17;
    bench("packed17",  k_packed,  bp, t17);
    bench("mis4",      k_mis4,    bp, t17);
    bench("coop",      k_coop,    bp, t17);
    bench("aligned16", k_aligned, ba, 0);
    return 0;
}
