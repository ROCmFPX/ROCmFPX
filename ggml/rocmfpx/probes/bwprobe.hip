// Probe for docs/rocmfpx/plans/rdna3-gfx1100-tg-pp-acceleration.md
// Measures achievable streaming read bandwidth (the decode ceiling).
// Build: hipcc -O3 --offload-arch=gfx1100 -o bwprobe bwprobe.hip
// Streaming read/write bandwidth probe for RDNA3 (gfx1100).
// Usage: bwprobe <GB> <blocks> <threads> <unroll>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

template <int UNROLL>
__global__ void __launch_bounds__(1024) k_read(const uint4 * __restrict__ in, size_t n4, uint4 * __restrict__ out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0,0,0,0);
    size_t i2 = i;
    // main unrolled loop
    while (i2 + (UNROLL-1)*stride < n4) {
        uint4 v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; u++) v[u] = in[i2 + (size_t)u*stride];
#pragma unroll
        for (int u = 0; u < UNROLL; u++) { acc.x += v[u].x; acc.y += v[u].y; acc.z += v[u].z; acc.w += v[u].w; }
        i2 += (size_t)UNROLL*stride;
    }
    for (; i2 < n4; i2 += stride) { uint4 v = in[i2]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
    // data-dependent store: forces the loads to be computed, but never actually stores
    if (acc.x == 0xDEADBEEFu && acc.y == 0xFEEDFACEu && acc.z == 0x12345678u && acc.w == 0x9ABCDEF0u) {
        out[threadIdx.x & 7] = acc;
    }
}

int main(int argc, char ** argv) {
    size_t gb = argc > 1 ? strtoull(argv[1], 0, 10) : 8;
    int blocks = argc > 2 ? atoi(argv[2]) : 1536;
    int threads = argc > 3 ? atoi(argv[3]) : 256;
    int unroll = argc > 4 ? atoi(argv[4]) : 4;
    size_t bytes = gb << 30;
    size_t n4 = bytes / sizeof(uint4);
    uint4 * buf = nullptr; uint4 * out = nullptr;
    if (hipMalloc(&buf, bytes) != hipSuccess) { printf("alloc fail\n"); return 1; }
    hipMemset(buf, 1, bytes);
    hipMalloc(&out, 4096);
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    // warmup
    hipLaunchKernelGGL((k_read<4>), dim3(256), dim3(256), 0, 0, buf, n4, out);
    hipDeviceSynchronize();
    float best = 0; double ms = 0;
    for (int r = 0; r < 3; r++) {
        hipEventRecord(e0);
        switch (unroll) {
            case 1: hipLaunchKernelGGL((k_read<1>), dim3(blocks), dim3(threads), 0, 0, buf, n4, out); break;
            case 2: hipLaunchKernelGGL((k_read<2>), dim3(blocks), dim3(threads), 0, 0, buf, n4, out); break;
            case 4: hipLaunchKernelGGL((k_read<4>), dim3(blocks), dim3(threads), 0, 0, buf, n4, out); break;
            case 8: hipLaunchKernelGGL((k_read<8>), dim3(blocks), dim3(threads), 0, 0, buf, n4, out); break;
        }
        hipEventRecord(e1); hipEventSynchronize(e1);
        float t; hipEventElapsedTime(&t, e0, e1); ms = t;
        double gbs = bytes / (t * 1e-3) / 1e9;
        if (gbs > best) best = gbs;
    }
    printf("read  %zu GB blocks=%d threads=%d unroll=%d : %.1f GB/s (last %.3f ms)\n", gb, blocks, threads, unroll, best, ms);
    hipFree(buf); hipFree(out);
    return 0;
}
