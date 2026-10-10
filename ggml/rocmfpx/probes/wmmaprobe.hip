// Probe for docs/rocmfpx/plans/rdna3-gfx1100-tg-pp-acceleration.md
// Measures V_WMMA_I32_16X16X16_IU4/IU8 and V_DOT8_I32_IU4 issue throughput (the prefill ceiling).
// Build: hipcc -O3 --offload-arch=gfx1100 -o wmmaprobe wmmaprobe.hip
// WMMA IU4 / IU8 / DOT throughput probe for gfx1100 (RDNA3).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;

#define CHAINS 4

__global__ void k_wmma_iu4(unsigned * out, int iters) {
    int32x8_t acc[CHAINS];
    int32x2_t a, b;
    a[0] = threadIdx.x * 0x1111 + 0x1234; a[1] = 0x2222;
    b[0] = 0x3333; b[1] = 0x4444;
#pragma unroll
    for (int c = 0; c < CHAINS; c++) acc[c] = (int32x8_t){0,0,0,0,0,0,0,0};
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int c = 0; c < CHAINS; c++) {
            // one 16x16x16 IU4 WMMA == 2 builtin calls over K (see ggml mma.cuh)
            acc[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, acc[c], false);
            acc[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, acc[c], false);
            a[0] ^= 1;
        }
    }
    int s = 0;
#pragma unroll
    for (int c = 0; c < CHAINS; c++) for (int k = 0; k < 8; k++) s += acc[c][k];
    if (s == 0x7ABCDEF1) out[threadIdx.x & 7] = s;
}

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
__global__ void k_wmma_iu8(unsigned * out, int iters) {
    int32x8_t acc[CHAINS];
    int32x4_t a, b;
    a[0] = 0x01020304; a[1] = 0x05060708; a[2] = 1; a[3] = 2;
    b[0] = 0x11121314; b[1] = 0x15161718; b[2] = 3; b[3] = 4;
#pragma unroll
    for (int c = 0; c < CHAINS; c++) acc[c] = (int32x8_t){0,0,0,0,0,0,0,0};
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int c = 0; c < CHAINS; c++) {
            acc[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, acc[c], false);
            a[0] ^= 1;
        }
    }
    int s = 0;
#pragma unroll
    for (int c = 0; c < CHAINS; c++) for (int k = 0; k < 8; k++) s += acc[c][k];
    if (s == 0x7ABCDEF1) out[threadIdx.x & 7] = s;
}

__global__ void k_dot8_iu4(unsigned * out, int iters) {
    unsigned a = 0x01234567u + threadIdx.x, b = 0x89ABCDEFu;
    int acc[CHAINS] = {0,0,0,0};
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int c = 0; c < CHAINS; c++) {
            unsigned d;
            asm volatile("v_dot8_i32_iu4 %0, %1, %2, %3 neg_lo:[1,1,0]"
                         : "=v"(d) : "v"(a), "v"(b), "v"((unsigned)acc[c]));
            acc[c] = (int)d;
        }
    }
    int s = 0;
#pragma unroll
    for (int c = 0; c < CHAINS; c++) s += acc[c];
    if (s == 0x7ABCDEF1) out[threadIdx.x & 7] = s;
}

template <typename K>
void run(const char * name, K kern, double ops_per_iter_per_wave, int blocks, int threads, int iters) {
    unsigned * out; hipMalloc(&out, 4096);
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    kern(blocks, threads, iters, out); hipDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < 3; r++) {
        hipEventRecord(e0); kern(blocks, threads, iters, out); hipEventRecord(e1); hipEventSynchronize(e1);
        float t; hipEventElapsedTime(&t, e0, e1); if (t < best) best = t;
    }
    double waves = (double)blocks * threads / 32.0;
    double tops = waves * ops_per_iter_per_wave * iters / (best * 1e-3) / 1e12;
    printf("%-22s blocks=%-5d thr=%-4d : %.1f TOPS  (%.3f ms)\n", name, blocks, threads, tops, best);
    hipFree(out);
}

int main(int argc, char ** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 20000;
    int threads = 256;
    for (int blocks : {96, 192, 384, 768, 1536}) {
        run("wmma_i32_iu4 (2/call)", [=](int b,int t,int i,unsigned*o){ hipLaunchKernelGGL(k_wmma_iu4, dim3(b), dim3(t), 0, 0, o, i); },
            CHAINS * 2.0 * (2.0*16*16*16) / 32.0 * 32.0, blocks, threads, iters);
    }
    for (int blocks : {192, 768, 1536}) {
        run("wmma_i32_iu8", [=](int b,int t,int i,unsigned*o){ hipLaunchKernelGGL(k_wmma_iu8, dim3(b), dim3(t), 0, 0, o, i); },
            CHAINS * (2.0*16*16*16) / 32.0 * 32.0, blocks, threads, iters);
        run("dot8_i32_iu4", [=](int b,int t,int i,unsigned*o){ hipLaunchKernelGGL(k_dot8_iu4, dim3(b), dim3(t), 0, 0, o, i); },
            CHAINS * (2.0*8) * 32.0, blocks, threads, iters);
    }
    return 0;
}
