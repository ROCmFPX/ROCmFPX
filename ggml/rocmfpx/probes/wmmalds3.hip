// Probe for docs/rocmfpx/plans/rdna3-gfx1100-tg-pp-acceleration.md
// Build: hipcc -O3 --offload-arch=gfx1100 -o wmmalds3 wmmalds3.hip
//   memcpy_1<8>(t.x+0, base+0);  memcpy_1<8>(t.x+2, base+2);
// i.e. per mma_iu4: A = 2x8B, B = 2x8B = four 8-byte LDS loads.
// Probe 2 used two 16-byte loads instead. This isolates load *count* vs *width*.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;

template <int CHAINS>
__global__ void __launch_bounds__(256) k_4x8(unsigned * out, int iters) {
    __shared__ uint2 sbuf[1024];                       // 8 KB
    for (int i = threadIdx.x; i < 1024; i += blockDim.x)
        sbuf[i] = make_uint2(i * 2654435761u, i ^ 0x9E3779B9u);
    __syncthreads();

    int32x8_t acc[CHAINS];
#pragma unroll
    for (int c = 0; c < CHAINS; c++) acc[c] = (int32x8_t){0,0,0,0,0,0,0,0};

    const int lane = threadIdx.x & 31;
    for (int i = 0; i < iters; i++) {
        const int rot = (i & 7) * 8;
#pragma unroll
        for (int c = 0; c < CHAINS; c++) {
            // four distinct 8-byte LDS loads, spread across banks
            const uint2 a0u = sbuf[(lane * 2 + rot + c * 61) & 1023];
            const uint2 a1u = sbuf[(lane * 2 + rot + c * 61 + 256) & 1023];
            const uint2 b0u = sbuf[(lane * 2 + rot + c * 97 + 512) & 1023];
            const uint2 b1u = sbuf[(lane * 2 + rot + c * 97 + 768) & 1023];
            int32x2_t a0, a1, b0, b1;
            a0[0] = a0u.x; a0[1] = a0u.y;
            a1[0] = a1u.x; a1[1] = a1u.y;
            b0[0] = b0u.x; b0[1] = b0u.y;
            b1[0] = b1u.x; b1[1] = b1u.y;
            acc[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a0, true, b0, acc[c], false);
            acc[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a1, true, b1, acc[c], false);
        }
    }
    int s = 0;
#pragma unroll
    for (int c = 0; c < CHAINS; c++) for (int k = 0; k < 8; k++) s += acc[c][k];
    if (s == 0x7ABCDEF1) out[threadIdx.x & 7] = s;
}

template <int CHAINS, typename K>
void run(const char * name, K kern, int blocks, int threads, int iters) {
    unsigned * out; hipMalloc(&out, 4096);
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    for (int r = 0; r < 2; r++) kern(blocks, threads, iters, out);
    hipDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < 3; r++) {
        hipEventRecord(e0); kern(blocks, threads, iters, out); hipEventRecord(e1); hipEventSynchronize(e1);
        float t; hipEventElapsedTime(&t, e0, e1); if (t < best) best = t;
    }
    const double waves = (double)blocks * threads / 32.0;
    const double ops = waves * (double)CHAINS * 2.0 * 8192.0 * (double)iters;
    printf("%-6s blocks=%-5d : %7.1f TOPS\n", name, blocks, ops / (best * 1e-3) / 1e12);
    hipFree(out);
}

template <int CHAINS>
void sweep(int iters) {
    for (int blocks : {192, 768, 3072})
        run<CHAINS>("4x8B", [=](int b,int t,int i,unsigned*o){ hipLaunchKernelGGL((k_4x8<CHAINS>), dim3(b), dim3(t), 0, 0, o, i); }, blocks, 256, iters);
    printf("  ^ four 8B LDS loads per mma_iu4 (real loader), CHAINS=%d\n", CHAINS);
}

int main(int argc, char ** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 20000;
    sweep<1>(iters); sweep<4>(iters);
    return 0;
}
