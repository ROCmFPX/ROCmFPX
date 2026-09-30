// Probe for docs/rocmfpx/plans/rdna3-gfx1100-tg-pp-acceleration.md (P-1)
// Build: hipcc -O3 --offload-arch=gfx1100 -o wmmalds4 wmmalds4.hip
// Models the real W4A4 MMQ operand reads: one 16 B LDS load per operand per
// mma_iu4, row = lane % 16, row-major stride 44 dwords (x tile) / 36 dwords
// (y tile). SA/SB are those strides in 16 B words; 0 selects consecutive
// chunks (the wmmalds2 control pattern).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;

template <int CHAINS, int SA, int SB>
__global__ void __launch_bounds__(256) k_lds(unsigned * out, int iters) {
    __shared__ uint4 sbuf[512];                       // 8 KB LDS, 16 B per lane/access
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        sbuf[i] = make_uint4(i * 2654435761u, i ^ 0x9E3779B9u, i * 40503u, i + 0x5bf03635u);
    }
    __syncthreads();

    int32x8_t acc[CHAINS];
#pragma unroll
    for (int c = 0; c < CHAINS; c++) acc[c] = (int32x8_t){0,0,0,0,0,0,0,0};

    const int lane = threadIdx.x & 31;
    const int row  = lane & 15;
    for (int i = 0; i < iters; i++) {
        const int rot = (i & 7) * 8;
#pragma unroll
        for (int c = 0; c < CHAINS; c++) {
            const int ia = SA ? row * SA + rot + c * 7 : ((lane + rot + c * 61) & 511);
            const int ib = 256 + (SB ? row * SB + rot + c * 5 : ((lane + rot + c * 97) & 511));
            const uint4 au = sbuf[ia];
            const uint4 bu = sbuf[ib];
            int32x2_t a0, a1, b0, b1;
            a0[0] = au.x; a0[1] = au.y;
            a1[0] = au.z; a1[1] = au.w;
            b0[0] = bu.x; b0[1] = bu.y;
            b1[0] = bu.z; b1[1] = bu.w;
            acc[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a0, true, b0, acc[c], false);
            acc[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a1, true, b1, acc[c], false);
        }
    }
    int s = 0;
#pragma unroll
    for (int c = 0; c < CHAINS; c++) for (int k = 0; k < 8; k++) s += acc[c][k];
    if (s == 0x7ABCDEF1) out[threadIdx.x & 7] = s;
}

template <int CHAINS, int SA, int SB>
void run(const char * name, int blocks, int threads, int iters) {
    unsigned * out; hipMalloc(&out, 4096);
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    auto kern = [&](int b, int t, int i, unsigned * o) {
        hipLaunchKernelGGL((k_lds<CHAINS, SA, SB>), dim3(b), dim3(t), 0, 0, o, i);
    };
    for (int r = 0; r < 2; r++) kern(blocks, threads, iters, out);
    hipDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < 3; r++) {
        hipEventRecord(e0); kern(blocks, threads, iters, out); hipEventRecord(e1); hipEventSynchronize(e1);
        float t; hipEventElapsedTime(&t, e0, e1); if (t < best) best = t;
    }
    const double waves = (double)blocks * threads / 32.0;
    const double ops = waves * (double)CHAINS * 2.0 * 8192.0 * (double)iters;
    printf("%-10s chains=%d blocks=%-5d : %7.1f TOPS\n", name, CHAINS, blocks, ops / (best * 1e-3) / 1e12);
    hipFree(out);
}

template <int CHAINS>
void sweep(int iters) {
    for (int blocks : {192, 768, 3072})
        run<CHAINS, 11, 9>("row44/36", blocks, 256, iters);
    for (int blocks : {192, 768, 3072})
        run<CHAINS, 0, 0>("consec  ", blocks, 256, iters);
}

int main(int argc, char ** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 20000;
    sweep<1>(iters); sweep<2>(iters); sweep<4>(iters);
    return 0;
}
