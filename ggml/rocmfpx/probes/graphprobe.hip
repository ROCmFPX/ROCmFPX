// Probe for docs/rocmfpx/plans/rdna3-gfx1100-tg-pp-acceleration.md
// Measures per-kernel dispatch cost, plain launches vs a HIP graph.
// Build: hipcc -O3 --offload-arch=gfx1100 -o graphprobe graphprobe.hip
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
__global__ void tiny(float * out, float v) { if (threadIdx.x == 1024) out[0] = v; }
__global__ void small(float * out, float v) { float a = v; for (int i=0;i<64;i++) a = fmaf(a,1.0000001f,v); if (a == 12345.f) out[0]=a; }

int main(int argc, char ** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 2000;
    float * d; hipMalloc(&d, 1024);
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipStream_t s; hipStreamCreate(&s);

    auto bench = [&](const char * name, void (*k)(float*,float), int iters) {
        for (int i=0;i<10;i++) k<<<1,64,0,s>>>(d, 1.f);
        hipStreamSynchronize(s);
        hipEventRecord(e0, s);
        for (int i = 0; i < iters; i++) k<<<1,64,0,s>>>(d, (float)i);
        hipEventRecord(e1, s); hipStreamSynchronize(s);
        float tp; hipEventElapsedTime(&tp, e0, e1);
        hipGraph_t g; hipGraphExec_t ge;
        hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal);
        for (int i = 0; i < iters; i++) k<<<1,64,0,s>>>(d, (float)i);
        if (hipStreamEndCapture(s, &g) != hipSuccess) { printf("  capture FAILED\n"); return; }
        if (hipGraphInstantiate(&ge, g, nullptr, nullptr, 0) != hipSuccess) { printf("  instantiate FAILED\n"); return; }
        hipGraphLaunch(ge, s); hipStreamSynchronize(s);
        hipEventRecord(e0, s); hipGraphLaunch(ge, s); hipEventRecord(e1, s); hipStreamSynchronize(s);
        float tg; hipEventElapsedTime(&tg, e0, e1);
        printf("%-8s N=%d : plain %8.3f ms (%5.2f us/disp) | graph %8.3f ms (%5.2f us/disp) | %.2fx\n",
               name, iters, tp, tp*1000/iters, tg, tg*1000/iters, tp/tg);
        hipGraphExecDestroy(ge); hipGraphDestroy(g);
    };
    bench("tiny", tiny, N);
    bench("small", small, N);
    return 0;
}
