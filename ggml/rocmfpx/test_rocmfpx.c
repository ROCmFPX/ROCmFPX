#include "rocmfpx.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void fill_row(float * x, int n) {
    for (int i = 0; i < n; ++i) {
        const float wave = 0.75f*sinf((float) i * 0.37f) + 0.25f*cosf((float) i * 0.13f);
        const float ramp = ((float) (i % 11) - 5.0f) * 0.035f;
        x[i] = wave + ramp;
    }

    x[7]  =  3.25f;
    x[19] = -2.75f;
    x[43] =  1.875f;
}

static float mse(const float * a, const float * b, int n) {
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        err += d*d;
    }

    return err / (float) n;
}

int main(void) {
    enum { N = 64 };

    float src[N];
    float fp3[N];
    float fp6[N];
    float fp8[N];

    block_rocmfp3 q3[N / QK_ROCMFP3];
    block_rocmfp6 q6[N / QK_ROCMFP6];
    block_rocmfp8 q8[N / QK_ROCMFP8];

    fill_row(src, N);

    rocmfpx_quantize_row_fp3_ref(src, q3, N);
    rocmfpx_quantize_row_fp6_ref(src, q6, N);
    rocmfpx_quantize_row_fp8_ref(src, q8, N);

    assert(rocmfpx_validate_row_data_fp3(q3, sizeof(q3)));
    assert(rocmfpx_validate_row_data_fp6(q6, sizeof(q6)));
    assert(rocmfpx_validate_row_data_fp8(q8, sizeof(q8)));

    rocmfpx_dequantize_row_fp3(q3, fp3, N);
    rocmfpx_dequantize_row_fp6(q6, fp6, N);
    rocmfpx_dequantize_row_fp8(q8, fp8, N);

    const float mse3 = mse(src, fp3, N);
    const float mse6 = mse(src, fp6, N);
    const float mse8 = mse(src, fp8, N);

    printf("ROCmFP3: block=%zu row=%zu bpw=%.2f mse=%g\n",
            sizeof(block_rocmfp3), rocmfpx_row_size_fp3(N),
            8.0f*(float) sizeof(block_rocmfp3)/(float) QK_ROCMFP3, mse3);
    printf("ROCmFP6: block=%zu row=%zu bpw=%.2f mse=%g\n",
            sizeof(block_rocmfp6), rocmfpx_row_size_fp6(N),
            8.0f*(float) sizeof(block_rocmfp6)/(float) QK_ROCMFP6, mse6);
    printf("ROCmFP8: block=%zu row=%zu bpw=%.2f mse=%g\n",
            sizeof(block_rocmfp8), rocmfpx_row_size_fp8(N),
            8.0f*(float) sizeof(block_rocmfp8)/(float) QK_ROCMFP8, mse8);

    assert(isfinite(mse3));
    assert(isfinite(mse6));
    assert(isfinite(mse8));
    assert(mse8 < mse6);
    assert(mse6 < mse3);

    return 0;
}
