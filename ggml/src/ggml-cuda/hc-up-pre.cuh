#pragma once
#include "common.cuh"

// HC gate projection (Q8_0 weights, F32 activations) fused with the gated stream mix that consumes it:
//   gate = W x lo        (same int8 arithmetic as MMQ: Q8_1 activations, per-32 scales, K order)
//   dst  = scale * sum_c xn[c] * sigmoid(gate[c])   (same expression and order as dsv4_hc_pre)
// The [hc*n_embd, T] gate tensor is never written. Returns false when the shapes are not supported.
bool ggml_cuda_hc_up_pre_supported(const ggml_tensor * w, const ggml_tensor * lo, const ggml_tensor * gate,
        const ggml_tensor * pre);
void ggml_cuda_hc_up_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * lo,
        const ggml_tensor * pre);
