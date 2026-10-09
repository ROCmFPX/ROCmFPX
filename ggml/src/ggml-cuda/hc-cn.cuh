#pragma once
#include "common.cuh"

struct ggml_cuda_hc_combine_norm_args {
    const ggml_tensor * inject;     // [hc, T]  F32 contiguous (pre-scale/sigmoid)
    const ggml_tensor * residual;   // [n_embd, hc, T] F32 contiguous
    const ggml_tensor * block_out;  // [n_embd, 1, T]  F32 contiguous
    const ggml_tensor * gamma;
    ggml_tensor *       out_res;
    ggml_tensor *       out_xn;     // [n_embd, hc, T]
    float               s1, b1, s2, b2;
    float               eps;
    uint16_t *          out_xn_bf16 = nullptr;
    bool                store_xn_f32 = true;     // false: consumers all read the BF16 copy
    const uint16_t *    res_in_bf16  = nullptr;   // `residual` is BF16 in place (marked bf16-only)
    const uint16_t *    blk_in_bf16  = nullptr;
    uint16_t *          res_out_bf16 = nullptr;
    void *              out_xn_q8    = nullptr;   // optional block_q8_1_mmq (D4) copy of xn for a following MMQ
};

// block_q8_1_mmq copy of an activation written by a fused producer; the next MMQ that reads that
// activation uses it instead of quantizing again (one slot per device, consumed once, stream ordered)
void         ggml_cuda_q8_mmq_cache_begin_graph(int device);
void *       ggml_cuda_q8_mmq_cache_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * t, size_t nbytes);
const void * ggml_cuda_q8_mmq_cache_lookup(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, size_t nbytes_min);

bool ggml_cuda_hc_combine_norm_supported(const ggml_cuda_hc_combine_norm_args & args, int warp_size);
void ggml_cuda_op_hc_combine_norm(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_combine_norm_args & args);
