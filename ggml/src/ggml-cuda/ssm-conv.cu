#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"
#include <cstring>

template <bool apply_silu, size_t split_d_inner, size_t d_conv, bool fuse_qk_l2>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t,
                                    float * q_l2_ptr, float * k_l2_ptr,
                                    const int n_q_heads, const int n_k_heads, const float l2_eps) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };
    float last_y = 0.0f;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        last_y = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
        y_block[i * stride_y + tid] = last_y;
    }

    if constexpr (fuse_qk_l2) {
        __shared__ float smem[split_d_inner];
        smem[tid] = last_y;
        __syncthreads();

        float * l2_row = nullptr;
        if (bidy < n_q_heads) {
            l2_row = q_l2_ptr + (int64_t) (bidx * n_q_heads + bidy) * split_d_inner;
        } else if (bidy < n_q_heads + n_k_heads) {
            l2_row = k_l2_ptr + (int64_t) (bidx * n_k_heads + (bidy - n_q_heads)) * split_d_inner;
        }

        if (l2_row != nullptr && tid < WARP_SIZE) {
            float tmp = 0.0f;
            for (int col = tid; col < (int) split_d_inner; col += WARP_SIZE) {
                const float xi = smem[col];
                tmp += xi * xi;
            }
            tmp = warp_reduce_sum(tmp);
            const float scale = rsqrtf(fmaxf(tmp, l2_eps * l2_eps));
            for (int col = tid; col < (int) split_d_inner; col += WARP_SIZE) {
                l2_row[col] = scale * smem[col];
            }
        }
    } else {
        GGML_UNUSED(q_l2_ptr);
        GGML_UNUSED(k_l2_ptr);
        GGML_UNUSED(n_q_heads);
        GGML_UNUSED(n_k_heads);
        GGML_UNUSED(l2_eps);
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void log_ssm_conv_l2_resources() {
    static bool logged = false;
    if (logged || getenv("GGML_CUDA_GRAPH_CENSUS") == nullptr) {
        return;
    }
    logged = true;
    int base_blocks = 0;
    int fused_blocks = 0;
#if defined(GGML_USE_HIP)
    hipFuncAttributes base{};
    hipFuncAttributes fused{};
    CUDA_CHECK(hipFuncGetAttributes(&base, (const void *) ssm_conv_f32<apply_silu, 128, 4, false>));
    CUDA_CHECK(hipFuncGetAttributes(&fused, (const void *) ssm_conv_f32<apply_silu, 128, 4, true>));
    CUDA_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &base_blocks, (const void *) ssm_conv_f32<apply_silu, 128, 4, false>, 128, 0));
    CUDA_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &fused_blocks, (const void *) ssm_conv_f32<apply_silu, 128, 4, true>, 128, 0));
#else
    cudaFuncAttributes base{};
    cudaFuncAttributes fused{};
    CUDA_CHECK(cudaFuncGetAttributes(&base, (const void *) ssm_conv_f32<apply_silu, 128, 4, false>));
    CUDA_CHECK(cudaFuncGetAttributes(&fused, (const void *) ssm_conv_f32<apply_silu, 128, 4, true>));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &base_blocks, (const void *) ssm_conv_f32<apply_silu, 128, 4, false>, 128, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &fused_blocks, (const void *) ssm_conv_f32<apply_silu, 128, 4, true>, 128, 0));
#endif
    GGML_LOG_INFO(
            "cuda_graph_census_ssm_conv_l2_resources threads=128 "
            "base_regs=%d fused_regs=%d base_shared=%zu fused_shared=%zu base_local=%zu fused_local=%zu "
            "base_blocks=%d fused_blocks=%d\n",
            base.numRegs, fused.numRegs, base.sharedSizeBytes, fused.sharedSizeBytes,
            base.localSizeBytes, fused.localSizeBytes, base_blocks, fused_blocks);
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream,
                              float * q_l2, float * k_l2, const int n_q_heads, const int n_k_heads, const float l2_eps) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);
    const bool fuse_qk_l2 = q_l2 != nullptr;
    GGML_ASSERT(fuse_qk_l2 == (k_l2 != nullptr));
    if (fuse_qk_l2) {
        GGML_ASSERT(n_t == 1);
        GGML_ASSERT(apply_silu);
        log_ssm_conv_l2_resources<apply_silu>();
    }

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            if (fuse_qk_l2) {
                ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC, true>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                            src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t,
                                            q_l2, k_l2, n_q_heads, n_k_heads, l2_eps);
            } else {
                ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC, false>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                            src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t,
                                            nullptr, nullptr, 0, 0, 0.0f);
            }
        } else {
            GGML_ASSERT(!fuse_qk_l2);
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst, ggml_tensor * q_l2_dst, ggml_tensor * k_l2_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;
    const bool fuse_qk_l2 = q_l2_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);
    GGML_ASSERT(fuse_qk_l2 == (k_l2_dst != nullptr));
    GGML_ASSERT(!fuse_qk_l2 || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    float *       q_l2_d = fuse_qk_l2 ? (float *) q_l2_dst->data : nullptr;
    float *       k_l2_d = fuse_qk_l2 ? (float *) k_l2_dst->data : nullptr;
    int           n_q_heads = 0;
    int           n_k_heads = 0;
    float         l2_eps = 0.0f;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }
    if (fuse_qk_l2) {
        GGML_ASSERT(n_t == 1);
        GGML_ASSERT(q_l2_dst->type == GGML_TYPE_F32);
        GGML_ASSERT(k_l2_dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(q_l2_dst));
        GGML_ASSERT(ggml_is_contiguous(k_l2_dst));
        GGML_ASSERT(q_l2_dst->ne[0] == 128);
        GGML_ASSERT(k_l2_dst->ne[0] == 128);
        n_q_heads = (int) q_l2_dst->ne[1];
        n_k_heads = (int) k_l2_dst->ne[1];
        GGML_ASSERT(q_l2_dst->ne[2] == n_t);
        GGML_ASSERT(k_l2_dst->ne[2] == n_t);
        GGML_ASSERT(q_l2_dst->ne[3] == n_s);
        GGML_ASSERT(k_l2_dst->ne[3] == n_s);
        GGML_ASSERT((n_q_heads + n_k_heads) * 128 <= nr);
        memcpy(&l2_eps, q_l2_dst->op_params, sizeof(float));
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream, q_l2_d, k_l2_d, n_q_heads, n_k_heads, l2_eps);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream, nullptr, nullptr, 0, 0, 0.0f);
    }
}
