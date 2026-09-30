#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr, ggml_tensor * q_l2_dst = nullptr, ggml_tensor * k_l2_dst = nullptr);
