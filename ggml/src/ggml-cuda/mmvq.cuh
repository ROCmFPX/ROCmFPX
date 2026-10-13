#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

// Per-graph cache of the q8_1-quantized matvec input (implemented in ggml-cuda.cu).
// ggml_cuda_mmvq_q8_cache_get() returns a device buffer of at least nbytes for src1 in the given layout, or nullptr
// when the cache is disabled or full. `hit` is set when the buffer already holds src1 quantized by an earlier
// matvec of the same graph on the same stream, so the caller can skip the quantize kernel.
void   ggml_cuda_mmvq_q8_cache_begin_graph(int device);
char * ggml_cuda_mmvq_q8_cache_get(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t ne10_padded, size_t nbytes, bool & hit);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
