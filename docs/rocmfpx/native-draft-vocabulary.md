# Native draft vocabulary

This opt-in Flash Next path adds a draft-vocabulary selector, its state, and graph integration. Enable it with `ROCMFPX_DRAFT_VOCAB=16384`. Target verification still evaluates the full vocabulary. The feature currently supports one MTP driver per model and one sequence; concurrent driver ownership is rejected.

The selector uses a sampled cutoff, SIMD screening, exact comparison-based partitioning, and ordered bitset emission. Sampling only reduces work: if the candidate pool is too small, a complete scan supplies the exact partition. Scratch storage is reused between decode steps. Signed zero, equal scores, NaNs, infinities, and token-ID tie order have explicit compatibility tests.

The draft driver holds an ownership lease and shared candidate state. Graph inputs retain the state while graphs exist. The model registry holds only weak references. Updated row IDs are uploaded into fixed-shape graph inputs, which support graph reuse for one-token decoding. Indirect ggml matrix multiplication computes selected head rows; masked scattering preserves the full-vocabulary output interface. Tied or non-finite winning logits use the existing sampler.

## Provenance and scope

The selector, state, and graph integration are original code under the repository MIT license. The lower half of the candidate budget covers low token IDs, a valid previous token is reserved, and the remaining slots follow target scores. Selected rows are projected for the draft head, and masked scattering keeps the full-vocabulary output interface.

The implementation and tests were written with AI assistance and require normal human review before upstream submission. Existing upstream ggml operators and the repository MIT license remain in use; model licensing is unchanged.

## Validation

`test-rocmfpx-draft-select` compares selection with a complete-sort oracle across deterministic fixtures, changing sizes, ties, adversarial score layouts, and unusual IEEE values. `test-rocmfpx-draft-vocab` checks F32/BF16/Q8_0 selected projection, masks across updates, graph reuse eligibility, winner mapping, fallback behavior, driver ownership, and resource lifetime. The latter test is registered only for builds with a directly linked CPU backend.

A faster selector microbenchmark is not an end-to-end decoding measurement.

## Enabling the fast Qwen3.8 Flash Next path

Every switch defaults off. A normal build and the existing ROCmFP4 v2 launcher keep their current kernels and draft behavior on RDNA 3, RDNA 3.5, and RDNA 4.

Set these only for the measured Flash Next server:

```text
ROCMFPX_DRAFT_VOCAB=16384
ROCMFPX_QWEN4_QSA_DECODE=1
ROCMFPX_QWEN4_PAD_HEAD=1
ROCMFPX_NGRAM_WINDOW=1
ROCMFPX_NGRAM_FLOOR4=1
ROCMFPX_HIP_GRAPH_COPY_KERNEL=1
ROCMFPX_MOE_SORT_IDS=1
ROCMFPX_ROCMI4_MOE_RPB8=1
```

`ROCMFPX_QWEN4_QSA_DECODE=1` requires the Radeon 8060S and stops if the device or tensor shape is outside the kernel it was measured on. `ROCMFPX_HIP_GRAPH_COPY_KERNEL` and `ROCMFPX_ROCMI4_MOE_RPB8` act only on RDNA 3.5. The signed ROCmI4 dot is compiled only for gfx1151; other GPU targets keep the previous dot.

`GGML_ROCMI4_Q8_1_MMVQ_VDR=4` is an optional gfx1151 compile definition. CMake does not set it. Leaving it unset preserves the existing ROCmI4 mat-vec width on every architecture.
