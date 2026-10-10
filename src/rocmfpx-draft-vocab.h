// SPDX-License-Identifier: MIT
#pragma once
#include "llama.h"
#include <memory>

struct ggml_context;
struct ggml_tensor;
class llm_graph_input_i;

namespace rocmfpx {

// Candidate rows for the draft head of one MTP context. The context owns it
// (one per draft context, so independent drivers never share candidates) and
// graph inputs share its lifetime. Candidates are kept per sequence: the draft
// graph uploads the rows of the sequence it decodes. Target verification never
// uses this object.
class LLAMA_API draft_vocabulary : public std::enable_shared_from_this<draft_vocabulary> {
public:
    draft_vocabulary(int vocabulary, int budget);
    ~draft_vocabulary();
    draft_vocabulary(const draft_vocabulary &) = delete;
    draft_vocabulary & operator=(const draft_vocabulary &) = delete;
    int size() const;
    int vocabulary_size() const;
    // Reject concurrent draft drivers using the same context. Releasing this
    // lease permits another driver, including during exception unwinding.
    std::shared_ptr<void> claim_driver();
    // Select the candidates of sequence seq_id from full-vocabulary scores.
    void update(const float * scores, int vocabulary, llama_token previous, llama_seq_id seq_id = 0);
    // Forget the candidates of seq_id (it starts again from the initial rows).
    void reset(llama_seq_id seq_id);
    bool unique_max(const float * full_logits, llama_token & token, llama_seq_id seq_id = 0) const;
    void upload(ggml_tensor * projection_ids, ggml_tensor * scatter_ids, llama_seq_id seq_id = 0) const;
private:
    struct storage;
    std::unique_ptr<storage> data_;
};

// Candidate budget for MTP draft contexts of this model: ROCMFPX_DRAFT_VOCAB
// when set (0 disables), otherwise kDefaultBudget for qwen4exp, 0 for other
// architectures. Never larger than the vocabulary.
constexpr int kDefaultDraftVocabularyBudget = 16384;
LLAMA_API int draft_vocabulary_budget(const llama_model * model);

// The candidate state of an MTP draft context, or nullptr when it has none.
LLAMA_API std::shared_ptr<draft_vocabulary> draft_vocabulary_for(const llama_context * ctx);

struct draft_projection {
    ggml_tensor * logits;
    std::unique_ptr<llm_graph_input_i> input;
};

// Project a single hidden row through the selected vocabulary rows and return
// a full-size logit tensor whose unselected entries are negative infinity.
LLAMA_API draft_projection build_draft_projection(ggml_context * ctx,
        std::shared_ptr<draft_vocabulary> vocabulary, ggml_tensor * weights, ggml_tensor * hidden);
}
