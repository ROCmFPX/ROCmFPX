// SPDX-License-Identifier: MIT
#pragma once
#include "llama.h"
#include <memory>

struct ggml_context;
struct ggml_tensor;
class llm_graph_input_i;

namespace rocmfpx {

// A draft driver owns the vocabulary; graph inputs share its lifetime.
// Target verification never uses this object.
class LLAMA_API draft_vocabulary : public std::enable_shared_from_this<draft_vocabulary> {
public:
    draft_vocabulary(int vocabulary, int budget);
    ~draft_vocabulary();
    draft_vocabulary(const draft_vocabulary &) = delete;
    draft_vocabulary & operator=(const draft_vocabulary &) = delete;
    int size() const;
    int vocabulary_size() const;
    // Reject concurrent draft drivers using the same model. Releasing this
    // lease permits another driver, including during exception unwinding.
    std::shared_ptr<void> claim_driver();
    void update(const float * scores, int vocabulary, llama_token previous);
    bool unique_max(const float * full_logits, llama_token & token) const;
    void upload(ggml_tensor * projection_ids, ggml_tensor * scatter_ids) const;
private:
    struct storage;
    std::unique_ptr<storage> data_;
};

LLAMA_API std::shared_ptr<draft_vocabulary> draft_vocabulary_for(const llama_model * model);

struct draft_projection {
    ggml_tensor * logits;
    std::unique_ptr<llm_graph_input_i> input;
};

// Project a single hidden row through the selected vocabulary rows and return
// a full-size logit tensor whose unselected entries are negative infinity.
LLAMA_API draft_projection build_draft_projection(ggml_context * ctx,
        std::shared_ptr<draft_vocabulary> vocabulary, ggml_tensor * weights, ggml_tensor * hidden);
}
