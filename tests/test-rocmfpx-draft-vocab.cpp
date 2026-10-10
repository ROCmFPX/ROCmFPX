// SPDX-License-Identifier: MIT
#include "../src/rocmfpx-draft-vocab.h"
#include "../src/llama-graph.h"
#include "../src/llama-batch.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

static void test_projection(ggml_type type) {
    constexpr int width = 32, vocab = 12, count = 6;
    auto state = std::make_shared<rocmfpx::draft_vocabulary>(vocab, count);
    ggml_init_params params = {2*1024*1024, nullptr, true};
    auto * ctx = ggml_init(params);
    auto * weights = ggml_new_tensor_2d(ctx, type, width, vocab);
    auto * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 1);
    auto projection = rocmfpx::build_draft_projection(ctx, state, weights, hidden);
    llm_graph_params graph_params{};
    graph_params.ubatch.n_tokens = 1;
    require(projection.input && projection.input->can_reuse(graph_params), "single-row graph reuses its fixed candidate inputs");
    graph_params.ubatch.n_tokens = 2;
    require(!projection.input->can_reuse(graph_params), "different row count requires a different graph");
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, projection.logits);
    auto backend = ggml_backend_cpu_init();
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "allocate graph");
    std::vector<float> matrix(width*vocab), vector(width, 1.0f/width);
    for (int id = 0; id < vocab; ++id) for (int col = 0; col < width; ++col) matrix[id*width + col] = float(id+1);
    std::vector<unsigned char> packed(ggml_nbytes(weights));
    require(ggml_quantize_chunk(type, matrix.data(), packed.data(), 0, vocab, width, nullptr) == packed.size(), "convert head weights");
    ggml_backend_tensor_set(weights, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(hidden, vector.data(), 0, vector.size()*sizeof(float));
    for (int pass = 0; pass < 2; ++pass) {
        const std::vector<float> scores = pass == 0 ? std::vector<float>{0,0,0,10,9,8,7,6,5,4,3,2}
                                                   : std::vector<float>{0,0,0,2,3,4,5,6,7,8,9,10};
        const std::vector<int> expected = pass == 0 ? std::vector<int>{0,1,2,3,4,11} : std::vector<int>{0,1,2,3,10,11};
        state->update(scores.data(), vocab, pass == 0 ? 11 : 3);
        if (projection.input) projection.input->set_input(nullptr);
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute selected projection");
        std::vector<float> result(vocab);
        ggml_backend_tensor_get(projection.logits, result.data(), 0, result.size()*sizeof(float));
        for (int id = 0; id < vocab; ++id) {
            const bool selected = std::find(expected.begin(), expected.end(), id) != expected.end();
            require(selected ? std::fabs(result[id] - (id+1)) < (type == GGML_TYPE_Q8_0 ? 0.01f : 1e-5f) : result[id] == -INFINITY,
                    "selected rows equal full projection; unselected rows are masked across updates");
        }
        llama_token winner = -1;
        require(state->unique_max(result.data(), winner) && winner == 11, "unique best token maps to vocabulary ID");
        result[0] = result[11];
        require(!state->unique_max(result.data(), winner), "ties fall back to sampler");
        result[0] = std::numeric_limits<float>::quiet_NaN();
        require(!state->unique_max(result.data(), winner), "NaN falls back to sampler");
        result[0] = INFINITY;
        require(!state->unique_max(result.data(), winner), "infinite maximum falls back to sampler");
    }
    // per-sequence candidates: the graph uploads the rows of the sequence it decodes
    {
        const std::vector<float> scores0{0,0,0,10,9,8,7,6,5,4,3,2};
        const std::vector<float> scores1{0,0,0,2,3,4,5,6,7,8,9,10};
        state->update(scores0.data(), vocab, 11, 0);
        state->update(scores1.data(), vocab, 3, 1);
        llama_seq_id seq_one = 1, seq_two = 2;
        llama_seq_id * seq_ptr = &seq_one;
        llama_ubatch ubatch{};
        ubatch.n_tokens = 1;
        ubatch.seq_id = &seq_ptr;
        const std::vector<int> expected0{0,1,2,3,4,11}, expected1{0,1,2,3,10,11}, initial{0,1,2,3,4,5};
        auto check = [&](const std::vector<int> & expected, const char * what) {
            require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute selected projection");
            std::vector<float> result(vocab);
            ggml_backend_tensor_get(projection.logits, result.data(), 0, result.size()*sizeof(float));
            for (int id = 0; id < vocab; ++id) {
                const bool selected = std::find(expected.begin(), expected.end(), id) != expected.end();
                require(selected ? std::isfinite(result[id]) : result[id] == -INFINITY, what);
            }
        };
        projection.input->set_input(&ubatch);
        check(expected1, "sequence 1 projects its own candidates");
        projection.input->set_input(nullptr);
        check(expected0, "sequence 0 keeps its candidates after sequence 1 updated");
        seq_ptr = &seq_two;
        projection.input->set_input(&ubatch);
        check(initial, "a sequence without an update uses the initial rows");
        std::vector<float> logits1(vocab, -INFINITY);
        logits1[10] = 1.0f; logits1[11] = 2.0f;
        llama_token winner = -1;
        require(state->unique_max(logits1.data(), winner, 1) && winner == 11, "winner within sequence 1 candidates");
        require(!state->unique_max(logits1.data(), winner, 2) || winner != 11, "sequence 2 does not see sequence 1 candidates");
        state->reset(1);
        seq_ptr = &seq_one;
        projection.input->set_input(&ubatch);
        check(initial, "reset sequence returns to the initial rows");
    }
    auto weak = std::weak_ptr<rocmfpx::draft_vocabulary>(state);
    state.reset();
    require(!weak.expired(), "graph input keeps its state alive");
    projection.input.reset();
    require(weak.expired(), "state is released with last owner");
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

int main() {
    auto state = std::make_shared<rocmfpx::draft_vocabulary>(12, 6);
    auto lease = state->claim_driver();
    require(bool(lease), "driver claim returns an ownership lease");
    bool rejected = false;
    try { auto duplicate = state->claim_driver(); } catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "two drivers cannot overwrite one draft context's candidate state");
    lease.reset();
    lease = state->claim_driver();
    require(bool(lease), "a finished driver releases ownership");
    const auto weak = std::weak_ptr<rocmfpx::draft_vocabulary>(state);
    state.reset();
    require(!weak.expired(), "driver lease retains vocabulary lifetime");
    lease.reset();
    require(weak.expired(), "driver release does not leak vocabulary");
    require(rocmfpx::draft_vocabulary_budget(nullptr) == 0, "no candidate budget without a model");
    require(rocmfpx::draft_vocabulary_for(nullptr) == nullptr, "no candidate state without a context");
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_Q8_0}) test_projection(type);
    std::puts("PASS: F32/BF16/Q8_0 row projection, changing masks, per-sequence candidates, exact winner, sampler fallback, state lifetime");
}
