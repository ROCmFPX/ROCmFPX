#include "models.h"

// Laguna-XS.2 (Poolside)
// - iSWA (interleaved sliding-window attention): every 4th layer is full-attention, rest are SWA(512).
// - Per-layer head count: full layers have n_head=48, sliding layers have n_head=64. n_head_kv=8 always.
// - Q-norm + K-norm at head_dim level (RMSNorm).
// - Per-head softplus attention gate: out = attn_out.view(*, n_heads, head_dim) * softplus(g_proj(x).float()).unsqueeze(-1).
// - Partial RoPE: full layers rotate first n_rot_full=64 dims (partial_rotary_factor=0.5), sliding rotates all n_rot_swa=128.
// - Full layers use YaRN scaling; sliding layers use default RoPE with rope_theta=10000.
// - Layer 0 dense MLP. Layers 1..n-1 sparse MoE: 256 experts, top-8, sigmoid router with score-correction bias,
//   sum-normalize selected weights, route_scale=2.5, plus an always-on shared expert (intermediate=512).
// - No post-attn norm, no post-ffn norm. Pre-attn (input_layernorm) and pre-ffn (post_attention_layernorm).

void llama_model_laguna::load_arch_hparams(llama_model_loader & ml) {
    // 40-layer iSWA model: every 4th layer is full-attention, rest are SWA(512).
    // Per-layer head count varies (48 vs 64). YaRN on full layers, default RoPE on sliding.
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp,           false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp,    false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm,  false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa, false);

    // Default leading dense layer count = 1 (layer 0 dense, rest MoE).
    if (hparams.n_layer_dense_lead == 0) {
        hparams.n_layer_dense_lead = 1;
    }
    // Sigmoid router by default.
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }
    // sum-normalize selected expert weights, scale routed combine by 2.5
    hparams.expert_weights_norm = true;
    if (hparams.expert_weights_scale == 0.0f) {
        hparams.expert_weights_scale = 2.5f;
    }

    // iSWA pattern: pattern of 4, full layer is the FIRST in each group of 4 (dense_first=true).
    // Reference: layer_types[i] = full if i%4==0 else sliding.
    if (hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        uint32_t swa_period = 4;
        ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false);
        hparams.set_swa_pattern(swa_period, /*dense_first=*/true);
    } else {
        hparams.swa_type = LLAMA_SWA_TYPE_NONE;
    }

    // Sliding layers use rope_theta=10000 (full rotation), full layers use the main rope_theta
    // (500000) with YaRN scaling. The converter writes both and TextModel.set_gguf_parameters
    // wires up rope_freq_base + rope_freq_base_swa.
    hparams.rope_freq_base_train_swa = 10000.0f;
    hparams.rope_freq_scale_train_swa = 1.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);

    switch (hparams.n_layer) {
        case 40: type = LLM_TYPE_UNKNOWN; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_laguna::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    const int64_t n_ff_exp   = hparams.n_ff_exp;       // 512
    const int64_t n_ff_shexp = hparams.n_ff_shexp;     // 512

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        // per-layer head count: 48 for full layers, 64 for sliding layers
        const int64_t n_head_il   = hparams.n_head(i);
        const int64_t n_embd_q_il = n_embd_head_k * n_head_il;

        // pre-attn norm
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // attention projections
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_q_il}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q_il, n_embd}, 0);

        // Q/K head-dim RMSNorm
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        // per-head softplus gate: g_proj: hidden -> n_head_il (NOT n_head_il*head_dim)
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head_il}, 0);

        // pre-ffn norm (post_attention_layernorm in HF)
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (static_cast<uint32_t>(i) >= hparams.n_layer_dense_lead) {
            // sparse MoE: 256 experts, top-8, sigmoid router with score-correction bias
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);

            // grouped expert weights
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,    n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,    n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,    n_ff_exp, n_expert}, 0);

            // always-on shared expert
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,    n_ff_shexp}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd   }, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,    n_ff_shexp}, 0);
        } else {
            // dense SwiGLU layer (layer 0)
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_laguna::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_laguna::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        // Per-layer head count + per-layer-type RoPE base/scale + partial-RoPE n_rot.
        const int64_t n_head_il = hparams.n_head(il);
        const int64_t n_rot_il  = hparams.n_rot(il);

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        // YaRN params apply only to full-attention layers (rope_scaling_type_train is YARN for them).
        // For sliding layers we drop ext_factor/attn_factor/beta_* by zeroing ext_factor,
        // which makes ggml_rope_ext fall back to plain RoPE math.
        const bool is_swa = hparams.is_swa(il);
        const float ext_factor_l  = is_swa ? 0.0f          : ext_factor;
        const float attn_factor_l = is_swa ? 1.0f          : attn_factor;
        const float beta_fast_l   = is_swa ? 32.0f         : beta_fast;
        const float beta_slow_l   = is_swa ? 1.0f          : beta_slow;

        ggml_tensor * inpSA = inpL;

        // pre-attn RMSNorm (input_layernorm)
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            ggml_tensor * attn_inp = cur;  // saved for gate projection

            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            // gate projection (before any reshaping/normalization), shape [n_head_il, n_tokens]
            ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, attn_inp);
            cb(gate, "attn_gate_proj", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head_il, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            // Q/K head-dim RMSNorm
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
            cb(Kcur, "Kcur_normed", il);

            // partial RoPE: rotate first n_rot_il dims (NeoX layout). full layers => 64/128, sliding => 128/128.
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot_il, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
            cb(Qcur, "Qcur_rope", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot_il, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
            cb(Kcur, "Kcur_rope", il);

            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // attention without applying o_proj (we still need the gating step)
            cur = build_attn(inp_attn,
                    NULL, NULL, NULL,  // wo will be applied after gating
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);

            // Per-head softplus gate. attn_out: [n_head_il*head_dim, n_tokens].
            // Reshape to [head_dim, n_head_il, n_tokens], multiply by softplus(gate)[1, n_head_il, n_tokens] (broadcast).
            // softplus computed in fp32 for numerical parity with the reference.
            gate = ggml_cast(ctx0, gate, GGML_TYPE_F32);
            gate = ggml_softplus(ctx0, gate);
            cb(gate, "attn_gate_softplus", il);

            // gate currently [n_head_il, n_tokens]; reshape to [1, n_head_il, n_tokens] so it broadcasts on head_dim.
            gate = ggml_reshape_3d(ctx0, gate, 1, n_head_il, n_tokens);
            // cast back to attn dtype
            gate = ggml_cast(ctx0, gate, cur->type);

            cur = ggml_reshape_3d(ctx0, cur, n_embd_head, n_head_il, n_tokens);
            cur = ggml_mul(ctx0, cur, gate);
            cb(cur, "attn_gated", il);
            cur = ggml_reshape_2d(ctx0, cur, n_embd_head * n_head_il, n_tokens);

            // o_proj
            cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
            cb(cur, "attn_o_proj", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // pre-ffn RMSNorm (post_attention_layernorm in HF)
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // dense (layer 0) or sparse MoE (layers >= n_layer_dense_lead)
        if ((uint32_t)il >= hparams.n_layer_dense_lead) {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU,
                    hparams.expert_weights_norm,
                    hparams.expert_weights_scale,                          // routed scaling factor (2.5)
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            // always-on shared expert
            ggml_tensor * ffn_shexp = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        } else {
            // dense SwiGLU
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}