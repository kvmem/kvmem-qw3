#include "qwen_executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace qw3 {
namespace {

void require_status(const DeviceStatus &st) {
    if (!st.ok) throw std::runtime_error(st.message);
}

uint64_t tensor_rows(const GgufTensorInfo &tensor) {
    if (tensor.dims.size() < 2) return 1;
    uint64_t rows = 1;
    for (size_t i = 1; i < tensor.dims.size(); ++i) rows *= tensor.dims[i];
    return rows;
}

} // namespace

QwenExecutor::QwenExecutor(const QwenNativeModel &model,
                           const QwenWeights &weights,
                           DeviceBackend &backend,
                           uint32_t kv_ctx_size)
    : model_(model), weights_(weights), backend_(backend),
      kv_ctx_size_(kv_ctx_size) {}

QwenExecutor::~QwenExecutor() = default;

void QwenExecutor::reset_state() {
    for (auto &s : recurrent_states_) {
        if (s) (void) backend_.zero_tensor(*s);
    }
    for (auto &s : conv_states_) {
        if (s) (void) backend_.zero_tensor(*s);
    }
    // KV caches stay allocated; just reset the position so the next forward
    // overwrites slot 0 (the seq_len passed to attention_decode is position+1).
    position_ = 0;
    decode_graph_warmup_pending_ = true;
}

void QwenExecutor::record(NativeExecutorReport &report, const std::string &op) const {
    report.ops_executed++;
    report.executed.push_back(op);
}

void QwenExecutor::ensure_scratch() {
    if (scratch_ready_) return;
    const QwenConfig &cfg = model_.config();

    // Walk layer shapes once to find the largest per-layer dims so scratch
    // buffers can be sized to the worst case and reused across all layers.
    uint64_t max_ffn = 0;
    uint64_t max_q = 0;
    uint64_t max_k = 0;
    uint64_t max_v = 0;
    uint64_t max_recurrent_qkv = 0;
    uint64_t max_recurrent_value = 0;
    for (uint32_t i = 0; i < weights_.n_layers(); ++i) {
        const QwenLayerWeights &l = weights_.layer(i);
        if (l.ffn_dim > max_ffn) max_ffn = l.ffn_dim;
        if (l.q_rows > max_q) max_q = l.q_rows;
        if (l.k_rows > max_k) max_k = l.k_rows;
        if (l.v_rows > max_v) max_v = l.v_rows;
        if (l.recurrent_qkv_dim > max_recurrent_qkv) max_recurrent_qkv = l.recurrent_qkv_dim;
        if (l.recurrent_value_dim > max_recurrent_value) max_recurrent_value = l.recurrent_value_dim;
    }

    h_ = backend_.tensor_f32(cfg.n_embd, "h");
    norm_ = backend_.tensor_f32(cfg.n_embd, "norm");
    attn_out_ = backend_.tensor_f32(cfg.n_embd, "attn_out");
    ffn_gate_ = backend_.tensor_f32(std::max<uint64_t>(max_ffn, 1), "ffn_gate");
    ffn_up_ = backend_.tensor_f32(std::max<uint64_t>(max_ffn, 1), "ffn_up");
    ffn_mid_ = backend_.tensor_f32(std::max<uint64_t>(max_ffn, 1), "ffn_mid");
    ffn_out_ = backend_.tensor_f32(cfg.n_embd, "ffn_out");
    if (max_recurrent_qkv > 0) proj_ = backend_.tensor_f32(max_recurrent_qkv, "recurrent_proj");
    if (max_recurrent_value > 0) gate_proj_ = backend_.tensor_f32(max_recurrent_value, "recurrent_gate");
    if (max_recurrent_value > 0) core_ = backend_.tensor_f32(max_recurrent_value, "recurrent_core");
    if (max_q > 0) q_ = backend_.tensor_f32(max_q, "attn_q");
    if (max_k > 0) k_ = backend_.tensor_f32(max_k, "attn_k");
    if (max_v > 0) v_ = backend_.tensor_f32(max_v, "attn_v");

    if (cfg.num_v_heads() > 0) {
        alpha_ = backend_.tensor_f32(cfg.num_v_heads(), "recurrent_alpha");
        beta_ = backend_.tensor_f32(cfg.num_v_heads(), "recurrent_beta");
        // Conv1d output (silu(conv(qkv_mixed))) is the same size as the
        // qkv_mixed projection. Allocate once and reuse across all 48
        // recurrent layers and across all tokens.
        if (max_recurrent_qkv > 0) {
            conv_out_ = backend_.tensor_f32(max_recurrent_qkv, "recurrent_conv_out");
        }
    }
    mid_ = backend_.tensor_f32(static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim, "attn_mid");

    // Per-layer DeltaNet state + conv1d ring buffer (only recurrent layers).
    recurrent_states_.resize(weights_.n_layers());
    conv_states_.resize(weights_.n_layers());
    if (cfg.num_v_heads() > 0 && cfg.ssm_conv_kernel > 0) {
        const uint64_t state_size = static_cast<uint64_t>(cfg.num_v_heads())
                                  * cfg.head_v_dim_ssm() * cfg.head_k_dim();
        // conv_dim = key_dim*2 + value_dim. This matches the shape of `proj_`
        // (recurrent_qkv_dim) tracked when we built QwenWeights.
        for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
            const QwenLayerWeights &l = weights_.layer(il);
            if (!l.recurrent) continue;
            const std::string slbl = "recurrent_state_l" + std::to_string(il);
            const std::string clbl = "conv_state_l" + std::to_string(il);
            recurrent_states_[il] = backend_.tensor_f32(state_size, slbl.c_str());
            const uint64_t conv_dim = l.recurrent_qkv_dim; // K+K+V channels
            conv_states_[il] = backend_.tensor_f32(
                conv_dim * (cfg.ssm_conv_kernel - 1), clbl.c_str());
        }
    }
    scores_ = backend_.tensor_f32(static_cast<uint64_t>(cfg.n_heads) * std::max<uint32_t>(kv_ctx_size_, 1), "attn_scores");

    // Per-layer KV cache for the standard-attention layers only.
    k_cache_.resize(weights_.n_layers());
    v_cache_.resize(weights_.n_layers());
    const uint64_t kv_per_pos = static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
    // Default KV cache dtype: FP16 (2x bandwidth at long context, ~equal
    // greedy-token output). Force back to FP32 with QW3_KV_DTYPE=fp32.
    const char *kv_dtype_env = std::getenv("QW3_KV_DTYPE");
    const bool kv_use_fp16 = !(kv_dtype_env && std::strcmp(kv_dtype_env, "fp32") == 0);
    for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
        if (!cfg.is_standard_attention_layer(il)) continue;
        const std::string klabel = "k_cache_l" + std::to_string(il);
        const std::string vlabel = "v_cache_l" + std::to_string(il);
        if (kv_use_fp16) {
            k_cache_[il] = backend_.tensor_f16(kv_per_pos * kv_ctx_size_, klabel.c_str());
            v_cache_[il] = backend_.tensor_f16(kv_per_pos * kv_ctx_size_, vlabel.c_str());
        } else {
            k_cache_[il] = backend_.tensor_f32(kv_per_pos * kv_ctx_size_, klabel.c_str());
            v_cache_[il] = backend_.tensor_f32(kv_per_pos * kv_ctx_size_, vlabel.c_str());
        }
    }

    const GgufTensorInfo *head = model_.output();
    logits_ = backend_.tensor_f32(tensor_rows(*head), "logits");

    scratch_ready_ = true;
}

void QwenExecutor::ensure_batch_scratch(uint32_t batch) {
    if (batch == 0) return;
    if (batch <= batch_capacity_) return;
    const QwenConfig &cfg = model_.config();

    // Determine worst-case per-layer dims (mirrors ensure_scratch).
    uint64_t max_ffn = 0, max_q = 0, max_k = 0, max_v = 0;
    uint64_t max_rqkv = 0, max_rvalue = 0;
    for (uint32_t i = 0; i < weights_.n_layers(); ++i) {
        const QwenLayerWeights &l = weights_.layer(i);
        if (l.ffn_dim > max_ffn) max_ffn = l.ffn_dim;
        if (l.q_rows > max_q) max_q = l.q_rows;
        if (l.k_rows > max_k) max_k = l.k_rows;
        if (l.v_rows > max_v) max_v = l.v_rows;
        if (l.recurrent_qkv_dim > max_rqkv) max_rqkv = l.recurrent_qkv_dim;
        if (l.recurrent_value_dim > max_rvalue) max_rvalue = l.recurrent_value_dim;
    }

    const uint64_t B = batch;
    h_batch_       = backend_.tensor_f32(B * cfg.n_embd,             "h_batch");
    norm_batch_    = backend_.tensor_f32(B * cfg.n_embd,             "norm_batch");
    attn_out_batch_= backend_.tensor_f32(B * cfg.n_embd,             "attn_out_batch");
    ffn_gate_batch_= backend_.tensor_f32(B * std::max<uint64_t>(max_ffn, 1), "ffn_gate_batch");
    ffn_up_batch_  = backend_.tensor_f32(B * std::max<uint64_t>(max_ffn, 1), "ffn_up_batch");
    ffn_mid_batch_ = backend_.tensor_f32(B * std::max<uint64_t>(max_ffn, 1), "ffn_mid_batch");
    ffn_out_batch_ = backend_.tensor_f32(B * cfg.n_embd,             "ffn_out_batch");
    if (max_rqkv  > 0) proj_batch_      = backend_.tensor_f32(B * max_rqkv,  "proj_batch");
    if (max_rqkv  > 0) conv_out_batch_  = backend_.tensor_f32(B * max_rqkv,  "conv_out_batch");
    if (max_rvalue> 0) gate_proj_batch_ = backend_.tensor_f32(B * max_rvalue,"gate_proj_batch");
    if (max_rvalue> 0) core_batch_      = backend_.tensor_f32(B * max_rvalue,"core_batch");
    if (cfg.num_v_heads() > 0) {
        alpha_batch_ = backend_.tensor_f32(B * cfg.num_v_heads(),    "alpha_batch");
        beta_batch_  = backend_.tensor_f32(B * cfg.num_v_heads(),    "beta_batch");
    }
    if (max_q > 0) q_batch_ = backend_.tensor_f32(B * max_q, "q_batch");
    if (max_k > 0) k_batch_ = backend_.tensor_f32(B * max_k, "k_batch");
    if (max_v > 0) v_batch_ = backend_.tensor_f32(B * max_v, "v_batch");
    mid_batch_ = backend_.tensor_f32(B * static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim, "mid_batch");

    batch_capacity_ = batch;
}

uint64_t QwenExecutor::per_token_scratch_bytes() const {
    const QwenConfig &cfg = model_.config();
    uint64_t max_ffn = 0, max_q = 0, max_k = 0, max_v = 0;
    uint64_t max_rqkv = 0, max_rvalue = 0;
    for (uint32_t i = 0; i < weights_.n_layers(); ++i) {
        const QwenLayerWeights &l = weights_.layer(i);
        if (l.ffn_dim > max_ffn) max_ffn = l.ffn_dim;
        if (l.q_rows > max_q) max_q = l.q_rows;
        if (l.k_rows > max_k) max_k = l.k_rows;
        if (l.v_rows > max_v) max_v = l.v_rows;
        if (l.recurrent_qkv_dim > max_rqkv) max_rqkv = l.recurrent_qkv_dim;
        if (l.recurrent_value_dim > max_rvalue) max_rvalue = l.recurrent_value_dim;
    }
    uint64_t per_tok = 0;
    per_tok += 3 * cfg.n_embd;                                    // h, norm, attn_out
    per_tok += 3 * std::max<uint64_t>(max_ffn, 1);                // ffn_gate, ffn_up, ffn_mid
    per_tok += cfg.n_embd;                                        // ffn_out
    if (max_rqkv  > 0) per_tok += 2 * max_rqkv;                   // proj, conv_out
    if (max_rvalue> 0) per_tok += 2 * max_rvalue;                 // gate_proj, core
    if (cfg.num_v_heads() > 0) per_tok += 2 * cfg.num_v_heads();  // alpha, beta
    if (max_q > 0) per_tok += max_q;
    if (max_k > 0) per_tok += max_k;
    if (max_v > 0) per_tok += max_v;
    per_tok += static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim; // mid
    return per_tok * sizeof(float);
}

NativeExecutorReport QwenExecutor::dry_run_token(uint32_t token_id, bool execute_heavy) {
    if (execute_heavy) return forward_one_token(token_id);

    NativeExecutorReport report;
    const NativePlanInfo &plan = model_.plan();
    if (!plan.supported) {
        report.missing_kernels.push_back("native model plan is incomplete");
        return report;
    }
    ensure_scratch();

    require_status(backend_.begin());
    require_status(backend_.q8_0_get_row(*h_, weights_.token_embd(), token_id));
    record(report, "token_embedding_lookup");
    require_status(backend_.rms_norm(*norm_, *h_, weights_.output_norm(), model_.config().rms_eps));
    record(report, "output_norm");
    require_status(backend_.end());

    report.ok = true;
    return report;
}

NativeExecutorReport QwenExecutor::forward_one_token(uint32_t token_id) {
    NativeExecutorReport report;
    const NativePlanInfo &plan = model_.plan();
    if (!plan.supported) {
        report.missing_kernels.push_back("native model plan is incomplete");
        return report;
    }
    ensure_scratch();

    const QwenConfig &cfg = model_.config();
    const uint32_t head_k_dim = cfg.head_k_dim();
    const uint32_t head_v_dim = cfg.head_v_dim_ssm();
    const uint32_t num_k_heads = cfg.num_k_heads();
    const uint32_t num_v_heads = cfg.num_v_heads();
    const uint32_t standard_head_dim = cfg.head_dim;
    const uint32_t standard_n_heads = cfg.n_heads;
    const uint32_t standard_n_kv_heads = cfg.n_kv_heads;
    const float eps = cfg.rms_eps;

    require_status(backend_.begin());

    // CUDA-graph capture path: skip on the first token (warm-up: every
    // backend-side scratch buffer needs to be sized before we record
    // pointers into a graph). Skip on layer 0 when we'd have no captured
    // graph to replay yet — i.e. when capture refused to start.
    const bool try_capture = !decode_graph_warmup_pending_ && backend_.begin_capture();

    require_status(backend_.q8_0_get_row(*h_, weights_.token_embd(), token_id));
    record(report, "token_embedding_lookup");

    for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
        const QwenLayerWeights &layer = weights_.layer(il);
        require_status(backend_.rms_norm(*norm_, *h_, *layer.attn_norm, eps));
        record(report, "layer." + std::to_string(il) + ".attn_norm");

        if (layer.recurrent) {
            {
                DeviceTensor *outs[4] = {proj_.get(), gate_proj_.get(), alpha_.get(), beta_.get()};
                const DeviceWeight *ws[4] = {layer.attn_qkv, layer.attn_gate,
                                              layer.ssm_alpha, layer.ssm_beta};
                require_status(backend_.q8_0_matvec_fanout(outs, ws, 4, *norm_));
            }
            record(report, "layer." + std::to_string(il) + ".recurrent_projections");
            if (!recurrent_states_[il] || !conv_states_[il]) {
                throw std::runtime_error("recurrent state not allocated for layer " + std::to_string(il));
            }
            require_status(backend_.recurrent_single_token(*core_,
                                                           *recurrent_states_[il],
                                                           *conv_states_[il],
                                                           *conv_out_,
                                                           *proj_,
                                                           *gate_proj_,
                                                           *alpha_,
                                                           *beta_,
                                                           *layer.ssm_conv1d,
                                                           *layer.ssm_a,
                                                           *layer.ssm_dt_bias,
                                                           *layer.ssm_norm,
                                                           num_k_heads,
                                                           num_v_heads,
                                                           head_k_dim,
                                                           head_v_dim,
                                                           cfg.ssm_conv_kernel,
                                                           eps));
            record(report, "layer." + std::to_string(il) + ".deltanet_single_token");
            // Fused matvec + residual add: h += W_out * core. Falls back to
            // separate matvec + add inside the backend when the fused path
            // is unavailable (e.g. legacy QW3_MATVEC=qw3).
            if (auto st = backend_.q8_0_matvec_add(*h_, *layer.ssm_out, *core_); !st.ok) {
                require_status(backend_.q8_0_matvec(*attn_out_, *layer.ssm_out, *core_));
                require_status(backend_.add(*h_, *h_, *attn_out_));
            }
            record(report, "layer." + std::to_string(il) + ".recurrent_output_add");
        } else {
            if (position_ >= kv_ctx_size_) {
                throw std::runtime_error("KV cache full: increase --ctx (current=" +
                                         std::to_string(kv_ctx_size_) + ")");
            }
            {
                DeviceTensor *outs[3] = {q_.get(), k_.get(), v_.get()};
                const DeviceWeight *ws[3] = {layer.attn_q, layer.attn_k, layer.attn_v};
                require_status(backend_.q8_0_matvec_fanout(outs, ws, 3, *norm_));
            }
            record(report, "layer." + std::to_string(il) + ".attention_qkv_projection");

            // Per-head RMS norm using the shared head_dim-vector. Q is laid
            // out as [n_heads, 2, head_dim] so the per-unit stride is 2 *
            // head_dim and we normalize only the first head_dim (attn-Q).
            require_status(backend_.rmsnorm_per_head(*q_, *layer.attn_q_norm,
                                                     standard_n_heads,
                                                     2 * standard_head_dim,
                                                     standard_head_dim, eps));
            require_status(backend_.rmsnorm_per_head(*k_, *layer.attn_k_norm,
                                                     standard_n_kv_heads,
                                                     standard_head_dim,
                                                     standard_head_dim, eps));

            // Partial RoPE on the first rope_dim of each head's first segment.
            require_status(backend_.rope_partial(*q_, standard_n_heads,
                                                 2 * standard_head_dim,
                                                 cfg.rope_dim, position_, cfg.rope_theta));
            require_status(backend_.rope_partial(*k_, standard_n_kv_heads,
                                                 standard_head_dim,
                                                 cfg.rope_dim, position_, cfg.rope_theta));

            // Append K and V to the live cache.
            const uint32_t per_pos = standard_n_kv_heads * standard_head_dim;
            require_status(backend_.kv_append(*k_cache_[il], *k_, position_, per_pos));
            require_status(backend_.kv_append(*v_cache_[il], *v_, position_, per_pos));
            record(report, "layer." + std::to_string(il) + ".kv_append");

            const float scale = 1.0f / std::sqrt(static_cast<float>(standard_head_dim));
            require_status(backend_.attention_decode(*mid_, *scores_, *q_,
                                                     2 * standard_head_dim,
                                                     *k_cache_[il], *v_cache_[il],
                                                     standard_n_heads, standard_n_kv_heads,
                                                     standard_head_dim,
                                                     position_ + 1, scale));
            require_status(backend_.apply_attn_gate(*mid_, *q_,
                                                     2 * standard_head_dim,
                                                     standard_n_heads, standard_head_dim));
            record(report, "layer." + std::to_string(il) + ".attention_sdpa");

            // Fused matvec + residual add: h += W_out * mid.
            if (auto st = backend_.q8_0_matvec_add(*h_, *layer.attn_output, *mid_); !st.ok) {
                require_status(backend_.q8_0_matvec(*attn_out_, *layer.attn_output, *mid_));
                require_status(backend_.add(*h_, *h_, *attn_out_));
            }
        }
        record(report, "layer." + std::to_string(il) + ".attn_residual");

        require_status(backend_.rms_norm(*norm_, *h_, *layer.ffn_norm, eps));
        record(report, "layer." + std::to_string(il) + ".ffn_norm");

        // Fused FFN SwiGLU: ffn_mid = silu(W_gate * norm) * (W_up * norm)
        // in a single matvec kernel. Falls back to the two-weight matvec +
        // silu_mul pipeline if the backend doesn't implement the fused op.
        if (auto st = backend_.q8_0_matvec_silu_mul(*ffn_mid_, *layer.ffn_gate,
                                                    *layer.ffn_up, *norm_);
            !st.ok) {
            DeviceTensor *outs[2] = {ffn_gate_.get(), ffn_up_.get()};
            const DeviceWeight *ws[2] = {layer.ffn_gate, layer.ffn_up};
            require_status(backend_.q8_0_matvec_fanout(outs, ws, 2, *norm_));
            require_status(backend_.silu_mul(*ffn_mid_, *ffn_gate_, *ffn_up_));
        }
        // Fused matvec + residual add: h += W_down * ffn_mid.
        if (auto st = backend_.q8_0_matvec_add(*h_, *layer.ffn_down, *ffn_mid_); !st.ok) {
            require_status(backend_.q8_0_matvec(*ffn_out_, *layer.ffn_down, *ffn_mid_));
            require_status(backend_.add(*h_, *h_, *ffn_out_));
        }
        record(report, "layer." + std::to_string(il) + ".ffn");
    }

    require_status(backend_.rms_norm(*norm_, *h_, weights_.output_norm(), eps));
    record(report, "output_norm");
    require_status(backend_.q8_0_matvec(*logits_, weights_.output(), *norm_));

    DeviceArgmax best;
    if (try_capture) {
        // Record argmax kernel + its async D2H into the captured graph.
        require_status(backend_.argmax_launch(*logits_));
        require_status(backend_.end_capture());
        require_status(backend_.replay_graph());
        // Sync + read the pinned argmax mirror after the graph has run.
        best = backend_.argmax_collect();
    } else {
        // Eager path: pulls in the warm-up token and any token where
        // capture refused. Sets decode_graph_warmup_pending_ to false so
        // the next call attempts capture.
        best = backend_.argmax(*logits_);
        decode_graph_warmup_pending_ = false;
    }
    require_status(backend_.end());

    position_++;
    report.argmax_token = best.token;
    report.argmax_logit = best.logit;
    report.argmax_text = model_.gguf().token_text(static_cast<uint32_t>(best.token));
    record(report, "lm_head_argmax");
    report.ok = true;
    return report;
}

NativeExecutorReport QwenExecutor::forward_n_tokens(const std::vector<uint32_t> &tokens) {
    NativeExecutorReport report;
    if (tokens.empty()) { report.ok = true; return report; }
    const NativePlanInfo &plan = model_.plan();
    if (!plan.supported) {
        report.missing_kernels.push_back("native model plan is incomplete");
        return report;
    }
    ensure_scratch();
    const uint32_t total = static_cast<uint32_t>(tokens.size());

    // Prefill chunking. llama.cpp processes prefill in n_batch=512 pieces
    // so its peak compute scratch stays constant in T. qw3 originally sized
    // batch scratch to the entire prompt, which made peak memory grow
    // linearly with T — at T=64K the per-prompt batch scratch alone exceeded
    // 30 GiB of FP32 storage, vs llama's flat ~30 GiB total. To match memory
    // parity we apply the same fixed chunk cap as llama.cpp.
    //
    // Memory is the primary constraint. Throughput recovery comes from
    // amortizing per-chunk fixed costs (HGEMM dequant pipeline restart,
    // cuBLAS kernel-selection penalty at small batch, MMQ-at-short-batch
    // dispatch, whole-prompt graph capture #35), NOT from hoarding scratch.
    //
    // Override with QW3_PREFILL_CHUNK=N. Set N=0 to disable the cap entirely
    // (whole-prompt batch — original behavior, useful for benchmarking the
    // throughput tax of chunking itself).
    constexpr uint32_t kQw3DefaultPrefillChunk = 512;
    uint32_t chunk_size = std::min<uint32_t>(kQw3DefaultPrefillChunk, total);
    if (const char *env = std::getenv("QW3_PREFILL_CHUNK")) {
        int v = std::atoi(env);
        if (v > 0) {
            chunk_size = static_cast<uint32_t>(v);
        } else if (v == 0) {
            // Explicit opt-out of chunking.
            chunk_size = total;
        }
    }
    // Safety floor: even if the user set a large chunk (or QW3_PREFILL_CHUNK=0
    // disabled the cap), don't exceed what fits in 80% of currently free
    // device memory. This handles edge cases where weights + KV cache leave
    // less headroom than the requested chunk's per-prompt scratch.
    const uint64_t per_tok = per_token_scratch_bytes();
    if (per_tok > 0) {
        const uint64_t free_b = backend_.free_device_bytes();
        if (free_b > 0) {
            const uint64_t budget = (free_b * 8) / 10;
            const uint64_t fits = budget / per_tok;
            if (fits > 0 && fits < chunk_size) {
                chunk_size = static_cast<uint32_t>(fits);
                if (chunk_size > 256) chunk_size &= ~static_cast<uint32_t>(255);
            }
        }
    }
    if (chunk_size > total) chunk_size = total;
    if (chunk_size == 0) chunk_size = total;
    ensure_batch_scratch(chunk_size);

    const QwenConfig &cfg = model_.config();
    const uint32_t head_k_dim = cfg.head_k_dim();
    const uint32_t head_v_dim = cfg.head_v_dim_ssm();
    const uint32_t num_k_heads = cfg.num_k_heads();
    const uint32_t num_v_heads = cfg.num_v_heads();
    const uint32_t standard_head_dim = cfg.head_dim;
    const uint32_t standard_n_heads = cfg.n_heads;
    const uint32_t standard_n_kv_heads = cfg.n_kv_heads;
    const float eps = cfg.rms_eps;

    auto row_stride = [this](const DeviceTensor *t) -> uint32_t {
        return static_cast<uint32_t>(t->count / batch_capacity_);
    };
    const uint32_t h_stride = row_stride(h_batch_.get());
    const uint32_t ffn_stride = row_stride(ffn_gate_batch_.get());
    const uint32_t q_stride_buf = q_batch_ ? row_stride(q_batch_.get()) : 0;
    const uint32_t k_stride_buf = k_batch_ ? row_stride(k_batch_.get()) : 0;
    const uint32_t v_stride_buf = v_batch_ ? row_stride(v_batch_.get()) : 0;
    const uint32_t mid_stride = row_stride(mid_batch_.get());
    const uint32_t proj_stride = proj_batch_ ? row_stride(proj_batch_.get()) : 0;
    const uint32_t gate_proj_stride = gate_proj_batch_ ? row_stride(gate_proj_batch_.get()) : 0;
    const uint32_t alpha_stride = alpha_batch_ ? row_stride(alpha_batch_.get()) : 0;
    const uint32_t beta_stride = beta_batch_ ? row_stride(beta_batch_.get()) : 0;
    const uint32_t core_stride = core_batch_ ? row_stride(core_batch_.get()) : 0;

    require_status(backend_.begin());

    // Per-chunk graph capture is gated separately: even when the user
    // forces chunking, capture is incompatible with the multi-stream
    // HGEMM ping-pong pipeline (see chunk_size comment above). Setting
    // QW3_PREFILL_GRAPH=1 attempts capture anyway for diagnostic use.
    const bool prefill_graph_enabled = []() {
        const char *e = std::getenv("QW3_PREFILL_GRAPH");
        return e && (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0);
    }();

    // Skip capture on the first chunk (warmup): backend-side scratch
    // (q8_1 staging, fattn workspace) sizes itself on first use. After
    // that, full-size chunks attempt capture+replay; the trailing partial
    // chunk always runs eagerly so we keep one stable graph topology.
    bool capture_warmup_pending = true;

    uint32_t last_chunk_batch = 0;
    for (uint32_t chunk_off = 0; chunk_off < total; chunk_off += chunk_size) {
        const uint32_t batch = std::min(chunk_size, total - chunk_off);
        last_chunk_batch = batch;
        const uint32_t base_pos = position_ + chunk_off;
        const bool record_ops = (chunk_off == 0);
        const bool full_chunk = (batch == chunk_size);

        if (base_pos + batch > kv_ctx_size_) {
            throw std::runtime_error("KV cache full: increase --ctx (current=" +
                                     std::to_string(kv_ctx_size_) + ")");
        }

        // Embedding lookup runs eagerly: q8_0_get_rows_batch issues a
        // pageable host->device memcpy which is unsafe inside stream capture.
        std::vector<uint64_t> rows_h(batch);
        for (uint32_t i = 0; i < batch; ++i) rows_h[i] = tokens[chunk_off + i];
        require_status(backend_.q8_0_get_rows_batch(*h_batch_, weights_.token_embd(), rows_h.data(), batch));
        if (record_ops) record(report, "token_embedding_lookup_batch");

        const bool try_capture = prefill_graph_enabled
            && full_chunk && !capture_warmup_pending && backend_.begin_capture();

        for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
        const QwenLayerWeights &layer = weights_.layer(il);
        require_status(backend_.rms_norm_batch(*norm_batch_, *h_batch_, *layer.attn_norm,
                                                batch, h_stride, eps));
        if (record_ops) record(report, "layer." + std::to_string(il) + ".attn_norm_batch");

        if (layer.recurrent) {
            require_status(backend_.q8_0_matmul(*proj_batch_, *layer.attn_qkv, *norm_batch_,
                                                 batch, h_stride, proj_stride));
            require_status(backend_.q8_0_matmul(*gate_proj_batch_, *layer.attn_gate, *norm_batch_,
                                                 batch, h_stride, gate_proj_stride));
            require_status(backend_.q8_0_matmul(*alpha_batch_, *layer.ssm_alpha, *norm_batch_,
                                                 batch, h_stride, alpha_stride));
            require_status(backend_.q8_0_matmul(*beta_batch_, *layer.ssm_beta, *norm_batch_,
                                                 batch, h_stride, beta_stride));
            if (record_ops) record(report, "layer." + std::to_string(il) + ".recurrent_projections_batch");
            if (!recurrent_states_[il] || !conv_states_[il] || !conv_out_batch_) {
                throw std::runtime_error("recurrent state not allocated for layer " + std::to_string(il));
            }
            // One batched call replaces the previous T-token loop (5 kernels x
            // T tokens). The CUDA backend overrides this with 4 launches per
            // layer that internally iterate over T.
            require_status(backend_.recurrent_batch(*core_batch_,
                                                     *recurrent_states_[il],
                                                     *conv_states_[il],
                                                     *conv_out_batch_,
                                                     *proj_batch_,
                                                     *gate_proj_batch_,
                                                     *alpha_batch_,
                                                     *beta_batch_,
                                                     *layer.ssm_conv1d,
                                                     *layer.ssm_a,
                                                     *layer.ssm_dt_bias,
                                                     *layer.ssm_norm,
                                                     batch,
                                                     num_k_heads,
                                                     num_v_heads,
                                                     head_k_dim,
                                                     head_v_dim,
                                                     cfg.ssm_conv_kernel,
                                                     layer.recurrent_qkv_dim,
                                                     proj_stride,
                                                     gate_proj_stride,
                                                     alpha_stride,
                                                     beta_stride,
                                                     core_stride,
                                                     eps));
            if (record_ops) record(report, "layer." + std::to_string(il) + ".deltanet_batch");
            require_status(backend_.q8_0_matmul(*attn_out_batch_, *layer.ssm_out, *core_batch_,
                                                 batch, core_stride, h_stride));
            if (record_ops) record(report, "layer." + std::to_string(il) + ".recurrent_output_batch");
        } else {
            require_status(backend_.q8_0_matmul(*q_batch_, *layer.attn_q, *norm_batch_,
                                                 batch, h_stride, q_stride_buf));
            require_status(backend_.q8_0_matmul(*k_batch_, *layer.attn_k, *norm_batch_,
                                                 batch, h_stride, k_stride_buf));
            require_status(backend_.q8_0_matmul(*v_batch_, *layer.attn_v, *norm_batch_,
                                                 batch, h_stride, v_stride_buf));
            if (record_ops) record(report, "layer." + std::to_string(il) + ".attention_qkv_projection_batch");

            require_status(backend_.rmsnorm_per_head_batch(*q_batch_, *layer.attn_q_norm,
                                                            batch, q_stride_buf,
                                                            standard_n_heads,
                                                            2 * standard_head_dim,
                                                            standard_head_dim, eps));
            require_status(backend_.rmsnorm_per_head_batch(*k_batch_, *layer.attn_k_norm,
                                                            batch, k_stride_buf,
                                                            standard_n_kv_heads,
                                                            standard_head_dim,
                                                            standard_head_dim, eps));

            require_status(backend_.rope_partial_batch(*q_batch_,
                                                        batch, q_stride_buf,
                                                        standard_n_heads,
                                                        2 * standard_head_dim,
                                                        cfg.rope_dim, base_pos, cfg.rope_theta));
            require_status(backend_.rope_partial_batch(*k_batch_,
                                                        batch, k_stride_buf,
                                                        standard_n_kv_heads,
                                                        standard_head_dim,
                                                        cfg.rope_dim, base_pos, cfg.rope_theta));

            const uint32_t per_pos = standard_n_kv_heads * standard_head_dim;
            require_status(backend_.kv_append_batch(*k_cache_[il], *k_batch_, base_pos, per_pos, batch));
            require_status(backend_.kv_append_batch(*v_cache_[il], *v_batch_, base_pos, per_pos, batch));
            if (record_ops) record(report, "layer." + std::to_string(il) + ".kv_append_batch");

            const float scale = 1.0f / std::sqrt(static_cast<float>(standard_head_dim));
            require_status(backend_.attention_decode_batch(*mid_batch_, *q_batch_,
                                                            2 * standard_head_dim,
                                                            *k_cache_[il], *v_cache_[il],
                                                            standard_n_heads, standard_n_kv_heads,
                                                            standard_head_dim,
                                                            base_pos, batch,
                                                            q_stride_buf, mid_stride, scale));
            require_status(backend_.apply_attn_gate_batch(*mid_batch_, *q_batch_,
                                                           2 * standard_head_dim,
                                                           batch, q_stride_buf, mid_stride,
                                                           standard_n_heads, standard_head_dim));
            if (record_ops) record(report, "layer." + std::to_string(il) + ".attention_sdpa_batch");

            require_status(backend_.q8_0_matmul(*attn_out_batch_, *layer.attn_output, *mid_batch_,
                                                 batch, mid_stride, h_stride));
        }

        require_status(backend_.add(*h_batch_, *h_batch_, *attn_out_batch_));
        if (record_ops) record(report, "layer." + std::to_string(il) + ".attn_residual_batch");

        require_status(backend_.rms_norm_batch(*norm_batch_, *h_batch_, *layer.ffn_norm,
                                                batch, h_stride, eps));
        if (record_ops) record(report, "layer." + std::to_string(il) + ".ffn_norm_batch");

        require_status(backend_.q8_0_matmul(*ffn_gate_batch_, *layer.ffn_gate, *norm_batch_,
                                             batch, h_stride, ffn_stride));
        require_status(backend_.q8_0_matmul(*ffn_up_batch_, *layer.ffn_up, *norm_batch_,
                                             batch, h_stride, ffn_stride));
        require_status(backend_.silu_mul(*ffn_mid_batch_, *ffn_gate_batch_, *ffn_up_batch_));
        require_status(backend_.q8_0_matmul(*ffn_out_batch_, *layer.ffn_down, *ffn_mid_batch_,
                                             batch, ffn_stride, h_stride));
        require_status(backend_.add(*h_batch_, *h_batch_, *ffn_out_batch_));
        if (record_ops) record(report, "layer." + std::to_string(il) + ".ffn_batch");
        }

        if (try_capture) {
            require_status(backend_.end_capture());
            require_status(backend_.replay_graph());
        } else if (full_chunk) {
            // First full chunk: ran eager so backend-side scratch buffers
            // (q8_1 staging, fattn workspace) get sized before we attempt
            // capture on the next chunk.
            capture_warmup_pending = false;
        }
    }

    // Only the LAST prompt token's logits are needed to start decoding. The
    // last chunk leaves its rows in h_batch_ rows [0, last_chunk_batch).
    require_status(backend_.copy_d2d(*h_, *h_batch_, (last_chunk_batch - 1) * h_stride, h_stride));
    require_status(backend_.rms_norm(*norm_, *h_, weights_.output_norm(), eps));
    record(report, "output_norm");
    require_status(backend_.q8_0_matvec(*logits_, weights_.output(), *norm_));
    const DeviceArgmax best = backend_.argmax(*logits_);
    require_status(backend_.end());

    position_ += total;
    report.argmax_token = best.token;
    report.argmax_logit = best.logit;
    report.argmax_text = model_.gguf().token_text(static_cast<uint32_t>(best.token));
    record(report, "lm_head_argmax");
    report.ok = true;
    return report;
}

bool QwenExecutor::copy_last_logits(std::vector<float> &out) const {
    if (!logits_) return false;
    out.resize(static_cast<size_t>(logits_->count));
    const DeviceStatus st = backend_.copy_to_host(*logits_, out.data(), 0, logits_->count);
    return st.ok;
}

} // namespace qw3
