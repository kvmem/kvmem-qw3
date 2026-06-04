#include "qwen_executor.hpp"
#include "env_flags.hpp"

#include <algorithm>
#include <chrono>
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

double steady_seconds() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

bool executor_trace_timing_enabled() {
    return env_flag_enabled("QW3_DECODE_TRACE") ||
           env_flag_enabled("QW3_EXECUTOR_TRACE") ||
           env_flag_enabled("QW3_MTP_VERIFY_TRACE");
}

bool full_executor_trace_enabled() {
    return env_flag_enabled("QW3_EXECUTOR_TRACE") ||
           env_flag_enabled("QW3_MTP_VERIFY_TRACE");
}

bool mtp_prefix_batch_enabled() {
    return env_flag_enabled("QW3_MTP_PREFIX_BATCH", true);
}

uint32_t mtp_prefix_batch_min_tokens() {
    return env_uint32_or("QW3_MTP_PREFIX_BATCH_MIN", 32);
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
    mtp_prefix_len_ = 0;
    decode_graph_warmup_pending_ = true;
}

void QwenExecutor::begin_record_timing(bool enabled) const {
    trace_last_seconds_ = 0.0;
    if (!enabled) return;
    require_status(backend_.synchronize());
    trace_last_seconds_ = steady_seconds();
}

void QwenExecutor::record(NativeExecutorReport &report, const std::string &op) const {
    if (trace_last_seconds_ > 0.0) {
        require_status(backend_.synchronize());
        const double now = steady_seconds();
        report.elapsed_us.push_back((now - trace_last_seconds_) * 1.0e6);
        trace_last_seconds_ = now;
    }
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
    if (const QwenMtpWeights *mtp = weights_.mtp()) {
        const QwenLayerWeights &l = mtp->layer;
        if (l.ffn_dim > max_ffn) max_ffn = l.ffn_dim;
        if (l.q_rows > max_q) max_q = l.q_rows;
        if (l.k_rows > max_k) max_k = l.k_rows;
        if (l.v_rows > max_v) max_v = l.v_rows;
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

void QwenExecutor::ensure_mtp_scratch() {
    ensure_scratch();
    if (mtp_scratch_ready_) return;
    if (!weights_.mtp()) return;

    const QwenConfig &cfg = model_.config();
    mtp_h_ = backend_.tensor_f32(cfg.n_embd, "mtp_h");
    mtp_embd_ = backend_.tensor_f32(cfg.n_embd, "mtp_embd");
    mtp_enorm_ = backend_.tensor_f32(cfg.n_embd, "mtp_enorm");
    mtp_hnorm_ = backend_.tensor_f32(cfg.n_embd, "mtp_hnorm");
    mtp_concat_ = backend_.tensor_f32(static_cast<uint64_t>(2) * cfg.n_embd, "mtp_concat");
    mtp_zero_h_ = backend_.tensor_f32(cfg.n_embd, "mtp_zero_h");
    mtp_prefix_h_ = backend_.tensor_f32(cfg.n_embd, "mtp_prefix_h");
    (void) backend_.zero_tensor(*mtp_zero_h_);

    const uint64_t kv_per_pos = static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
    const uint64_t kv_slots = std::max<uint32_t>(kv_ctx_size_, 1);
    const char *kv_dtype_env = std::getenv("QW3_KV_DTYPE");
    const bool kv_use_fp16 = !(kv_dtype_env && std::strcmp(kv_dtype_env, "fp32") == 0);
    if (kv_use_fp16) {
        mtp_k_cache_ = backend_.tensor_f16(kv_per_pos * kv_slots, "mtp_k_cache");
        mtp_v_cache_ = backend_.tensor_f16(kv_per_pos * kv_slots, "mtp_v_cache");
    } else {
        mtp_k_cache_ = backend_.tensor_f32(kv_per_pos * kv_slots, "mtp_k_cache");
        mtp_v_cache_ = backend_.tensor_f32(kv_per_pos * kv_slots, "mtp_v_cache");
    }
    mtp_scratch_ready_ = true;
}

void QwenExecutor::ensure_mtp_batch_scratch(uint32_t batch) {
    ensure_mtp_scratch();
    if (batch == 0 || batch <= mtp_batch_capacity_) return;
    const QwenMtpWeights *mtp = weights_.mtp();
    if (!mtp) return;
    const QwenConfig &cfg = model_.config();
    const QwenLayerWeights &layer = mtp->layer;
    const uint64_t B = batch;
    mtp_h_input_batch_ = backend_.scratch_f32(B * cfg.n_embd, "mtp_h_input_batch");
    mtp_h_batch_ = backend_.scratch_f32(B * cfg.n_embd, "mtp_h_batch");
    mtp_norm_batch_ = backend_.scratch_f32(B * cfg.n_embd, "mtp_norm_batch");
    mtp_concat_batch_ = backend_.scratch_f32(B * static_cast<uint64_t>(2) * cfg.n_embd,
                                             "mtp_concat_batch");
    mtp_q_batch_ = backend_.scratch_f32(B * std::max<uint64_t>(layer.q_rows, 1), "mtp_q_batch");
    mtp_k_batch_ = backend_.scratch_f32(B * std::max<uint64_t>(layer.k_rows, 1), "mtp_k_batch");
    mtp_v_batch_ = backend_.scratch_f32(B * std::max<uint64_t>(layer.v_rows, 1), "mtp_v_batch");
    mtp_mid_batch_ = backend_.scratch_f32(B * static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim,
                                          "mtp_mid_batch");
    mtp_ffn_gate_batch_ = backend_.scratch_f32(B * std::max<uint64_t>(layer.ffn_dim, 1),
                                               "mtp_ffn_gate_batch");
    mtp_ffn_up_batch_ = backend_.scratch_f32(B * std::max<uint64_t>(layer.ffn_dim, 1),
                                             "mtp_ffn_up_batch");
    mtp_ffn_mid_batch_ = backend_.scratch_f32(B * std::max<uint64_t>(layer.ffn_dim, 1),
                                              "mtp_ffn_mid_batch");
    mtp_ffn_out_batch_ = backend_.scratch_f32(B * cfg.n_embd, "mtp_ffn_out_batch");
    mtp_batch_capacity_ = batch;
}

void QwenExecutor::ensure_logits_batch_scratch(uint32_t batch) {
    if (batch == 0 || batch <= logits_batch_capacity_) return;
    const uint64_t vocab = weights_.output().rows;
    logits_batch_ = backend_.scratch_f32(static_cast<uint64_t>(batch) * vocab,
                                         "logits_batch");
    logits_batch_capacity_ = batch;
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
    begin_record_timing(full_executor_trace_enabled());
    require_status(backend_.q8_0_get_row(*h_, weights_.token_embd(), token_id));
    record(report, "token_embedding_lookup");
    require_status(backend_.rms_norm(*norm_, *h_, weights_.output_norm(), model_.config().rms_eps));
    record(report, "output_norm");
    require_status(backend_.end());

    report.ok = true;
    return report;
}

NativeExecutorReport QwenExecutor::forward_one_token(uint32_t token_id,
                                                     bool compute_logits) {
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
    begin_record_timing(executor_trace_timing_enabled());

    // CUDA-graph capture path: skip on the first token (warm-up: every
    // backend-side scratch buffer needs to be sized before we record
    // pointers into a graph). Also disabled whenever we're inside an MTP
    // verify/replay pass (compute_logits == false) — the captured topology
    // assumes the full LM-head argmax tail runs, which the no-logits path
    // skips, so re-using a stale graph would be incorrect.
    const bool try_capture = compute_logits &&
        !decode_graph_warmup_pending_ && backend_.begin_capture();

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

    if (!compute_logits) {
        // MTP verify/replay re-runs the target model only to advance KV +
        // recurrent state; the LM-head argmax tail is skipped. Note: graph
        // capture is suppressed in this path (try_capture is false), so the
        // eager begin/end pairing here is always correct.
        require_status(backend_.end());
        position_++;
        report.ok = true;
        return report;
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

NativeExecutorReport QwenExecutor::forward_n_tokens(const std::vector<uint32_t> &tokens,
                                                    bool compute_logits,
                                                    std::vector<DeviceArgmax> *row_argmaxes,
                                                    StateCheckpointSet *state_checkpoints,
                                                    uint32_t state_checkpoint_count,
                                                    bool copy_last_logits) {
    NativeExecutorReport report;
    if (tokens.empty()) { report.ok = true; return report; }
    const NativePlanInfo &plan = model_.plan();
    if (!plan.supported) {
        report.missing_kernels.push_back("native model plan is incomplete");
        return report;
    }
    ensure_scratch();
    const uint32_t total = static_cast<uint32_t>(tokens.size());

    // MTP verify/replay requires the whole batch to live in h_batch_ at the
    // tail (per-row argmax) and consistent checkpoint base positions, so it
    // must run as a single chunk. The verifier batch is tiny (chain length,
    // typically <= 8), so this never grows peak memory.
    const bool mtp_single_chunk = (row_argmaxes != nullptr) || (state_checkpoints != nullptr);

    // Prefill chunking. The chunk cap controls peak compute scratch (per-token
    // batch tensors: residual, FFN gate/up, q/k/v projections, etc.). qw3
    // originally sized batch scratch to the entire prompt, which made peak
    // memory grow linearly with T — at T=64K the per-prompt batch scratch
    // alone exceeded 30 GiB of FP32 storage. Capping the chunk holds peak
    // memory roughly flat in T.
    //
    // The cap is 2048: empirically this recovers most of the chunking
    // throughput tax (vs whole-prompt) while keeping peak memory close to
    // chunk=512 (within ~1.1 GiB at T=65K). Smaller chunks pay a per-chunk
    // amortization tax (HGEMM autotuner restart, MMQ-at-short-batch dispatch,
    // sub-saturation grids) without buying meaningful memory back; larger
    // chunks (≥4096) re-grow the per-chunk scratch significantly.
    //
    // Override with QW3_PREFILL_CHUNK=N. Set N=0 to disable the cap entirely
    // (whole-prompt batch — original behavior, useful for benchmarking the
    // throughput tax of chunking itself).
    constexpr uint32_t kQw3DefaultPrefillChunk = 2048;
    uint32_t chunk_size = std::min<uint32_t>(kQw3DefaultPrefillChunk, total);
    // CLI override (`--prefill-chunk N`) takes precedence over env. -1 means
    // "no override, use env or default".
    if (prefill_chunk_override_ >= 0) {
        if (prefill_chunk_override_ == 0) {
            chunk_size = total;  // whole-prompt batch
        } else {
            chunk_size = static_cast<uint32_t>(prefill_chunk_override_);
        }
    } else if (const char *env = std::getenv("QW3_PREFILL_CHUNK")) {
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
    if (mtp_single_chunk) chunk_size = total;  // MTP verify: never split
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
    begin_record_timing(full_executor_trace_enabled());

    // MTP rollback support: when state_checkpoints is requested, the recurrent
    // layers below snapshot their per-token DeltaNet + conv state into the
    // checkpoint slots so the orchestrator can roll back to any accepted
    // verifier row. Only meaningful in the single-chunk MTP path.
    const uint32_t save_state_checkpoints =
        state_checkpoints ? std::min<uint32_t>(state_checkpoint_count, total) : 0;
    if (state_checkpoints) {
        state_checkpoints->ready = false;
        state_checkpoints->base_position = position_;
        state_checkpoints->count = save_state_checkpoints;
        state_checkpoints->h_stride = h_stride;
        if (state_checkpoints->recurrent_states.size() != recurrent_states_.size()) {
            state_checkpoints->recurrent_states.resize(recurrent_states_.size());
        }
        if (state_checkpoints->conv_states.size() != conv_states_.size()) {
            state_checkpoints->conv_states.resize(conv_states_.size());
        }
    }

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
            {
                DeviceTensor *outs[4] = {proj_batch_.get(), gate_proj_batch_.get(),
                                         alpha_batch_.get(), beta_batch_.get()};
                const DeviceWeight *ws[4] = {layer.attn_qkv, layer.attn_gate,
                                             layer.ssm_alpha, layer.ssm_beta};
                const uint32_t strides[4] = {proj_stride, gate_proj_stride,
                                             alpha_stride, beta_stride};
                require_status(backend_.q8_0_matmul_fanout(outs, ws, strides, 4,
                                                           *norm_batch_, batch, h_stride));
            }
            if (record_ops) record(report, "layer." + std::to_string(il) + ".recurrent_projections_batch");
            if (!recurrent_states_[il] || !conv_states_[il] || !conv_out_batch_) {
                throw std::runtime_error("recurrent state not allocated for layer " + std::to_string(il));
            }
            DeviceTensor *state_checkpoint = nullptr;
            DeviceTensor *conv_state_checkpoint = nullptr;
            if (save_state_checkpoints > 0) {
                const uint64_t recurrent_count =
                    recurrent_states_[il]->count * save_state_checkpoints;
                if (!state_checkpoints->recurrent_states[il] ||
                    state_checkpoints->recurrent_states[il]->count != recurrent_count) {
                    state_checkpoints->recurrent_states[il] =
                        backend_.scratch_f32(recurrent_count,
                                             "mtp_state_checkpoint_recurrent");
                }
                const uint64_t conv_count =
                    conv_states_[il]->count * save_state_checkpoints;
                if (!state_checkpoints->conv_states[il] ||
                    state_checkpoints->conv_states[il]->count != conv_count) {
                    state_checkpoints->conv_states[il] =
                        backend_.scratch_f32(conv_count,
                                             "mtp_state_checkpoint_conv");
                }
                state_checkpoint = state_checkpoints->recurrent_states[il].get();
                conv_state_checkpoint = state_checkpoints->conv_states[il].get();
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
                                                     eps,
                                                     state_checkpoint,
                                                     conv_state_checkpoint,
                                                     save_state_checkpoints));
            if (record_ops) record(report, "layer." + std::to_string(il) + ".deltanet_batch");
            require_status(backend_.q8_0_matmul(*attn_out_batch_, *layer.ssm_out, *core_batch_,
                                                 batch, core_stride, h_stride));
            if (record_ops) record(report, "layer." + std::to_string(il) + ".recurrent_output_batch");
        } else {
            {
                DeviceTensor *outs[3] = {q_batch_.get(), k_batch_.get(), v_batch_.get()};
                const DeviceWeight *ws[3] = {layer.attn_q, layer.attn_k, layer.attn_v};
                const uint32_t strides[3] = {q_stride_buf, k_stride_buf, v_stride_buf};
                require_status(backend_.q8_0_matmul_fanout(outs, ws, strides, 3,
                                                           *norm_batch_, batch, h_stride));
            }
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
            if (mtp_single_chunk) {
                require_status(backend_.attention_decode_batch_gated(*mid_batch_, *q_batch_,
                                                                      2 * standard_head_dim,
                                                                      *k_cache_[il], *v_cache_[il],
                                                                      standard_n_heads, standard_n_kv_heads,
                                                                      standard_head_dim,
                                                                      base_pos, batch,
                                                                      q_stride_buf, mid_stride, scale));
            } else {
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
            }
            if (record_ops) record(report, "layer." + std::to_string(il) + ".attention_sdpa_batch");

            require_status(backend_.q8_0_matmul(*attn_out_batch_, *layer.attn_output, *mid_batch_,
                                                 batch, mid_stride, h_stride));
        }

        require_status(backend_.add_n(*h_batch_, *h_batch_, *attn_out_batch_,
                                      static_cast<uint64_t>(batch) * h_stride));
        if (record_ops) record(report, "layer." + std::to_string(il) + ".attn_residual_batch");

        require_status(backend_.rms_norm_batch(*norm_batch_, *h_batch_, *layer.ffn_norm,
                                                batch, h_stride, eps));
        if (record_ops) record(report, "layer." + std::to_string(il) + ".ffn_norm_batch");

        require_status(backend_.q8_0_matmul(*ffn_gate_batch_, *layer.ffn_gate, *norm_batch_,
                                             batch, h_stride, ffn_stride));
        require_status(backend_.q8_0_matmul(*ffn_up_batch_, *layer.ffn_up, *norm_batch_,
                                             batch, h_stride, ffn_stride));
        // Batched scratch buffers are capacity-sized; only the active `batch`
        // rows hold valid data. silu_mul/add over the full o.count would
        // process batch_capacity_ rows (e.g. the 2048-wide prefill chunk) for
        // a 2..5-row verify batch — that elementwise overhead, not the matmul,
        // was the verify FFN's dominant cost. Cap to batch rows via *_n.
        require_status(backend_.silu_mul_n(*ffn_mid_batch_, *ffn_gate_batch_, *ffn_up_batch_,
                                           static_cast<uint64_t>(batch) * ffn_stride));
        require_status(backend_.q8_0_matmul(*ffn_out_batch_, *layer.ffn_down, *ffn_mid_batch_,
                                             batch, ffn_stride, h_stride));
        require_status(backend_.add_n(*h_batch_, *h_batch_, *ffn_out_batch_,
                                      static_cast<uint64_t>(batch) * h_stride));
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
    if (state_checkpoints && save_state_checkpoints > 0) {
        state_checkpoints->ready = true;
    }

    // MTP verify/replay: caller only wants KV + recurrent state advanced (and
    // optionally checkpoints), no logits at all.
    if (!compute_logits && !row_argmaxes) {
        position_ += total;
        require_status(backend_.end());
        report.ok = true;
        return report;
    }

    // MTP verify: per-row argmax over every token in the batch. Used by the
    // speculative verifier to compare drafted tokens against the target's
    // greedy continuation row-by-row.
    if (row_argmaxes) {
        const uint64_t vocab = weights_.output().rows;
        ensure_logits_batch_scratch(total);
        require_status(backend_.rms_norm_batch(*norm_batch_, *h_batch_, weights_.output_norm(),
                                                total, h_stride, eps));
        record(report, "output_norm_batch");
        require_status(backend_.q8_0_matmul(*logits_batch_, weights_.output(),
                                             *norm_batch_, total, h_stride,
                                             static_cast<uint32_t>(vocab)));
        record(report, "lm_head_batch");

        require_status(backend_.argmax_batch(*logits_batch_, total,
                                             static_cast<uint32_t>(vocab),
                                             *row_argmaxes));
        if (copy_last_logits) {
            require_status(backend_.copy_d2d(*logits_, *logits_batch_,
                                             static_cast<uint64_t>(total - 1) * vocab,
                                             vocab));
        }
        require_status(backend_.end());

        position_ += total;
        const DeviceArgmax &best = row_argmaxes->back();
        report.argmax_token = best.token;
        report.argmax_logit = best.logit;
        report.argmax_text = model_.gguf().token_text(static_cast<uint32_t>(best.token));
        record(report, "lm_head_argmax_batch");
        report.ok = true;
        return report;
    }

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

// ===========================================================================
// MTP (Multi-Token Prediction) draft head + speculative-decode plumbing.
// Ported from qw3_ly, adapted to qw3's backend op signatures (3-arg
// q8_0_matvec_add / 4-arg q8_0_matvec_silu_mul with eager fallback).
// ===========================================================================

NativeExecutorReport QwenExecutor::forward_mtp_draft(uint32_t token_id) {
    ensure_mtp_scratch();
    return forward_mtp_draft_from(token_id, *h_, position_, 0, 1);
}

std::vector<NativeExecutorReport> QwenExecutor::forward_mtp_draft_chain(uint32_t token_id,
                                                                        uint32_t max_tokens) {
    std::vector<NativeExecutorReport> reports;
    if (max_tokens == 0) return reports;
    ensure_mtp_scratch();
    uint32_t current = token_id;
    for (uint32_t i = 0; i < max_tokens; ++i) {
        if (i >= kv_ctx_size_) break;
        const DeviceTensor &h_input = (i == 0) ? *h_ : *mtp_h_;
        NativeExecutorReport report = forward_mtp_draft_from(current, h_input,
                                                             position_ + i, i, i + 1);
        const int next = report.argmax_token;
        reports.push_back(std::move(report));
        if (next < 0) break;
        current = static_cast<uint32_t>(next);
    }
    return reports;
}

std::vector<NativeExecutorReport> QwenExecutor::forward_mtp_draft_chain_with_prefix(uint32_t token_id,
                                                                                   uint32_t max_tokens) {
    std::vector<NativeExecutorReport> reports;
    if (max_tokens == 0) return reports;
    ensure_mtp_scratch();
    if (position_ > mtp_prefix_len_) {
        NativeExecutorReport report;
        report.missing_kernels.push_back("native MTP prefix KV is behind target position");
        reports.push_back(std::move(report));
        return reports;
    }
    uint32_t current = token_id;
    for (uint32_t i = 0; i < max_tokens; ++i) {
        const uint32_t cache_pos = position_ + i;
        if (cache_pos >= kv_ctx_size_) break;
        const DeviceTensor &h_input = (i == 0) ? *h_ : *mtp_h_;
        NativeExecutorReport report = forward_mtp_draft_from(current, h_input,
                                                             cache_pos, cache_pos,
                                                             cache_pos + 1);
        const bool ok = report.ok;
        const int next = report.argmax_token;
        reports.push_back(std::move(report));
        if (!ok || next < 0) break;
        if (i == 0) {
            mtp_prefix_len_ = std::max<uint32_t>(mtp_prefix_len_, position_ + 1);
        }
        current = static_cast<uint32_t>(next);
    }
    return reports;
}

std::vector<NativeExecutorReport> QwenExecutor::forward_mtp_draft_chain_with_prefix_device(uint32_t token_id,
                                                                                           uint32_t max_tokens) {
    if (max_tokens == 0) return {};
    ensure_mtp_scratch();
    if (!mtp_draft_argmaxes_ || mtp_draft_argmax_capacity_ < max_tokens) {
        mtp_draft_argmaxes_ = backend_.argmax_buffer(max_tokens);
        mtp_draft_argmax_capacity_ = mtp_draft_argmaxes_ ? max_tokens : 0;
    }
    if (!mtp_draft_argmaxes_) {
        return forward_mtp_draft_chain_with_prefix(token_id, max_tokens);
    }
    if (position_ > mtp_prefix_len_) {
        NativeExecutorReport report;
        report.missing_kernels.push_back("native MTP prefix KV is behind target position");
        return {std::move(report)};
    }

    std::vector<NativeExecutorReport> reports;
    reports.reserve(max_tokens);
    for (uint32_t i = 0; i < max_tokens; ++i) {
        const uint32_t cache_pos = position_ + i;
        if (cache_pos >= kv_ctx_size_) break;
        const DeviceTensor &h_input = (i == 0) ? *h_ : *mtp_h_;
        const DeviceArgmaxBuffer *token_source = i == 0 ? nullptr : mtp_draft_argmaxes_.get();
        NativeExecutorReport report = forward_mtp_draft_from(token_id,
                                                             h_input,
                                                             cache_pos,
                                                             cache_pos,
                                                             cache_pos + 1,
                                                             /*compute_logits=*/true,
                                                             mtp_draft_argmaxes_.get(),
                                                             i,
                                                             token_source,
                                                             i == 0 ? 0 : i - 1);
        const bool ok = report.ok;
        reports.push_back(std::move(report));
        if (!ok) break;
        if (i == 0) {
            mtp_prefix_len_ = std::max<uint32_t>(mtp_prefix_len_, position_ + 1);
        }
    }

    if (reports.empty() || !reports.back().ok) return reports;
    std::vector<DeviceArgmax> host(reports.size());
    if (auto st = backend_.copy_argmax_buffer_to_host(*mtp_draft_argmaxes_,
                                                      host.data(),
                                                      static_cast<uint32_t>(host.size()));
        !st.ok) {
        NativeExecutorReport report;
        report.missing_kernels.push_back(st.message);
        return {std::move(report)};
    }
    for (size_t i = 0; i < reports.size(); ++i) {
        reports[i].argmax_token = host[i].token;
        reports[i].argmax_logit = host[i].logit;
    }
    return reports;
}

NativeExecutorReport QwenExecutor::forward_mtp_draft_from(uint32_t token_id,
                                                          const DeviceTensor &h_input,
                                                          uint32_t rope_pos,
                                                          uint32_t cache_pos,
                                                          uint32_t seq_len,
                                                          bool compute_logits,
                                                          DeviceArgmaxBuffer *argmax_out,
                                                          uint32_t argmax_out_index,
                                                          const DeviceArgmaxBuffer *token_source,
                                                          uint32_t token_source_index) {
    NativeExecutorReport report;
    const NativePlanInfo &plan = model_.plan();
    if (!plan.mtp_supported) {
        report.missing_kernels.push_back("native MTP plan is incomplete");
        return report;
    }
    const QwenMtpWeights *mtp = weights_.mtp();
    if (!mtp || !mtp->eh_proj || !mtp->embed_tokens || !mtp->enorm ||
        !mtp->hnorm || !mtp->shared_head_head || !mtp->shared_head_norm) {
        report.missing_kernels.push_back("native MTP weights are incomplete");
        return report;
    }
    if (mtp->layer.recurrent) {
        report.missing_kernels.push_back("native MTP recurrent draft layer is not supported");
        return report;
    }
    ensure_mtp_scratch();
    if (seq_len == 0 || cache_pos >= kv_ctx_size_ || seq_len > kv_ctx_size_) {
        report.missing_kernels.push_back("native MTP KV cache is too small for requested draft chain");
        return report;
    }

    const QwenConfig &cfg = model_.config();
    const uint32_t standard_head_dim = cfg.head_dim;
    const uint32_t standard_n_heads = cfg.n_heads;
    const uint32_t standard_n_kv_heads = cfg.n_kv_heads;
    const float eps = cfg.rms_eps;
    const QwenLayerWeights &layer = mtp->layer;

    require_status(backend_.begin());
    begin_record_timing(executor_trace_timing_enabled());

    if (token_source) {
        require_status(backend_.q8_0_get_row_from_argmax(*mtp_embd_,
                                                         *mtp->embed_tokens,
                                                         *token_source,
                                                         token_source_index));
    } else {
        require_status(backend_.q8_0_get_row(*mtp_embd_, *mtp->embed_tokens, token_id));
    }
    record(report, "mtp.token_embedding_lookup");
    require_status(backend_.rms_norm(*mtp_enorm_, *mtp_embd_, *mtp->enorm, eps));
    record(report, "mtp.enorm");
    require_status(backend_.rms_norm(*mtp_hnorm_, h_input, *mtp->hnorm, eps));
    record(report, "mtp.hnorm");
    require_status(backend_.copy_d2d_into(*mtp_concat_, 0, *mtp_enorm_, 0, cfg.n_embd));
    require_status(backend_.copy_d2d_into(*mtp_concat_, cfg.n_embd, *mtp_hnorm_, 0, cfg.n_embd));
    record(report, "mtp.concat");
    require_status(backend_.q8_0_matvec(*mtp_h_, *mtp->eh_proj, *mtp_concat_));
    record(report, "mtp.eh_proj");

    require_status(backend_.rms_norm(*norm_, *mtp_h_, *layer.attn_norm, eps));
    record(report, "mtp.attn_norm");
    {
        DeviceTensor *outs[3] = {q_.get(), k_.get(), v_.get()};
        const DeviceWeight *ws[3] = {layer.attn_q, layer.attn_k, layer.attn_v};
        require_status(backend_.q8_0_matvec_fanout(outs, ws, 3, *norm_));
    }
    record(report, "mtp.attention_qkv_projection");
    require_status(backend_.rmsnorm_per_head(*q_, *layer.attn_q_norm,
                                             standard_n_heads,
                                             2 * standard_head_dim,
                                             standard_head_dim, eps));
    require_status(backend_.rmsnorm_per_head(*k_, *layer.attn_k_norm,
                                             standard_n_kv_heads,
                                             standard_head_dim,
                                             standard_head_dim, eps));
    require_status(backend_.rope_partial(*q_, standard_n_heads,
                                         2 * standard_head_dim,
                                         cfg.rope_dim, rope_pos, cfg.rope_theta));
    require_status(backend_.rope_partial(*k_, standard_n_kv_heads,
                                         standard_head_dim,
                                         cfg.rope_dim, rope_pos, cfg.rope_theta));

    const uint32_t per_pos = standard_n_kv_heads * standard_head_dim;
    require_status(backend_.kv_append(*mtp_k_cache_, *k_, cache_pos, per_pos));
    require_status(backend_.kv_append(*mtp_v_cache_, *v_, cache_pos, per_pos));
    record(report, "mtp.kv_append");

    const float scale = 1.0f / std::sqrt(static_cast<float>(standard_head_dim));
    require_status(backend_.attention_decode(*mid_, *scores_, *q_,
                                             2 * standard_head_dim,
                                             *mtp_k_cache_, *mtp_v_cache_,
                                             standard_n_heads, standard_n_kv_heads,
                                             standard_head_dim,
                                             seq_len, scale));
    require_status(backend_.apply_attn_gate(*mid_, *q_,
                                             2 * standard_head_dim,
                                             standard_n_heads, standard_head_dim));
    record(report, "mtp.attention_sdpa");
    // h += W_out * mid (fused matvec+add with eager fallback, mirroring the
    // main decode path).
    if (auto st = backend_.q8_0_matvec_add(*mtp_h_, *layer.attn_output, *mid_); !st.ok) {
        require_status(backend_.q8_0_matvec(*attn_out_, *layer.attn_output, *mid_));
        require_status(backend_.add(*mtp_h_, *mtp_h_, *attn_out_));
    }
    record(report, "mtp.attn_residual");

    require_status(backend_.rms_norm(*norm_, *mtp_h_, *layer.ffn_norm, eps));
    record(report, "mtp.ffn_norm");
    if (auto st = backend_.q8_0_matvec_silu_mul(*ffn_mid_, *layer.ffn_gate,
                                                *layer.ffn_up, *norm_);
        !st.ok) {
        DeviceTensor *outs[2] = {ffn_gate_.get(), ffn_up_.get()};
        const DeviceWeight *ws[2] = {layer.ffn_gate, layer.ffn_up};
        require_status(backend_.q8_0_matvec_fanout(outs, ws, 2, *norm_));
        require_status(backend_.silu_mul(*ffn_mid_, *ffn_gate_, *ffn_up_));
    }
    if (auto st = backend_.q8_0_matvec_add(*mtp_h_, *layer.ffn_down, *ffn_mid_); !st.ok) {
        require_status(backend_.q8_0_matvec(*ffn_out_, *layer.ffn_down, *ffn_mid_));
        require_status(backend_.add(*mtp_h_, *mtp_h_, *ffn_out_));
    }
    record(report, "mtp.ffn");

    if (!compute_logits) {
        require_status(backend_.end());
        report.ok = true;
        return report;
    }

    require_status(backend_.rms_norm(*norm_, *mtp_h_, *mtp->shared_head_norm, eps));
    record(report, "mtp.shared_head_norm");
    require_status(backend_.q8_0_matvec(*logits_, *mtp->shared_head_head, *norm_));
    DeviceArgmax best;
    if (argmax_out) {
        require_status(backend_.argmax_to_buffer(*logits_, *argmax_out, argmax_out_index));
    } else {
        best = backend_.argmax(*logits_);
    }
    require_status(backend_.end());

    if (!argmax_out) {
        report.argmax_token = best.token;
        report.argmax_logit = best.logit;
        report.argmax_text = model_.gguf().token_text(static_cast<uint32_t>(best.token));
    }
    record(report, "mtp.lm_head_argmax");
    report.ok = true;
    return report;
}

void QwenExecutor::commit_mtp_prefix(uint32_t prefix_len) {
    mtp_prefix_len_ = std::max<uint32_t>(mtp_prefix_len_,
                                         std::min<uint32_t>(prefix_len, kv_ctx_size_));
}

void QwenExecutor::commit_mtp_prefix_from_current_hidden(uint32_t prefix_len) {
    ensure_mtp_scratch();
    if (h_ && mtp_prefix_h_) {
        require_status(backend_.copy_d2d(*mtp_prefix_h_, *h_, 0, h_->count));
    }
    commit_mtp_prefix(prefix_len);
}

QwenExecutor::StateSnapshot QwenExecutor::snapshot_state() {
    StateSnapshot snapshot;
    capture_state(snapshot);
    return snapshot;
}

void QwenExecutor::capture_state(StateSnapshot &snapshot) {
    ensure_scratch();
    snapshot.position = position_;
    if (h_) {
        if (!snapshot.h || snapshot.h->count != h_->count) {
            snapshot.h = backend_.scratch_f32(h_->count, "snapshot_h");
        }
        require_status(backend_.copy_d2d(*snapshot.h, *h_, 0, h_->count));
    }
    if (snapshot.recurrent_states.size() != recurrent_states_.size()) {
        snapshot.recurrent_states.resize(recurrent_states_.size());
    }
    if (snapshot.conv_states.size() != conv_states_.size()) {
        snapshot.conv_states.resize(conv_states_.size());
    }
    for (size_t i = 0; i < recurrent_states_.size(); ++i) {
        if (recurrent_states_[i]) {
            if (!snapshot.recurrent_states[i] ||
                snapshot.recurrent_states[i]->count != recurrent_states_[i]->count) {
                snapshot.recurrent_states[i] =
                    backend_.scratch_f32(recurrent_states_[i]->count, "snapshot_recurrent");
            }
            require_status(backend_.copy_d2d(*snapshot.recurrent_states[i],
                                             *recurrent_states_[i],
                                             0, recurrent_states_[i]->count));
        }
        if (conv_states_[i]) {
            if (!snapshot.conv_states[i] ||
                snapshot.conv_states[i]->count != conv_states_[i]->count) {
                snapshot.conv_states[i] =
                    backend_.scratch_f32(conv_states_[i]->count, "snapshot_conv");
            }
            require_status(backend_.copy_d2d(*snapshot.conv_states[i],
                                             *conv_states_[i],
                                             0, conv_states_[i]->count));
        }
    }
    snapshot.ready = true;
}

void QwenExecutor::restore_state(const StateSnapshot &snapshot) {
    if (!snapshot.ready) {
        throw std::runtime_error("cannot restore an empty QwenExecutor snapshot");
    }
    ensure_scratch();
    if (snapshot.h && h_) {
        require_status(backend_.copy_d2d(*h_, *snapshot.h, 0, h_->count));
    }
    for (size_t i = 0; i < recurrent_states_.size(); ++i) {
        if (recurrent_states_[i] && i < snapshot.recurrent_states.size() &&
            snapshot.recurrent_states[i]) {
            require_status(backend_.copy_d2d(*recurrent_states_[i],
                                             *snapshot.recurrent_states[i],
                                             0, recurrent_states_[i]->count));
        }
        if (conv_states_[i] && i < snapshot.conv_states.size() &&
            snapshot.conv_states[i]) {
            require_status(backend_.copy_d2d(*conv_states_[i],
                                             *snapshot.conv_states[i],
                                             0, conv_states_[i]->count));
        }
    }
    position_ = snapshot.position;
}

void QwenExecutor::restore_state_checkpoint(const StateCheckpointSet &checkpoints,
                                            uint32_t index) {
    if (!checkpoints.ready || index >= checkpoints.count) {
        throw std::runtime_error("cannot restore an empty QwenExecutor checkpoint");
    }
    ensure_scratch();
    if (!h_batch_) {
        throw std::runtime_error("cannot restore checkpoint without batch hidden state");
    }
    if (h_ && checkpoints.h_stride > 0) {
        require_status(backend_.copy_d2d(*h_, *h_batch_,
                                         static_cast<uint64_t>(index) * checkpoints.h_stride,
                                         h_->count));
    }
    for (size_t i = 0; i < recurrent_states_.size(); ++i) {
        if (recurrent_states_[i] && i < checkpoints.recurrent_states.size() &&
            checkpoints.recurrent_states[i]) {
            const uint64_t state_count = recurrent_states_[i]->count;
            require_status(backend_.copy_d2d(*recurrent_states_[i],
                                             *checkpoints.recurrent_states[i],
                                             static_cast<uint64_t>(index) * state_count,
                                             state_count));
        }
        if (conv_states_[i] && i < checkpoints.conv_states.size() &&
            checkpoints.conv_states[i]) {
            const uint64_t state_count = conv_states_[i]->count;
            require_status(backend_.copy_d2d(*conv_states_[i],
                                             *checkpoints.conv_states[i],
                                             static_cast<uint64_t>(index) * state_count,
                                             state_count));
        }
    }
    position_ = checkpoints.base_position + index + 1;
}

NativeExecutorReport QwenExecutor::prime_mtp_prefix_from_current(uint32_t token,
                                                                 uint32_t base_position) {
    NativeExecutorReport report;
    ensure_mtp_scratch();
    if (base_position >= kv_ctx_size_) {
        report.missing_kernels.push_back("native MTP prefix KV cache is too small");
        return report;
    }
    if (base_position > 0 && base_position > mtp_prefix_len_) {
        report.missing_kernels.push_back("native MTP prefix chunks are not contiguous");
        return report;
    }

    const DeviceTensor &h_input = (base_position == 0) ? *mtp_zero_h_ : *mtp_prefix_h_;
    NativeExecutorReport step = forward_mtp_draft_from(token, h_input,
                                                       base_position,
                                                       base_position,
                                                       base_position + 1,
                                                       /*compute_logits=*/false);
    report.ops_executed += step.ops_executed;
    if (!step.ok) {
        report.missing_kernels = std::move(step.missing_kernels);
        return report;
    }
    require_status(backend_.copy_d2d(*mtp_prefix_h_, *h_, 0, h_->count));
    mtp_prefix_len_ = std::max<uint32_t>(mtp_prefix_len_, base_position + 1);
    report.ok = true;
    return report;
}

NativeExecutorReport QwenExecutor::replay_tokens_with_mtp_prefix(
        const std::vector<uint32_t> &tokens,
        uint32_t base_position,
        bool rebuild_prefix,
        double *prefix_seconds,
        uint64_t *prefix_ops) {
    NativeExecutorReport report;
    if (tokens.empty()) {
        report.ok = true;
        return report;
    }

    require_status(backend_.begin());
    bool replay_ok = true;
    try {
        for (uint32_t i = 0; i < tokens.size(); ++i) {
            NativeExecutorReport target =
                forward_one_token(tokens[i], /*compute_logits=*/false);
            report.ops_executed += target.ops_executed;
            if (!target.ok) {
                replay_ok = false;
                report.ok = false;
                report.missing_kernels = std::move(target.missing_kernels);
                break;
            }

            if (rebuild_prefix) {
                const double t0 = steady_seconds();
                NativeExecutorReport prefix =
                    prime_mtp_prefix_from_current(tokens[i], base_position + i);
                if (prefix_seconds) *prefix_seconds += steady_seconds() - t0;
                if (prefix_ops) *prefix_ops += prefix.ops_executed;
                if (!prefix.ok) {
                    replay_ok = false;
                    report.ok = false;
                    report.missing_kernels = std::move(prefix.missing_kernels);
                    break;
                }
            }
        }
        require_status(backend_.end());
    } catch (...) {
        backend_.end();
        throw;
    }

    if (replay_ok && report.missing_kernels.empty()) report.ok = true;
    return report;
}

NativeExecutorReport QwenExecutor::prime_mtp_prefix_from_last_batch(const std::vector<uint32_t> &tokens,
                                                                    uint32_t base_position,
                                                                    uint32_t batch_min_override) {
    NativeExecutorReport report;
    if (tokens.empty()) {
        report.ok = true;
        return report;
    }
    ensure_mtp_scratch();
    if (!h_batch_ || batch_capacity_ == 0 || tokens.size() > batch_capacity_) {
        report.missing_kernels.push_back("native MTP prefix requires the last target batch hidden states");
        return report;
    }
    if (base_position + tokens.size() > kv_ctx_size_) {
        report.missing_kernels.push_back("native MTP prefix KV cache is too small");
        return report;
    }
    if (base_position > 0 && base_position > mtp_prefix_len_) {
        report.missing_kernels.push_back("native MTP prefix chunks are not contiguous");
        return report;
    }

    const uint32_t batch = static_cast<uint32_t>(tokens.size());
    const uint32_t h_stride = static_cast<uint32_t>(h_batch_->count / batch_capacity_);
    auto prime_sequential = [&]() -> NativeExecutorReport {
        NativeExecutorReport seq_report;
        for (uint32_t i = 0; i < batch; ++i) {
            const uint32_t pos = base_position + i;
            const DeviceTensor *h_input = mtp_zero_h_.get();
            if (pos > 0) {
                if (i == 0) {
                    h_input = mtp_prefix_h_.get();
                } else {
                    require_status(backend_.copy_d2d(*mtp_prefix_h_, *h_batch_,
                                                     static_cast<uint64_t>(i - 1) * h_stride,
                                                     h_stride));
                    h_input = mtp_prefix_h_.get();
                }
            }

            NativeExecutorReport step = forward_mtp_draft_from(tokens[i], *h_input,
                                                               pos, pos, pos + 1,
                                                               /*compute_logits=*/false);
            seq_report.ops_executed += step.ops_executed;
            if (!step.ok) {
                seq_report.missing_kernels = std::move(step.missing_kernels);
                return seq_report;
            }
        }
        require_status(backend_.copy_d2d(*mtp_prefix_h_, *h_batch_,
                                         static_cast<uint64_t>(batch - 1) * h_stride,
                                         h_stride));
        mtp_prefix_len_ = std::max<uint32_t>(mtp_prefix_len_, base_position + batch);
        seq_report.ok = true;
        return seq_report;
    };

    const uint32_t batch_min =
        batch_min_override > 0 ? batch_min_override : mtp_prefix_batch_min_tokens();
    if (!mtp_prefix_batch_enabled() || batch < batch_min) {
        return prime_sequential();
    }

    const NativePlanInfo &plan = model_.plan();
    if (!plan.mtp_supported) {
        report.missing_kernels.push_back("native MTP plan is incomplete");
        return report;
    }
    const QwenMtpWeights *mtp = weights_.mtp();
    if (!mtp || !mtp->eh_proj || !mtp->embed_tokens || !mtp->enorm ||
        !mtp->hnorm || !mtp->shared_head_head || !mtp->shared_head_norm) {
        report.missing_kernels.push_back("native MTP weights are incomplete");
        return report;
    }
    if (mtp->layer.recurrent) {
        report.missing_kernels.push_back("native MTP recurrent draft layer is not supported");
        return report;
    }
    ensure_mtp_batch_scratch(batch);
    if (!mtp_h_input_batch_ || !mtp_h_batch_ || !mtp_norm_batch_ ||
        !mtp_concat_batch_ || !mtp_q_batch_ || !mtp_k_batch_ ||
        !mtp_v_batch_ || !mtp_mid_batch_ || !mtp_ffn_gate_batch_ ||
        !mtp_ffn_up_batch_ || !mtp_ffn_mid_batch_ || !mtp_ffn_out_batch_) {
        report.missing_kernels.push_back("native MTP batch prefix scratch allocation failed");
        return report;
    }

    const QwenConfig &cfg = model_.config();
    const QwenLayerWeights &layer = mtp->layer;
    const uint32_t standard_head_dim = cfg.head_dim;
    const uint32_t standard_n_heads = cfg.n_heads;
    const uint32_t standard_n_kv_heads = cfg.n_kv_heads;
    const float eps = cfg.rms_eps;

    auto mtp_row_stride = [this](const DeviceTensor *t) -> uint32_t {
        return static_cast<uint32_t>(t->count / mtp_batch_capacity_);
    };
    const uint32_t mtp_h_stride = mtp_row_stride(mtp_h_batch_.get());
    const uint32_t concat_stride = mtp_row_stride(mtp_concat_batch_.get());
    const uint32_t ffn_stride = mtp_row_stride(mtp_ffn_gate_batch_.get());
    const uint32_t q_stride_buf = mtp_row_stride(mtp_q_batch_.get());
    const uint32_t k_stride_buf = mtp_row_stride(mtp_k_batch_.get());
    const uint32_t v_stride_buf = mtp_row_stride(mtp_v_batch_.get());
    const uint32_t mid_stride = mtp_row_stride(mtp_mid_batch_.get());

    require_status(backend_.begin());
    begin_record_timing(full_executor_trace_enabled());

    DeviceTensor &h_inputs = *mtp_h_input_batch_;
    const DeviceTensor &first_h = (base_position == 0) ? *mtp_zero_h_ : *mtp_prefix_h_;
    require_status(backend_.pack_mtp_prefix_hinputs(h_inputs, first_h, *h_batch_,
                                                    batch, h_stride));
    record(report, "mtp.prefix_hinput_batch");

    std::vector<uint64_t> rows(batch);
    for (uint32_t i = 0; i < batch; ++i) rows[i] = tokens[i];
    require_status(backend_.q8_0_get_rows_batch(*mtp_norm_batch_, *mtp->embed_tokens,
                                                rows.data(), batch));
    record(report, "mtp.token_embedding_lookup_batch");
    require_status(backend_.rms_norm_batch(*mtp_ffn_out_batch_, *mtp_norm_batch_,
                                           *mtp->enorm, batch, h_stride, eps));
    record(report, "mtp.enorm_batch");
    require_status(backend_.rms_norm_batch(*mtp_h_batch_, h_inputs,
                                           *mtp->hnorm, batch, h_stride, eps));
    record(report, "mtp.hnorm_batch");

    require_status(backend_.pack_mtp_concat(*mtp_concat_batch_,
                                            *mtp_ffn_out_batch_,
                                            *mtp_h_batch_,
                                            batch,
                                            h_stride,
                                            mtp_h_stride,
                                            concat_stride,
                                            h_stride));
    record(report, "mtp.concat_batch");
    require_status(backend_.q8_0_matmul(*mtp_h_batch_, *mtp->eh_proj,
                                        *mtp_concat_batch_,
                                        batch, concat_stride, mtp_h_stride));
    record(report, "mtp.eh_proj_batch");

    require_status(backend_.rms_norm_batch(*mtp_norm_batch_, *mtp_h_batch_,
                                           *layer.attn_norm, batch, mtp_h_stride, eps));
    record(report, "mtp.attn_norm_batch");
    {
        DeviceTensor *outs[3] = {mtp_q_batch_.get(), mtp_k_batch_.get(), mtp_v_batch_.get()};
        const DeviceWeight *ws[3] = {layer.attn_q, layer.attn_k, layer.attn_v};
        const uint32_t strides[3] = {q_stride_buf, k_stride_buf, v_stride_buf};
        require_status(backend_.q8_0_matmul_fanout(outs, ws, strides, 3,
                                                   *mtp_norm_batch_, batch, mtp_h_stride));
    }
    record(report, "mtp.attention_qkv_projection_batch");
    require_status(backend_.rmsnorm_per_head_batch(*mtp_q_batch_, *layer.attn_q_norm,
                                                   batch, q_stride_buf,
                                                   standard_n_heads,
                                                   2 * standard_head_dim,
                                                   standard_head_dim, eps));
    require_status(backend_.rmsnorm_per_head_batch(*mtp_k_batch_, *layer.attn_k_norm,
                                                   batch, k_stride_buf,
                                                   standard_n_kv_heads,
                                                   standard_head_dim,
                                                   standard_head_dim, eps));
    require_status(backend_.rope_partial_batch(*mtp_q_batch_,
                                               batch, q_stride_buf,
                                               standard_n_heads,
                                               2 * standard_head_dim,
                                               cfg.rope_dim, base_position, cfg.rope_theta));
    require_status(backend_.rope_partial_batch(*mtp_k_batch_,
                                               batch, k_stride_buf,
                                               standard_n_kv_heads,
                                               standard_head_dim,
                                               cfg.rope_dim, base_position, cfg.rope_theta));
    const uint32_t per_pos = standard_n_kv_heads * standard_head_dim;
    require_status(backend_.kv_append_batch(*mtp_k_cache_, *mtp_k_batch_,
                                            base_position, per_pos, batch));
    require_status(backend_.kv_append_batch(*mtp_v_cache_, *mtp_v_batch_,
                                            base_position, per_pos, batch));
    record(report, "mtp.kv_append_batch");

    const float scale = 1.0f / std::sqrt(static_cast<float>(standard_head_dim));
    require_status(backend_.attention_decode_batch_gated(*mtp_mid_batch_, *mtp_q_batch_,
                                                         2 * standard_head_dim,
                                                         *mtp_k_cache_, *mtp_v_cache_,
                                                         standard_n_heads, standard_n_kv_heads,
                                                         standard_head_dim,
                                                         base_position, batch,
                                                         q_stride_buf, mid_stride, scale));
    record(report, "mtp.attention_sdpa_batch");
    require_status(backend_.q8_0_matmul_add(*mtp_h_batch_,
                                            *mtp_h_batch_,
                                            *mtp_ffn_out_batch_,
                                            *layer.attn_output,
                                            *mtp_mid_batch_,
                                            batch, mid_stride, mtp_h_stride));
    record(report, "mtp.attn_residual_batch");

    require_status(backend_.rms_norm_batch(*mtp_norm_batch_, *mtp_h_batch_,
                                           *layer.ffn_norm, batch, mtp_h_stride, eps));
    record(report, "mtp.ffn_norm_batch");
    {
        DeviceTensor *outs[2] = {mtp_ffn_gate_batch_.get(), mtp_ffn_up_batch_.get()};
        const DeviceWeight *ws[2] = {layer.ffn_gate, layer.ffn_up};
        const uint32_t strides[2] = {ffn_stride, ffn_stride};
        require_status(backend_.q8_0_matmul_fanout(outs, ws, strides, 2,
                                                   *mtp_norm_batch_, batch, mtp_h_stride));
    }
    require_status(backend_.silu_mul_n(*mtp_ffn_mid_batch_,
                                       *mtp_ffn_gate_batch_,
                                       *mtp_ffn_up_batch_,
                                       static_cast<uint64_t>(batch) * ffn_stride));
    require_status(backend_.q8_0_matmul_add(*mtp_h_batch_,
                                            *mtp_h_batch_,
                                            *mtp_ffn_out_batch_,
                                            *layer.ffn_down,
                                            *mtp_ffn_mid_batch_,
                                            batch, ffn_stride, mtp_h_stride));
    record(report, "mtp.ffn_batch");

    require_status(backend_.end());
    require_status(backend_.copy_d2d(*mtp_prefix_h_, *h_batch_,
                                     static_cast<uint64_t>(batch - 1) * h_stride,
                                     h_stride));
    mtp_prefix_len_ = std::max<uint32_t>(mtp_prefix_len_, base_position + batch);
    report.ok = true;
    return report;
}

} // namespace qw3
