#include "dflash_executor.hpp"
#include "dflash_archive.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace qw3 {
namespace {
void check(DeviceStatus s) {
    if (!s.ok) throw std::runtime_error(std::string("DFlash execution: ") + s.message);
}
void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(std::string("DFlash execution: ") + message);
}
} // namespace

DFlashExecutor::DFlashExecutor(const DFlashWeights &weights, DeviceBackend &backend)
    : weights_(weights), backend_(backend) {
    check(backend_.begin());
    require(backend_.supports_tensor_views(), "tensor views are required");
    const auto &c = weights_.config();
    const uint64_t h = c.hidden_size, b = c.block_size, w = c.sliding_window;
    const uint64_t kv = uint64_t(c.num_kv_heads) * c.head_dim;
    require(w + b <= 8192, "draft window exceeds the current CUDA attention capacity");
    require(w + b <= c.max_position_embeddings, "draft context must leave room for a speculative block");
    for (uint32_t i = 0; i < c.num_layers; ++i) {
        const auto p = "layers." + std::to_string(i) + ".";
        auto get = [&](const char *name) { return &weights_.at(p + name); };
        Layer layer;
        layer.input_norm = get("input_layernorm.weight");
        layer.ffn_norm = get("post_attention_layernorm.weight");
        layer.q = get("self_attn.q_proj.weight"); layer.k = get("self_attn.k_proj.weight");
        layer.v = get("self_attn.v_proj.weight"); layer.o = get("self_attn.o_proj.weight");
        layer.q_norm = get("self_attn.q_norm.weight"); layer.k_norm = get("self_attn.k_norm.weight");
        layer.gate = get("mlp.gate_proj.weight"); layer.up = get("mlp.up_proj.weight");
        layer.down = get("mlp.down_proj.weight");
        layer.attn_conv = get("attention_conv.kernel_projection.weight");
        layer.attn_base = get("attention_conv.base_kernel");
        layer.mlp_conv = get("mlp_conv.kernel_projection.weight");
        layer.mlp_base = get("mlp_conv.base_kernel");
        layer.keys = backend_.scratch_f32((w + b) * kv, "dflash-K");
        layer.values = backend_.scratch_f32((w + b) * kv, "dflash-V");
        layer.raw_keys = backend_.scratch_f32(w * kv, "dflash-raw-K");
        layers_.push_back(std::move(layer));
    }
    auto alloc = [&](uint64_t n, const char *label) { return backend_.scratch_f32(n, label); };
    context_projection_ = alloc(w * h, "dflash-context-projection");
    context_norm_ = alloc(w * h, "dflash-context-norm");
    compact_ = alloc(w * kv, "dflash-compact");
    hidden_ = alloc(b * h, "dflash-hidden"); norm_ = alloc(b * h, "dflash-norm");
    mixed_ = alloc(b * h, "dflash-mixed"); branch_ = alloc(b * h, "dflash-branch");
    conv_out_ = alloc(b * h, "dflash-conv-out"); output_ = alloc(b * h, "dflash-output");
    delta_ = alloc(b * 2 * c.conv_kernel_size * (h / c.conv_group_size), "dflash-dynamic");
    q_ = alloc(b * c.num_heads * c.head_dim, "dflash-Q");
    attention_ = alloc(b * c.num_heads * c.head_dim, "dflash-attention");
    ffn_gate_ = alloc(b * c.intermediate_size, "dflash-ffn-gate");
    ffn_up_ = alloc(b * c.intermediate_size, "dflash-ffn-up");
    ffn_mid_ = alloc(b * c.intermediate_size, "dflash-ffn-mid");
    const uint64_t proposals = b - 1, k = c.selector_top_k;
    logits_ = alloc(proposals * c.vocab_size, "dflash-logits");
    top_ids_ = backend_.tensor_i32(proposals * k, "dflash-top-ids");
    top_values_ = alloc(proposals * k, "dflash-top-values");
    const uint64_t parts = (uint64_t(c.vocab_size) + 4095) / 4096;
    top_scratch_ids_ = backend_.tensor_i32(proposals * parts * k, "dflash-top-scratch-ids");
    top_scratch_values_ = alloc(proposals * parts * k, "dflash-top-scratch-values");
    selector_hidden_ = alloc(proposals * c.selector_rank, "dflash-selector-hidden");
    tokens_ = backend_.tensor_i32(proposals, "dflash-tokens");
    probabilities_ = alloc(proposals * k, "dflash-probabilities");
    anchor_ = backend_.tensor_i32(1, "dflash-anchor");
    uniforms_ = alloc(proposals, "dflash-uniforms");
}

std::unique_ptr<DeviceTensor> DFlashExecutor::view(DeviceTensor &buffer, uint64_t offset, uint64_t count) {
    auto result = backend_.tensor_view(buffer, offset * sizeof(float), count, sizeof(float),
                                       DeviceTensorDType::F32, "dflash-view");
    require(bool(result), "cannot create FP32 tensor view");
    return result;
}

void DFlashExecutor::linear(DeviceTensor &out, const DeviceWeight &weight,
                            const DeviceTensor &input, uint32_t rows) {
    check(backend_.linear_matmul(out, weight, input, rows, weight.cols, weight.rows));
}

void DFlashExecutor::reset(uint64_t next_position) {
    context_rows_ = 0;
    next_position_ = next_position;
    // Draft RoPE is independent of the target's selected-window coordinate.
    // A common translation of all draft Q/K leaves relative phases unchanged.
    // Logical history can therefore span uint64_t without entering RoPE.
    rope_next_ = 0;
    last_rows_ = 0;
}

DFlashExecutor::StateSnapshot DFlashExecutor::snapshot() {
    StateSnapshot result;
    result.weights = &weights_; result.backend = &backend_;
    result.next_position = next_position_; result.rows = context_rows_; result.rope_next = rope_next_;
    const uint64_t count = uint64_t(context_rows_) * weights_.config().num_kv_heads * weights_.config().head_dim;
    if (!count) return result;
    check(backend_.begin());
    for (auto &layer : layers_) {
        auto k = backend_.scratch_f32(count, "dflash-snapshot-K");
        auto v = backend_.scratch_f32(count, "dflash-snapshot-V");
        check(backend_.copy_d2d(*k, *layer.raw_keys, 0, count));
        check(backend_.copy_d2d(*v, *layer.values, 0, count));
        result.raw_keys.emplace_back(std::move(k)); result.values.emplace_back(std::move(v));
    }
    return result;
}

void DFlashExecutor::restore(const StateSnapshot &state) {
    const auto &c = weights_.config();
    require(state.weights == &weights_ && state.backend == &backend_, "snapshot belongs to another draft model or device");
    require(state.rows <= c.sliding_window && state.rows <= state.rope_next &&
            uint64_t(state.rope_next) + c.block_size <= c.max_position_embeddings,
            "invalid draft snapshot positions");
    const size_t layers = state.rows ? layers_.size() : 0;
    require(state.raw_keys.size() == layers && state.values.size() == layers, "invalid draft snapshot layer count");
    const uint64_t kv = uint64_t(c.num_kv_heads) * c.head_dim, count = uint64_t(state.rows) * kv;
    for (size_t i = 0; i < layers; ++i)
        require(state.raw_keys[i] && state.values[i] && state.raw_keys[i]->dtype == DeviceTensorDType::F32 &&
                state.values[i]->dtype == DeviceTensorDType::F32 && state.raw_keys[i]->count == count &&
                state.values[i]->count == count, "invalid draft snapshot tensor");
    check(backend_.begin());
    for (size_t i = 0; i < layers; ++i) {
        auto &layer = layers_[i];
        check(backend_.copy_d2d(*layer.raw_keys, *state.raw_keys[i], 0, count));
        check(backend_.copy_d2d(*layer.keys, *state.raw_keys[i], 0, count));
        check(backend_.copy_d2d(*layer.values, *state.values[i], 0, count));
        check(backend_.rope_partial_batch(*layer.keys, state.rows, kv, c.num_kv_heads, c.head_dim,
            c.head_dim, state.rope_next - state.rows, c.rope_theta));
    }
    context_rows_ = state.rows; next_position_ = state.next_position; rope_next_ = state.rope_next; last_rows_ = 0;
}

void DFlashExecutor::capture_archive_state(KvMemArchiveState &state) {
    const auto &c = weights_.config();
    require(state.position == next_position_, "archive target/draft position mismatch");
    const auto text = nlohmann::json{
        {"version", 1u}, {"identity_sha256", weights_.identity_sha256()},
        {"next_position", next_position_}, {"rows", context_rows_}, {"rope_next", rope_next_},
        {"layers", c.num_layers}, {"kv_heads", c.num_kv_heads}, {"head_dim", c.head_dim},
        {"window", c.sliding_window}, {"max_position", c.max_position_embeddings},
        {"block_size", c.block_size}}.dump();
    state.entries.push_back({KvMemArchiveStateEntry::DFlashMetadata, 0, text.size()});
    state.payload.insert(state.payload.end(), text.begin(), text.end());
    const uint64_t bytes = uint64_t(context_rows_) * c.num_kv_heads * c.head_dim * sizeof(float);
    for (uint32_t i = 0; bytes && i < layers_.size(); ++i) {
        for (bool key : {true, false}) {
            const size_t base = state.payload.size();
            state.payload.resize(base + bytes);
            check(backend_.copy_bytes_to_host(key ? *layers_[i].raw_keys : *layers_[i].values,
                                             state.payload.data() + base, 0, bytes));
            state.entries.push_back({key ? KvMemArchiveStateEntry::DFlashRawK : KvMemArchiveStateEntry::DFlashV, i, bytes});
        }
    }
    (void)detail::read_dflash_archive(state);
}

void DFlashExecutor::restore_archive_state(const KvMemArchiveState &state) {
    const auto view = detail::read_dflash_archive(state);
    if (!view || view->identity != weights_.identity_sha256()) {
        reset(state.position);
        std::fprintf(stderr, "[kvmem-archive] dflash=cold reason=%s position=%llu rows=0\n",
                     view ? "different-draft" : "no-draft-state", static_cast<unsigned long long>(state.position));
        return;
    }
    const auto &v = *view;
    const auto &c = weights_.config();
    require(v.layers == c.num_layers && v.kv_heads == c.num_kv_heads && v.head_dim == c.head_dim &&
            v.window == c.sliding_window && v.max_position == c.max_position_embeddings &&
            v.block_size == c.block_size, "archive draft configuration mismatch");
    check(backend_.begin());
    const uint64_t kv = uint64_t(c.num_kv_heads) * c.head_dim;
    for (size_t i = 0; i < v.keys.size(); ++i) {
        auto &layer = layers_[i];
        check(backend_.copy_bytes_from_host(*layer.raw_keys, 0, state.payload.data() + v.keys[i], v.tensor_bytes()));
        check(backend_.copy_bytes_from_host(*layer.values, 0, state.payload.data() + v.values[i], v.tensor_bytes()));
        check(backend_.copy_d2d(*layer.keys, *layer.raw_keys, 0, uint64_t(v.rows) * kv));
        check(backend_.rope_partial_batch(*layer.keys, v.rows, kv, c.num_kv_heads, c.head_dim,
                                         c.head_dim, v.rope_next - v.rows, c.rope_theta));
    }
    context_rows_ = v.rows; next_position_ = v.next_position; rope_next_ = v.rope_next; last_rows_ = 0;
    std::fprintf(stderr, "[kvmem-archive] dflash=restored position=%llu rows=%u rope_next=%u bytes=%llu\n",
                 static_cast<unsigned long long>(next_position_), context_rows_, rope_next_,
                 static_cast<unsigned long long>(2 * v.layers * v.tensor_bytes()));
}

void DFlashExecutor::append_context(const DeviceTensor &features, uint32_t rows, uint64_t base_position) {
    const auto &c = weights_.config();
    const uint64_t input_width = uint64_t(c.hidden_size) * c.target_layer_ids.size();
    require(base_position == next_position_, "non-contiguous committed target features");
    require(base_position <= std::numeric_limits<uint64_t>::max() - rows, "feature logical position overflow");
    require(features.dtype == DeviceTensorDType::F32 && features.count >= uint64_t(rows) * input_width,
            "invalid target feature tensor");
    if (!rows) return;
    check(backend_.begin());
    const uint32_t take = std::min(rows, c.sliding_window), skip = rows - take;
    const uint32_t keep = std::min(context_rows_, c.sliding_window - take);
    const uint32_t removed = context_rows_ - keep;
    const bool rebase = uint64_t(rope_next_) + rows + c.block_size > c.max_position_embeddings;
    const uint32_t new_rope_next = rebase ? keep + take : rope_next_ + rows;
    auto feature_view = backend_.tensor_view(const_cast<DeviceTensor &>(features),
        uint64_t(skip) * input_width * sizeof(float), uint64_t(take) * input_width,
        sizeof(float), DeviceTensorDType::F32, "dflash-target-features");
    require(bool(feature_view), "cannot view target feature suffix");
    linear(*context_projection_, weights_.at("fc.weight"), *feature_view, take);
    check(backend_.rms_norm_batch(*context_norm_, *context_projection_, weights_.at("hidden_norm.weight"),
                                  take, c.hidden_size, c.rms_norm_eps));
    const uint64_t kv = uint64_t(c.num_kv_heads) * c.head_dim;
    for (auto &layer : layers_) {
        if (keep && removed) {
            check(backend_.copy_d2d(*compact_, *layer.keys, uint64_t(removed) * kv, uint64_t(keep) * kv));
            check(backend_.copy_d2d(*layer.keys, *compact_, 0, uint64_t(keep) * kv));
            check(backend_.copy_d2d(*compact_, *layer.values, uint64_t(removed) * kv, uint64_t(keep) * kv));
            check(backend_.copy_d2d(*layer.values, *compact_, 0, uint64_t(keep) * kv));
            check(backend_.copy_d2d(*compact_, *layer.raw_keys, uint64_t(removed) * kv, uint64_t(keep) * kv));
            check(backend_.copy_d2d(*layer.raw_keys, *compact_, 0, uint64_t(keep) * kv));
        }
        auto raw_k = view(*layer.raw_keys, uint64_t(keep) * kv, uint64_t(take) * kv);
        auto v = view(*layer.values, uint64_t(keep) * kv, uint64_t(take) * kv);
        linear(*raw_k, *layer.k, *context_norm_, take);
        linear(*v, *layer.v, *context_norm_, take);
        check(backend_.rmsnorm_per_head_batch(*raw_k, *layer.k_norm, take, kv,
            c.num_kv_heads, c.head_dim, c.head_dim, c.rms_norm_eps));
        if (rebase) {
            check(backend_.copy_d2d(*layer.keys, *layer.raw_keys, 0, uint64_t(keep + take) * kv));
            check(backend_.rope_partial_batch(*layer.keys, keep + take, kv, c.num_kv_heads, c.head_dim,
                c.head_dim, 0, c.rope_theta));
        } else {
            auto k = view(*layer.keys, uint64_t(keep) * kv, uint64_t(take) * kv);
            check(backend_.copy_d2d(*k, *raw_k, 0, uint64_t(take) * kv));
            check(backend_.rope_partial_batch(*k, take, kv, c.num_kv_heads, c.head_dim, c.head_dim,
                new_rope_next - take, c.rope_theta));
        }
    }
    context_rows_ = keep + take;
    next_position_ += rows;
    rope_next_ = new_rope_next;
    last_rows_ = 0;
}

void DFlashExecutor::convolve(DeviceTensor &out, const DeviceTensor &input,
                              const DeviceWeight &base, uint32_t rows, uint32_t side) {
    const auto &c = weights_.config();
    check(backend_.dflash_grouped_conv(out, input, *delta_, base, rows, c.hidden_size,
                                       rows, c.conv_kernel_size, c.conv_group_size, side));
}

const DeviceTensor &DFlashExecutor::forward(const DeviceTensor &noise, uint32_t rows, uint64_t base_position) {
    const auto &c = weights_.config();
    require(rows && rows <= c.block_size, "invalid draft block width");
    require(base_position == next_position_, "draft position differs from committed context");
    require(base_position <= std::numeric_limits<uint64_t>::max() - rows, "draft logical position overflow");
    require(uint64_t(rope_next_) + rows <= c.max_position_embeddings, "draft RoPE positions exceed context");
    require(noise.dtype == DeviceTensorDType::F32 && noise.count >= uint64_t(rows) * c.hidden_size,
            "invalid anchor/mask embedding tensor");
    check(backend_.begin());
    last_rows_ = 0;
    check(backend_.copy_d2d(*hidden_, noise, 0, uint64_t(rows) * c.hidden_size));
    const uint64_t q_width = uint64_t(c.num_heads) * c.head_dim;
    const uint64_t kv = uint64_t(c.num_kv_heads) * c.head_dim;
    for (auto &layer : layers_) {
        check(backend_.rms_norm_batch(*norm_, *hidden_, *layer.input_norm, rows, c.hidden_size, c.rms_norm_eps));
        linear(*delta_, *layer.attn_conv, *norm_, rows);
        convolve(*mixed_, *norm_, *layer.attn_base, rows, 0);
        linear(*q_, *layer.q, *mixed_, rows);
        auto k = view(*layer.keys, uint64_t(context_rows_) * kv, uint64_t(rows) * kv);
        auto v = view(*layer.values, uint64_t(context_rows_) * kv, uint64_t(rows) * kv);
        linear(*k, *layer.k, *mixed_, rows); linear(*v, *layer.v, *mixed_, rows);
        check(backend_.rmsnorm_per_head_batch(*q_, *layer.q_norm, rows, q_width,
            c.num_heads, c.head_dim, c.head_dim, c.rms_norm_eps));
        check(backend_.rmsnorm_per_head_batch(*k, *layer.k_norm, rows, kv,
            c.num_kv_heads, c.head_dim, c.head_dim, c.rms_norm_eps));
        check(backend_.rope_partial_batch(*q_, rows, q_width, c.num_heads, c.head_dim, c.head_dim,
            rope_next_, c.rope_theta));
        check(backend_.rope_partial_batch(*k, rows, kv, c.num_kv_heads, c.head_dim, c.head_dim,
            rope_next_, c.rope_theta));
        check(backend_.dflash_attention(*attention_, *q_, *layer.keys, *layer.values,
            rows, context_rows_ + rows, c.num_heads, c.num_kv_heads, c.head_dim, c.sliding_window));
        linear(*branch_, *layer.o, *attention_, rows);
        convolve(*conv_out_, *branch_, *layer.attn_base, rows, 1);
        check(backend_.add_n(*hidden_, *hidden_, *conv_out_, uint64_t(rows) * c.hidden_size));
        check(backend_.rms_norm_batch(*norm_, *hidden_, *layer.ffn_norm, rows, c.hidden_size, c.rms_norm_eps));
        linear(*delta_, *layer.mlp_conv, *norm_, rows);
        convolve(*mixed_, *norm_, *layer.mlp_base, rows, 0);
        linear(*ffn_gate_, *layer.gate, *mixed_, rows); linear(*ffn_up_, *layer.up, *mixed_, rows);
        check(backend_.silu_mul_n(*ffn_mid_, *ffn_gate_, *ffn_up_, uint64_t(rows) * c.intermediate_size));
        linear(*branch_, *layer.down, *ffn_mid_, rows);
        convolve(*conv_out_, *branch_, *layer.mlp_base, rows, 1);
        check(backend_.add_n(*hidden_, *hidden_, *conv_out_, uint64_t(rows) * c.hidden_size));
    }
    check(backend_.rms_norm_batch(*output_, *hidden_, weights_.at("norm.weight"), rows, c.hidden_size, c.rms_norm_eps));
    last_rows_ = rows;
    return *output_;
}

DFlashProposal DFlashExecutor::propose(const DeviceWeight &target_head, uint32_t anchor,
                                      float temperature, std::mt19937_64 &rng) {
    const auto &c = weights_.config();
    require(last_rows_ != 0, "proposal requested without draft forward");
    require(target_head.rows == c.vocab_size && target_head.cols == c.hidden_size,
            "incompatible target LM head");
    require(anchor < c.vocab_size && std::isfinite(temperature) && temperature >= 0,
            "invalid anchor or sampling temperature");
    DFlashProposal proposal;
    proposal.top_k = c.selector_top_k;
    const uint32_t rows = last_rows_ - 1;
    if (!rows) return proposal;
    auto hidden = view(*output_, c.hidden_size, uint64_t(rows) * c.hidden_size);
    linear(*logits_, target_head, *hidden, rows);
    check(backend_.dflash_topk(*top_ids_, *top_values_, *logits_, *top_scratch_ids_, *top_scratch_values_,
                               rows, c.vocab_size, c.selector_top_k));
    linear(*selector_hidden_, weights_.at("candidate_selector.hidden_projection.weight"), *hidden, rows);
    const int32_t anchor_id = anchor;
    check(backend_.copy_i32_from_host(*anchor_, 0, &anchor_id, 1));
    std::vector<float> uniforms(rows, 0);
    if (temperature > 0) {
        std::uniform_real_distribution<float> distribution(0, 1);
        for (auto &u : uniforms) u = distribution(rng);
    }
    check(backend_.copy_bytes_from_host(*uniforms_, 0, uniforms.data(), uint64_t(rows) * sizeof(float)));
    check(backend_.dflash_select(*tokens_, *probabilities_, *top_ids_, *top_values_, *selector_hidden_,
        weights_.at("candidate_selector.predecessor_codebook"), weights_.at("candidate_selector.successor_codebook"),
        *anchor_, *uniforms_, 1, rows, c.selector_top_k, c.selector_rank, temperature));
    proposal.tokens.resize(rows);
    proposal.candidates.resize(uint64_t(rows) * c.selector_top_k);
    proposal.probabilities.resize(proposal.candidates.size());
    check(backend_.copy_bytes_to_host(*tokens_, proposal.tokens.data(), 0, uint64_t(rows) * sizeof(int32_t)));
    check(backend_.copy_bytes_to_host(*top_ids_, proposal.candidates.data(), 0, proposal.candidates.size() * sizeof(int32_t)));
    check(backend_.copy_to_host(*probabilities_, proposal.probabilities.data(), 0, proposal.probabilities.size()));
    return proposal;
}

} // namespace qw3
