#include "dflash_batch_executor.hpp"
#include "q8_verification_math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace qw3 {
namespace {
void check(DeviceStatus status) {
    if (!status.ok) throw std::runtime_error(std::string("DFlash batch: ") + status.message);
}
void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(std::string("DFlash batch: ") + message);
}
}

std::unique_ptr<DeviceTensor> DFlashBatchExecutor::view(DeviceTensor &buffer, uint64_t offset, uint64_t count) {
    auto result = backend_.tensor_view(buffer, offset * 4, count, 4, DeviceTensorDType::F32, "dflash-batch-view");
    require(bool(result), "invalid tensor view");
    return result;
}
void DFlashBatchExecutor::linear(DeviceTensor &out, const DeviceWeight &weight,
                                const DeviceTensor &input, uint32_t rows) {
    // The shared Q8 target head must retain its narrow draft-head arithmetic.
    // All official draft projection weights continue through the BF16 path.
    if (weight.format == DeviceWeightFormat::Q8_0) {
        detail::Q8VerificationMath math(backend_, true);
        check(math.matmul(out, weight, input, rows, weight.cols, weight.rows));
    } else {
        check(backend_.dflash_linear_blocks(out, weight, input, active_blocks_, rows / active_blocks_));
    }
}
void DFlashBatchExecutor::reserve(uint32_t rows, uint32_t blocks) {
    const auto &c = weights_.config();
    if (rows > capacity_) {
        const uint64_t n = rows, h = c.hidden_size;
        auto alloc = [&](uint64_t count, const char *label) { return backend_.scratch_f32(count, label); };
        hidden_ = alloc(n * h, "dflash-batch-hidden"); norm_ = alloc(n * h, "dflash-batch-norm");
        mixed_ = alloc(n * h, "dflash-batch-mixed"); branch_ = alloc(n * h, "dflash-batch-branch");
        conv_out_ = alloc(n * h, "dflash-batch-conv"); output_ = alloc(n * h, "dflash-batch-output");
        delta_ = alloc(n * 2 * c.conv_kernel_size * (h / c.conv_group_size), "dflash-batch-dynamic");
        q_ = alloc(n * c.num_heads * c.head_dim, "dflash-batch-Q");
        k_ = alloc(n * c.num_kv_heads * c.head_dim, "dflash-batch-K");
        v_ = alloc(n * c.num_kv_heads * c.head_dim, "dflash-batch-V");
        attention_ = alloc(n * c.num_heads * c.head_dim, "dflash-batch-attention");
        gate_ = alloc(n * c.intermediate_size, "dflash-batch-gate");
        up_ = alloc(n * c.intermediate_size, "dflash-batch-up");
        mid_ = alloc(n * c.intermediate_size, "dflash-batch-mid");
        proposal_hidden_ = alloc(n * h, "dflash-batch-proposal-hidden");
        logits_ = alloc(n * c.vocab_size, "dflash-batch-logits");
        top_ids_ = backend_.tensor_i32(n * c.selector_top_k, "dflash-batch-top-ids");
        top_values_ = alloc(n * c.selector_top_k, "dflash-batch-top-values");
        const uint64_t parts = (uint64_t(c.vocab_size) + 4095) / 4096;
        scratch_ids_ = backend_.tensor_i32(n * parts * c.selector_top_k, "dflash-batch-top-scratch-ids");
        scratch_values_ = alloc(n * parts * c.selector_top_k, "dflash-batch-top-scratch-values");
        selector_ = alloc(n * c.selector_rank, "dflash-batch-selector");
        tokens_ = backend_.tensor_i32(n, "dflash-batch-tokens");
        probabilities_ = alloc(n * c.selector_top_k, "dflash-batch-probabilities");
        uniforms_ = alloc(n, "dflash-batch-uniforms");
        capacity_ = rows;
    }
    if (blocks > blocks_capacity_) {
        metadata_ = backend_.tensor_i32(uint64_t(blocks) * 6, "dflash-batch-attention-metadata");
        anchors_ = backend_.tensor_i32(blocks, "dflash-batch-anchors");
        blocks_capacity_ = blocks;
    }
}

std::vector<DFlashProposal> DFlashBatchExecutor::propose(
        const std::vector<Request> &requests, const DeviceWeight &target_head) {
    if (requests.empty()) return {};
    const auto &c = weights_.config();
    const uint32_t block = requests.front().rows;
    const float temperature = requests.front().temperature;
    require(block > 0 && block <= c.block_size && requests.size() <= 65535 &&
            requests.size() * uint64_t(block) <= UINT32_MAX, "invalid request/block count");
    require(std::isfinite(temperature) && temperature >= 0, "invalid temperature");
    require(target_head.rows == c.vocab_size && target_head.cols == c.hidden_size, "incompatible target head");
    std::unordered_set<DFlashExecutor *> unique;
    for (const auto &r : requests) {
        require(r.executor && r.noise && r.rng && unique.insert(r.executor).second,
                "missing or duplicate request state");
        const auto &e = *r.executor;
        require(&e.weights_ == &weights_ && &e.backend_ == &backend_, "request belongs to another draft/device");
        require(r.rows == block && r.temperature == temperature && r.anchor < c.vocab_size,
                "batch must share proposal width and temperature");
        require(r.base_position == e.next_position_ && r.base_position <= UINT64_MAX - block &&
                uint64_t(e.rope_next_) + block <= c.max_position_embeddings, "invalid logical/RoPE position");
        require(r.noise->dtype == DeviceTensorDType::F32 && r.noise->count >= uint64_t(block) * c.hidden_size,
                "invalid noise embedding tensor");
    }
    check(backend_.begin());
    const uint32_t blocks = requests.size(), rows = blocks * block;
    active_blocks_ = blocks;
    const uint64_t h = c.hidden_size, qw = uint64_t(c.num_heads) * c.head_dim;
    const uint64_t kv = uint64_t(c.num_kv_heads) * c.head_dim;
    reserve(rows, blocks);
    std::vector<const DeviceTensor *> keys(blocks), values(blocks);
    std::vector<uint32_t> lengths(blocks);
    for (uint32_t i = 0; i < blocks; ++i) {
        auto dst = view(*hidden_, uint64_t(i) * block * h, uint64_t(block) * h);
        check(backend_.copy_d2d(*dst, *requests[i].noise, 0, uint64_t(block) * h));
        requests[i].executor->last_rows_ = 0;
        lengths[i] = requests[i].executor->context_rows_ + block;
    }
    auto conv = [&](DeviceTensor &out, const DeviceTensor &input, const DeviceWeight &base, uint32_t side) {
        check(backend_.dflash_grouped_conv(out, input, *delta_, base, rows, c.hidden_size,
            block, c.conv_kernel_size, c.conv_group_size, side));
    };
    for (uint32_t index = 0; index < c.num_layers; ++index) {
        const auto &layer = requests.front().executor->layers_[index];
        check(backend_.rms_norm_batch(*norm_, *hidden_, *layer.input_norm, rows, c.hidden_size, c.rms_norm_eps));
        linear(*delta_, *layer.attn_conv, *norm_, rows);
        conv(*mixed_, *norm_, *layer.attn_base, 0);
        linear(*q_, *layer.q, *mixed_, rows);
        linear(*k_, *layer.k, *mixed_, rows); linear(*v_, *layer.v, *mixed_, rows);
        check(backend_.rmsnorm_per_head_batch(*q_, *layer.q_norm, rows, qw,
            c.num_heads, c.head_dim, c.head_dim, c.rms_norm_eps));
        check(backend_.rmsnorm_per_head_batch(*k_, *layer.k_norm, rows, kv,
            c.num_kv_heads, c.head_dim, c.head_dim, c.rms_norm_eps));
        for (uint32_t i = 0; i < blocks; ++i) {
            auto &e = *requests[i].executor; auto &cache = e.layers_[index];
            auto q = view(*q_, uint64_t(i) * block * qw, uint64_t(block) * qw);
            auto k = view(*k_, uint64_t(i) * block * kv, uint64_t(block) * kv);
            check(backend_.rope_partial_batch(*q, block, qw, c.num_heads, c.head_dim, c.head_dim,
                e.rope_next_, c.rope_theta));
            check(backend_.rope_partial_batch(*k, block, kv, c.num_kv_heads, c.head_dim, c.head_dim,
                e.rope_next_, c.rope_theta));
            auto kd = view(*cache.keys, uint64_t(e.context_rows_) * kv, uint64_t(block) * kv);
            auto vd = view(*cache.values, uint64_t(e.context_rows_) * kv, uint64_t(block) * kv);
            check(backend_.copy_d2d(*kd, *k, 0, uint64_t(block) * kv));
            check(backend_.copy_d2d(*vd, *v_, uint64_t(i) * block * kv, uint64_t(block) * kv));
            keys[i] = cache.keys.get(); values[i] = cache.values.get();
        }
        check(backend_.dflash_attention_batch(*attention_, *q_, *metadata_, keys.data(), values.data(),
            lengths.data(), blocks, block, c.num_heads, c.num_kv_heads, c.head_dim, c.sliding_window));
        linear(*branch_, *layer.o, *attention_, rows);
        conv(*conv_out_, *branch_, *layer.attn_base, 1);
        check(backend_.add_n(*hidden_, *hidden_, *conv_out_, uint64_t(rows) * h));
        check(backend_.rms_norm_batch(*norm_, *hidden_, *layer.ffn_norm, rows, c.hidden_size, c.rms_norm_eps));
        linear(*delta_, *layer.mlp_conv, *norm_, rows);
        conv(*mixed_, *norm_, *layer.mlp_base, 0);
        linear(*gate_, *layer.gate, *mixed_, rows); linear(*up_, *layer.up, *mixed_, rows);
        check(backend_.silu_mul_n(*mid_, *gate_, *up_, uint64_t(rows) * c.intermediate_size));
        linear(*branch_, *layer.down, *mid_, rows);
        conv(*conv_out_, *branch_, *layer.mlp_base, 1);
        check(backend_.add_n(*hidden_, *hidden_, *conv_out_, uint64_t(rows) * h));
    }
    check(backend_.rms_norm_batch(*output_, *hidden_, weights_.at("norm.weight"), rows, c.hidden_size, c.rms_norm_eps));
    std::vector<DFlashProposal> proposals(blocks);
    const uint32_t positions = block - 1, proposal_rows = blocks * positions, top_k = c.selector_top_k;
    for (uint32_t i = 0; i < blocks; ++i) {
        auto &e = *requests[i].executor;
        check(backend_.copy_d2d(*e.output_, *output_, uint64_t(i) * block * h, uint64_t(block) * h));
        e.last_rows_ = block; proposals[i].top_k = top_k;
        if (positions) {
            auto dst = view(*proposal_hidden_, uint64_t(i) * positions * h, uint64_t(positions) * h);
            check(backend_.copy_d2d(*dst, *output_, (uint64_t(i) * block + 1) * h, uint64_t(positions) * h));
        }
    }
    if (!positions) return proposals;
    linear(*logits_, target_head, *proposal_hidden_, proposal_rows);
    check(backend_.dflash_topk(*top_ids_, *top_values_, *logits_, *scratch_ids_, *scratch_values_,
        proposal_rows, c.vocab_size, top_k));
    linear(*selector_, weights_.at("candidate_selector.hidden_projection.weight"), *proposal_hidden_, proposal_rows);
    std::vector<int32_t> anchors(blocks);
    std::vector<float> uniforms(proposal_rows, 0);
    for (uint32_t i = 0; i < blocks; ++i) {
        anchors[i] = requests[i].anchor;
        if (temperature > 0) {
            std::uniform_real_distribution<float> distribution(0, 1);
            for (uint32_t j = 0; j < positions; ++j) uniforms[i * positions + j] = distribution(*requests[i].rng);
        }
    }
    check(backend_.copy_i32_from_host(*anchors_, 0, anchors.data(), anchors.size()));
    check(backend_.copy_bytes_from_host(*uniforms_, 0, uniforms.data(), uint64_t(proposal_rows) * 4));
    check(backend_.dflash_select(*tokens_, *probabilities_, *top_ids_, *top_values_, *selector_,
        weights_.at("candidate_selector.predecessor_codebook"), weights_.at("candidate_selector.successor_codebook"),
        *anchors_, *uniforms_, blocks, positions, top_k, c.selector_rank, temperature));
    std::vector<int32_t> tokens(proposal_rows), candidates(uint64_t(proposal_rows) * top_k);
    std::vector<float> probabilities(candidates.size());
    check(backend_.copy_bytes_to_host(*tokens_, tokens.data(), 0, uint64_t(tokens.size()) * 4));
    check(backend_.copy_bytes_to_host(*top_ids_, candidates.data(), 0, uint64_t(candidates.size()) * 4));
    check(backend_.copy_to_host(*probabilities_, probabilities.data(), 0, probabilities.size()));
    for (uint32_t i = 0; i < blocks; ++i) {
        const uint64_t begin = uint64_t(i) * positions, first = begin * top_k, count = uint64_t(positions) * top_k;
        proposals[i].tokens.assign(tokens.begin() + begin, tokens.begin() + begin + positions);
        proposals[i].candidates.assign(candidates.begin() + first, candidates.begin() + first + count);
        proposals[i].probabilities.assign(probabilities.begin() + first, probabilities.begin() + first + count);
    }
    return proposals;
}
} // namespace qw3
