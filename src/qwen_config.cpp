#include "qw3/qwen_config.hpp"

#include <stdexcept>

namespace qw3 {
namespace {

const GgufValue &require(const GgufFile &gguf, const std::string &key) {
    const auto &meta = gguf.metadata();
    const auto it = meta.find(key);
    if (it == meta.end()) throw std::runtime_error("GGUF is missing required key: " + key);
    return it->second;
}

const GgufValue *optional(const GgufFile &gguf, const std::string &key) {
    const auto &meta = gguf.metadata();
    const auto it = meta.find(key);
    if (it == meta.end()) return nullptr;
    return &it->second;
}

uint32_t as_u32(const GgufValue &v) {
    if (v.type == GgufValueType::Uint8 || v.type == GgufValueType::Uint16 ||
        v.type == GgufValueType::Uint32 || v.type == GgufValueType::Uint64) {
        return static_cast<uint32_t>(v.unsigned_value);
    }
    if (v.type == GgufValueType::Int8 || v.type == GgufValueType::Int16 ||
        v.type == GgufValueType::Int32 || v.type == GgufValueType::Int64) {
        return static_cast<uint32_t>(v.signed_value > 0 ? v.signed_value : 0);
    }
    throw std::runtime_error("unexpected metadata type for u32 field");
}

float as_f32(const GgufValue &v) {
    if (v.type == GgufValueType::Float32 || v.type == GgufValueType::Float64) {
        return static_cast<float>(v.float_value);
    }
    if (v.type == GgufValueType::Uint32 || v.type == GgufValueType::Uint64) {
        return static_cast<float>(v.unsigned_value);
    }
    throw std::runtime_error("unexpected metadata type for f32 field");
}

} // namespace

QwenConfig::QwenConfig(const GgufFile &gguf) {
    architecture = require(gguf, "general.architecture").string_value;
    if (architecture != "qwen35" && architecture != "qwen3") {
        throw std::runtime_error("unsupported architecture: " + architecture);
    }
    const std::string p = architecture + ".";

    block_count = as_u32(require(gguf, p + "block_count"));
    if (const auto *nextn = optional(gguf, p + "nextn_predict_layers")) {
        nextn_predict_layers = as_u32(*nextn);
    }
    if (nextn_predict_layers > block_count) {
        throw std::runtime_error("nextn_predict_layers exceeds block_count");
    }
    n_layers = block_count - nextn_predict_layers;
    n_embd = as_u32(require(gguf, p + "embedding_length"));
    n_ff = as_u32(require(gguf, p + "feed_forward_length"));
    n_ctx_train = as_u32(require(gguf, p + "context_length"));

    n_heads = as_u32(require(gguf, p + "attention.head_count"));
    n_kv_heads = as_u32(require(gguf, p + "attention.head_count_kv"));
    head_dim = as_u32(require(gguf, p + "attention.key_length"));
    if (const auto *vl = optional(gguf, p + "attention.value_length")) {
        head_v_dim = as_u32(*vl);
    } else {
        head_v_dim = head_dim;
    }
    if (const auto *eps = optional(gguf, p + "attention.layer_norm_rms_epsilon")) {
        rms_eps = as_f32(*eps);
    }

    if (const auto *rd = optional(gguf, p + "rope.dimension_count")) {
        rope_dim = as_u32(*rd);
    } else {
        rope_dim = head_dim;
    }
    if (const auto *rt = optional(gguf, p + "rope.freq_base")) {
        rope_theta = as_f32(*rt);
    }

    if (const auto *ck = optional(gguf, p + "ssm.conv_kernel")) ssm_conv_kernel = as_u32(*ck);
    if (const auto *isz = optional(gguf, p + "ssm.inner_size")) ssm_inner_size = as_u32(*isz);
    if (const auto *ss = optional(gguf, p + "ssm.state_size")) ssm_state_size = as_u32(*ss);
    if (const auto *tsr = optional(gguf, p + "ssm.time_step_rank")) ssm_time_step_rank = as_u32(*tsr);
    if (const auto *gc = optional(gguf, p + "ssm.group_count")) ssm_group_count = as_u32(*gc);

    if (const auto *fi = optional(gguf, p + "full_attention_interval")) {
        full_attention_interval = as_u32(*fi);
    }

    const auto &tokens = require(gguf, "tokenizer.ggml.tokens");
    vocab_size = static_cast<uint32_t>(tokens.string_array.size());
    if (const auto *b = optional(gguf, "tokenizer.ggml.bos_token_id")) bos_id = as_u32(*b);
    if (const auto *e = optional(gguf, "tokenizer.ggml.eos_token_id")) eos_id = as_u32(*e);
    if (const auto *ab = optional(gguf, "tokenizer.ggml.add_bos_token")) add_bos = ab->bool_value;
}

} // namespace qw3
