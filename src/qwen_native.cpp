#include "qwen_native.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace qw3 {
namespace {

std::string layer_tensor(uint32_t layer, const char *suffix) {
    std::ostringstream ss;
    ss << "blk." << layer << "." << suffix;
    return ss.str();
}

void add_op(std::vector<std::string> &ops, const std::string &name) {
    if (ops.empty() || ops.back() != name) ops.push_back(name);
}

} // namespace

QwenNativeModel::QwenNativeModel(std::unique_ptr<GgufFile> gguf) : gguf_(std::move(gguf)) {
    if (!gguf_) throw std::invalid_argument("QwenNativeModel requires GGUF");
    config_ = std::make_unique<QwenConfig>(*gguf_);
    bind();
}

const GgufFile &QwenNativeModel::gguf() const {
    return *gguf_;
}

const NativePlanInfo &QwenNativeModel::plan() const {
    return plan_;
}

const QwenConfig &QwenNativeModel::config() const {
    return *config_;
}

const std::vector<QwenLayerTensors> &QwenNativeModel::layers() const {
    return layers_;
}

const GgufTensorInfo *QwenNativeModel::token_embedding() const {
    return token_embd_;
}

const GgufTensorInfo *QwenNativeModel::output_norm() const {
    return output_norm_;
}

const GgufTensorInfo *QwenNativeModel::output() const {
    return output_;
}

const GgufTensorInfo *QwenNativeModel::optional_tensor(const std::string &name) {
    return gguf_->find_tensor(name);
}

void QwenNativeModel::add_missing(const std::string &name) {
    plan_.missing_tensors.push_back(name);
}

const GgufTensorInfo *QwenNativeModel::require_tensor(const std::string &name) {
    const GgufTensorInfo *t = optional_tensor(name);
    if (!t) add_missing(name);
    count_bound(t);
    return t;
}

const GgufTensorInfo *QwenNativeModel::require_any_tensor(const std::vector<std::string> &names) {
    for (const std::string &name : names) {
        const GgufTensorInfo *t = optional_tensor(name);
        if (t) {
            count_bound(t);
            return t;
        }
    }
    std::ostringstream ss;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i) ss << " | ";
        ss << names[i];
    }
    add_missing(ss.str());
    return nullptr;
}

void QwenNativeModel::count_bound(const GgufTensorInfo *tensor) {
    if (tensor) plan_.n_bound_tensors++;
}

void QwenNativeModel::bind() {
    const ModelInfo info = gguf_->model_info();
    plan_.architecture = info.architecture;
    plan_.n_layers = info.block_count;
    plan_.n_embd = info.embedding_length;
    plan_.n_heads = info.head_count;
    plan_.n_kv_heads = info.head_count_kv;
    plan_.n_ctx_train = info.context_length;
    plan_.n_tensors = info.tensor_count;

    for (const GgufTensorInfo &t : gguf_->tensors()) plan_.tensor_bytes += t.bytes;

    if (plan_.architecture != "qwen35" && plan_.architecture != "qwen3") {
        plan_.missing_tensors.push_back("unsupported architecture: " + plan_.architecture);
        return;
    }
    if (plan_.n_layers == 0 || plan_.n_embd == 0) {
        plan_.missing_tensors.push_back("missing Qwen shape metadata");
        return;
    }

    token_embd_ = require_tensor("token_embd.weight");
    output_norm_ = require_tensor("output_norm.weight");
    output_ = optional_tensor("output.weight");
    if (output_) {
        count_bound(output_);
    } else {
        output_ = token_embd_;
    }

    layers_.resize(plan_.n_layers);
    for (uint32_t i = 0; i < plan_.n_layers; ++i) {
        QwenLayerTensors &l = layers_[i];
        l.attn_norm = require_tensor(layer_tensor(i, "attn_norm.weight"));
        l.ffn_norm = require_any_tensor({
            layer_tensor(i, "ffn_norm.weight"),
            layer_tensor(i, "post_attention_norm.weight"),
        });

        l.attn_q = optional_tensor(layer_tensor(i, "attn_q.weight"));
        if (l.attn_q) {
            count_bound(l.attn_q);
            l.recurrent = false;
            plan_.standard_attention_layers++;
            l.attn_k = require_tensor(layer_tensor(i, "attn_k.weight"));
            l.attn_v = require_tensor(layer_tensor(i, "attn_v.weight"));
            l.attn_q_norm = require_tensor(layer_tensor(i, "attn_q_norm.weight"));
            l.attn_k_norm = require_tensor(layer_tensor(i, "attn_k_norm.weight"));
            l.attn_output = require_tensor(layer_tensor(i, "attn_output.weight"));
        } else {
            l.recurrent = true;
            plan_.recurrent_layers++;
            l.attn_qkv = require_tensor(layer_tensor(i, "attn_qkv.weight"));
            l.attn_gate = require_tensor(layer_tensor(i, "attn_gate.weight"));
            l.ssm_a = require_tensor(layer_tensor(i, "ssm_a"));
            l.ssm_alpha = require_tensor(layer_tensor(i, "ssm_alpha.weight"));
            l.ssm_beta = require_tensor(layer_tensor(i, "ssm_beta.weight"));
            l.ssm_conv1d = require_tensor(layer_tensor(i, "ssm_conv1d.weight"));
            l.ssm_dt_bias = require_tensor(layer_tensor(i, "ssm_dt.bias"));
            l.ssm_norm = require_tensor(layer_tensor(i, "ssm_norm.weight"));
            l.ssm_out = require_tensor(layer_tensor(i, "ssm_out.weight"));
        }
        l.ffn_gate = require_tensor(layer_tensor(i, "ffn_gate.weight"));
        l.ffn_up = require_tensor(layer_tensor(i, "ffn_up.weight"));
        l.ffn_down = require_tensor(layer_tensor(i, "ffn_down.weight"));
    }

    plan_.supported = plan_.missing_tensors.empty();

    add_op(plan_.op_plan, "token_embedding_lookup");
    add_op(plan_.op_plan, "for_each_layer");
    add_op(plan_.op_plan, "rms_norm(attn_norm)");
    add_op(plan_.op_plan, "if standard_attention_layer");
    add_op(plan_.op_plan, "linear_q(attn_q)");
    add_op(plan_.op_plan, "linear_kv(attn_k, attn_v)");
    add_op(plan_.op_plan, "qk_norm");
    add_op(plan_.op_plan, "qwen_rope");
    add_op(plan_.op_plan, "paged_kv_cache_write");
    add_op(plan_.op_plan, "scaled_dot_product_attention");
    add_op(plan_.op_plan, "linear(attn_output)");
    add_op(plan_.op_plan, "else recurrent_deltanet_layer");
    add_op(plan_.op_plan, "linear_qkv(attn_qkv)");
    add_op(plan_.op_plan, "linear_gate(attn_gate)");
    add_op(plan_.op_plan, "causal_conv1d(ssm_conv1d)");
    add_op(plan_.op_plan, "deltanet_scan(ssm_a, ssm_alpha, ssm_beta, ssm_dt)");
    add_op(plan_.op_plan, "rms_norm(ssm_norm)");
    add_op(plan_.op_plan, "linear(ssm_out)");
    add_op(plan_.op_plan, "residual_add");
    add_op(plan_.op_plan, "rms_norm(ffn_norm)");
    add_op(plan_.op_plan, "swiglu(linear_gate, linear_up)");
    add_op(plan_.op_plan, "linear(ffn_down)");
    add_op(plan_.op_plan, "residual_add");
    add_op(plan_.op_plan, "output_rms_norm");
    add_op(plan_.op_plan, "lm_head");
    add_op(plan_.op_plan, "sampler");
}

std::string QwenNativeModel::describe_plan() const {
    std::ostringstream ss;
    ss << "native backend: " << (plan_.supported ? "supported" : "incomplete") << "\n"
       << "architecture: " << plan_.architecture << "\n"
       << "layers: " << plan_.n_layers << "\n"
       << "embedding: " << plan_.n_embd << "\n"
       << "heads: " << plan_.n_heads << "\n"
       << "kv_heads: " << plan_.n_kv_heads << "\n"
       << "context_train: " << plan_.n_ctx_train << "\n"
       << "tensors: " << plan_.n_tensors << "\n"
       << "tensor_bytes: " << plan_.tensor_bytes << "\n"
       << "bound_tensors: " << plan_.n_bound_tensors << "\n";
    if (!plan_.missing_tensors.empty()) {
        ss << "missing_tensors:\n";
        for (const std::string &name : plan_.missing_tensors) ss << "  " << name << "\n";
    }
    ss << "op_plan:\n";
    for (const std::string &op : plan_.op_plan) ss << "  " << op << "\n";
    return ss.str();
}

NativePlanInfo inspect_native_plan(const std::string &path) {
    auto gguf = std::make_unique<GgufFile>(path);
    QwenNativeModel model(std::move(gguf));
    return model.plan();
}

} // namespace qw3
