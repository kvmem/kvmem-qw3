#include "qw3/model_source.hpp"
#include "qw3/qwen_config.hpp"
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    auto source = qw3::open_model_source(argv[1]);
    qw3::QwenConfig config(source->model_directory());
    if (config.n_layers != 64 || config.head_dim != 256) {
        throw std::runtime_error("unexpected comparison checkpoint geometry");
    }
    auto require = [&](const char *name) {
        auto *tensor = source->find_tensor(name);
        if (!tensor || !tensor->data || tensor->type != qw3::ModelTensorType::BF16)
            throw std::runtime_error(std::string("missing BF16 tensor: ") + name);
        return tensor;
    };
    auto *q = require("blk.3.attn_q.weight");
    if (q->dims != std::vector<uint64_t>{5120, 12288})
        throw std::runtime_error("gated Q projection shape mismatch");
    if (!require("blk.3.attn_q_norm.weight")->add_one ||
        require("blk.0.ssm_norm.weight")->add_one ||
        !require("blk.0.ssm_a")->negative_exp ||
        require("blk.0.attn_qkv.weight")->reorder != qw3::ModelTensorReorder::Rows)
        throw std::runtime_error("HF-to-native normalization/head layout transforms missing");
    require("blk.64.nextn.eh_proj.weight");
    require("token_embd.weight");
    require("output.weight");
    std::cout << "PASS: BF16 source, gated-Q geometry, norm offsets, recurrence layout, MTP mapping\n";
}
