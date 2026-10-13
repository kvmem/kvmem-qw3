#pragma once

#include "qw3/dflash_config.hpp"
#include "qw3/model_source.hpp"

#include <map>
#include <memory>
#include <string>

namespace qw3 {

// Owns the mmap lifetime and validates every tensor before any GPU allocation.
class DFlashModel {
public:
    explicit DFlashModel(const std::string &directory);
    const DFlashConfig &config() const { return config_; }
    const ModelTensorInfo &tensor(const std::string &name) const;
    const std::map<std::string, const ModelTensorInfo *> &tensors() const { return tensors_; }
    uint64_t parameter_count() const { return parameters_; }
    uint64_t tensor_bytes() const { return bytes_; }
    std::string identity_sha256() const;

private:
    void bind(const std::string &name, const std::vector<uint64_t> &hf_shape);

    DFlashConfig config_;
    std::unique_ptr<ModelSource> source_;
    std::map<std::string, const ModelTensorInfo *> tensors_;
    uint64_t parameters_ = 0;
    uint64_t bytes_ = 0;
};

} // namespace qw3
