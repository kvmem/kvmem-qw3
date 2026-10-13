#pragma once

#include "dflash_model.hpp"
#include "qw3/device_backend.hpp"

#include <map>
#include <memory>
#include <string>

namespace qw3 {

class DFlashWeights {
public:
    DFlashWeights(const DFlashModel &model, DeviceBackend &backend);
    DeviceWeight &at(const std::string &name) const;
    const DFlashConfig &config() const { return config_; }
    uint64_t uploaded_bytes() const { return bytes_; }
    const std::string &identity_sha256() const { return identity_sha256_; }

private:
    DFlashConfig config_;
    std::string identity_sha256_;
    std::map<std::string, std::unique_ptr<DeviceWeight>> weights_;
    uint64_t bytes_ = 0;
};

} // namespace qw3
