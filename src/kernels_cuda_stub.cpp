#include "qw3/device_backend.hpp"

#include <memory>

namespace qw3 {

bool cuda_device_backend_available() {
    return false;
}

std::unique_ptr<DeviceBackend> make_cuda_device_backend(LinearBackend) {
    return nullptr;
}

} // namespace qw3
