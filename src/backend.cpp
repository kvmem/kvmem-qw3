#include "backend.hpp"
#include "qw3/device_backend.hpp"

#include <stdexcept>

namespace qw3 {

std::unique_ptr<Backend> make_llama_cli_backend();
std::unique_ptr<Backend> make_mock_backend();
std::unique_ptr<Backend> make_qwen_native_backend();

std::string backend_kind_name(BackendKind kind) {
    switch (kind) {
    case BackendKind::Mock:
        return "mock";
    case BackendKind::LlamaCli:
        return "llama-cli";
    case BackendKind::QwenNative:
        return "qwen-native";
    }
    return "unknown";
}

BackendKind parse_backend_kind(const std::string &name) {
    if (name == "mock") return BackendKind::Mock;
    if (name == "llama-cli" || name == "llama.cpp" || name == "llama") {
        return BackendKind::LlamaCli;
    }
    if (name == "qwen-native" || name == "native") return BackendKind::QwenNative;
    throw std::invalid_argument("unknown backend: " + name);
}

LinearBackend parse_linear_backend(const std::string &name) {
    if (name == "auto") return LinearBackend::Auto;
    if (name == "custom") return LinearBackend::Custom;
    if (name == "cublas") return LinearBackend::Cublas;
    throw std::invalid_argument("unknown linear backend: " + name);
}

const char *linear_backend_name(LinearBackend backend) {
    switch (backend) {
    case LinearBackend::Auto: return "auto";
    case LinearBackend::Custom: return "custom";
    case LinearBackend::Cublas: return "cublas";
    }
    return "unknown";
}

std::unique_ptr<Backend> make_backend(BackendKind kind) {
    switch (kind) {
    case BackendKind::Mock:
        return make_mock_backend();
    case BackendKind::LlamaCli:
        return make_llama_cli_backend();
    case BackendKind::QwenNative:
        return make_qwen_native_backend();
    }
    throw std::invalid_argument("unsupported backend");
}

} // namespace qw3
