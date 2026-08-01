#include "backend.hpp"
#include "qw3/qw3.hpp"

#include <stdexcept>
#include <utility>

namespace qw3 {

struct Engine::Impl {
    EngineOptions options;
    std::unique_ptr<Backend> backend;
};

Engine::Engine(EngineOptions options) : impl_(std::make_unique<Impl>()) {
    impl_->options = std::move(options);
    impl_->backend = make_backend(impl_->options.backend);
    impl_->backend->load(impl_->options);
}

Engine::~Engine() = default;

const EngineOptions &Engine::options() const {
    return impl_->options;
}

ModelInfo Engine::inspect_model() const {
    if (impl_->options.model_path.empty()) {
        return {};
    }
    return inspect_gguf(impl_->options.model_path);
}

NativePlanInfo Engine::native_plan() const {
    if (impl_->options.model_path.empty()) return {};
    return inspect_native_plan(impl_->options.model_path);
}

std::string Engine::generate(const std::string &prompt, const GenerationOptions &options) {
    return impl_->backend->generate(prompt, options, nullptr);
}

void Engine::generate_stream(const std::string &prompt,
                             const GenerationOptions &options,
                             const TokenCallback &on_text) {
    impl_->backend->generate(
        prompt, options,
        on_text ? CancellableTokenCallback(
                      [&on_text](const std::string &text) {
                          on_text(text);
                          return true;
                      })
                : CancellableTokenCallback{});
}

void Engine::generate_stream_cancellable(
        const std::string &prompt,
        const GenerationOptions &options,
        const CancellableTokenCallback &on_text) {
    impl_->backend->generate(prompt, options, on_text);
}

void Engine::generate_session_stream(const std::string &prompt_fragment,
                                     const GenerationOptions &options,
                                     const TokenCallback &on_text,
                                     bool reset) {
    impl_->backend->generate_session(prompt_fragment, options, on_text, reset);
}

KvMemLocalCacheInfo Engine::kvmem_local_cache_info(const std::string &id) {
    return impl_->backend->kvmem_local_cache_info(id);
}

bool Engine::erase_kvmem_local_cache(const std::string &id) {
    return impl_->backend->erase_kvmem_local_cache(id);
}

std::string render_qwen3_chat_prompt(const std::string &system,
                                     const std::string &user,
                                     bool enable_thinking) {
    std::string prompt;
    if (!system.empty()) {
        prompt += "<|im_start|>system\n";
        prompt += system;
        prompt += "<|im_end|>\n";
    }
    prompt += "<|im_start|>user\n";
    prompt += user;
    prompt += "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n";
    if (!enable_thinking) {
        prompt += "<think>\n\n</think>\n\n";
    }
    return prompt;
}

} // namespace qw3
