#include "qw3/qw3.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    const std::string rendered = qw3::render_qwen3_chat_prompt(
        "system",
        "hello",
        false);
    if (rendered.find("<|im_start|>user") == std::string::npos) {
        throw std::runtime_error("chat template missing user marker");
    }
    if (rendered.find("</think>") == std::string::npos) {
        throw std::runtime_error("nothink prefix missing");
    }

    qw3::EngineOptions options;
    options.backend = qw3::BackendKind::Mock;
    options.model_path = "dummy.gguf";
    qw3::Engine engine(options);
    const std::string out = engine.generate("abc", {});
    if (out.find("prompt_chars=3") == std::string::npos) {
        throw std::runtime_error("mock backend output mismatch");
    }

    std::cout << "qw3 smoke ok\n";
    return 0;
}
