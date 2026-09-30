// CPU-only request observer. Reuse the actual server renderer and native
// tokenizer, including the MiniSWE stable-reasoning policy. No model is loaded.
#include "qw3_server.cpp"
#include "qw3/gguf.hpp"

int main(int argc, char **argv) try {
    if (argc != 2) throw std::invalid_argument("usage: RENDER_PROBE TARGET_GGUF");
    qw3::GgufFile model(argv[1]);
    qw3::QwenTokenizer tokenizer(model);
    std::string line;
    while (std::getline(std::cin, line)) {
        const auto request = nlohmann::json::parse(line);
        const auto &messages = request.at("messages");
        if (!messages.is_array() || messages.empty())
            throw std::runtime_error("missing messages");
        for (const auto &m : messages) {
            const auto role = m.at("role").get<std::string>();
            if (role != "system" && role != "developer" && role != "user" &&
                role != "assistant" && role != "tool")
                throw std::runtime_error("unsupported trajectory role: " + role);
        }
        const auto &tools = request.at("tools");
        if (!tools.is_array() || tools.empty())
            throw std::runtime_error("missing original tool schema");
        const auto prompt = qw3::render_messages(messages, &tools, true, {}, nullptr,
            true, false, qw3::detail::HarnessKind::MiniSweAgent, true);
        const auto ids = tokenizer.encode(prompt, false);
        nlohmann::json result{{"prompt_tokens", ids.size()}, {"prompt_bytes", prompt.size()},
            {"renderer", "native_server_mini_swe_stable_reasoning"}};
        if (request.value("include_tokens", false)) result["input_ids"] = ids;
        if (request.value("include_prompt", false)) result["prompt"] = prompt;
        std::cout << result.dump() << std::endl;
    }
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
