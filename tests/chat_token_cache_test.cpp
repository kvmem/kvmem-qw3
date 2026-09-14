#include "chat_token_cache.hpp"
#include "qw3/gguf.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>

using Clock = std::chrono::steady_clock;
int main(int argc, char **argv) {
    if (argc != 2) throw std::runtime_error("expected GGUF model path");
    qw3::GgufFile gguf(argv[1]);
    qw3::QwenTokenizer tokenizer(gguf);
    qw3::detail::ChatTokenCache cache(tokenizer);
    std::vector<std::string> fragments = {"", "中文 café 👩‍💻\r\n", "word   next\n",
        "<|im_start|>user\n", "<|im_end|>\n", "<|vision_start|><|image_pad|><|vision_end|>",
        "<tool_call>\n{\"x\": 3}\n</tool_call>", "<|im_sta", "rt|>", "    "};
    std::mt19937 rng(13);
    for (int trial = 0; trial < 80; ++trial) {
        std::string text;
        for (int i = 0; i < 150; ++i) text += fragments[rng() % fragments.size()];
        for (int edit = 0; edit < 3; ++edit) {
            if (cache.encode(text) != tokenizer.encode(text))
                throw std::runtime_error("cached token IDs differ");
            text.resize(text.size() - 7); // Includes arbitrary partial UTF-8/sentinel rewrites.
            text += "rewritten <|im_start|>assistant\n";
        }
    }
    qw3::detail::ChatTokenCache tiny(tokenizer, 32);
    std::string visual;
    for (int i = 0; i < 2040; ++i) visual += "<|image_pad|>";
    std::string prompt;
    for (int i = 0; i < 500; ++i) {
        prompt += "<|im_start|>user\n<|vision_start|>" + visual + "<|vision_end|><|im_end|>\n";
        prompt += "<|im_start|>assistant\nStep " + std::to_string(i) + " click.<|im_end|>\n";
    }
    auto started = Clock::now();
    const auto expected = tokenizer.encode(prompt);
    const auto full_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    if (tiny.encode(prompt) != expected || cache.encode(prompt) != expected)
        throw std::runtime_error("long visual history differs");
    started = Clock::now();
    const auto actual = cache.encode(prompt);
    const auto warm_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    if (actual != expected) throw std::runtime_error("warm visual history differs");
    cache.prepare_decoded_pieces();
    for (int32_t id = 0; id < tokenizer.vocab_size(); ++id) {
        if (cache.decode_one(id) != tokenizer.decode_one(id))
            throw std::runtime_error("cached vocabulary bytes differ");
    }
    if (!cache.decode_one(-1).empty() || !cache.decode_one(tokenizer.vocab_size()).empty())
        throw std::runtime_error("invalid token ID did not decode to empty bytes");
    auto map_spans = [&](bool cached) {
        size_t offset = 0;
        const auto start = Clock::now();
        for (int32_t id : actual) {
            const std::string piece = cached ? cache.decode_one(id) : tokenizer.decode_one(id);
            if (prompt.compare(offset, piece.size(), piece) != 0)
                throw std::runtime_error("token-to-byte span map differs");
            offset += piece.size();
        }
        if (offset != prompt.size()) throw std::runtime_error("incomplete span map");
        return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
    };
    const double full_span_ms = map_spans(false), cached_span_ms = map_spans(true);
    std::cout << "token cache: 240 rewrite cases plus capacity fallback and million-token visual history passed\n";
    std::cout << "{\"tokens\":" << expected.size() << ",\"full_ms\":" << full_ms
              << ",\"warm_ms\":" << warm_ms << "}\n";
    std::cout << "{\"full_span_ms\":" << full_span_ms
              << ",\"cached_span_ms\":" << cached_span_ms << "}\n";
}
