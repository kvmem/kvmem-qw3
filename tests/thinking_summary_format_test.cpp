#include "thinking_compaction.hpp"
#include <iostream>
#include <stdexcept>

using namespace qw3::detail;
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }

int main() {
    const std::vector<uint32_t> prefix{1, 2}, current{41, 42, 99}, suffix{3, 4};
    require(thinking_summary_input(prefix, current, 99, suffix) ==
            std::vector<uint32_t>({1, 2, 41, 42, 3, 4}),
            "summary must contain exactly the current block between its instructions");
    for (const auto &bad : {std::vector<uint32_t>{}, std::vector<uint32_t>{41, 42}}) {
        bool rejected = false;
        try { (void)thinking_summary_input(prefix, bad, 99, suffix); }
        catch (const std::invalid_argument &) { rejected = true; }
        require(rejected, "unclosed thinking must not enter isolated summary");
    }
    require(thinking_summary_prefix().find("no earlier conversation") != std::string::npos,
            "summary prompt must not assume historical context");
    // Guard all partitions, including ordinary BPE pieces rather than a single
    // special token. Main-task identifiers and inequalities must remain legal.
    for (const std::string marker : {"<tool_call>", "</tool_call>", "<think>", "</think>",
                                    "<|im_start|>", "<function=Bash>", "</parameter>", "```"}) {
        require(!thinking_summary_piece_allowed("Memory: ", marker), "whole marker admitted");
        for (size_t split = 1; split < marker.size(); ++split) {
            const auto prefix = "Memory: " + marker.substr(0, split);
            // Some prefixes already contain the forbidden marker; those could
            // never have been committed by the constrained decoder.
            if (thinking_summary_piece_allowed("Memory: ", marker.substr(0, split)))
                require(!thinking_summary_piece_allowed(prefix, marker.substr(split)), "split marker admitted");
        }
    }
    require(thinking_summary_piece_allowed("Keep x ", "< 5; inspect foo::tool_call next."), "ordinary text masked");
    require(thinking_summary_piece_allowed("", "先保留关键结论，测试尚未运行。"), "UTF-8 summary masked");
    require(thinking_summary_has_partial_marker("Read next. <too"), "partial tag not detected");
    require(!thinking_summary_has_partial_marker("Read next. x < 5."), "inequality treated as tag");

    const std::vector<std::string> vocab = {"<tool_call>", "Read ", "<think>", "done", "<|im_end|>"};
    const std::vector<float> logits = {10.f, 8.f, 9.f, 2.f, 3.f};
    auto allowed = [&](int32_t id) { return id == 4 || thinking_summary_piece_allowed("", vocab[id]); };
    require(thinking_summary_argmax(logits, allowed) == 1, "did not choose best permitted token");
    require(thinking_summary_argmax(std::vector<float>{0, 0, 0, 0, 12}, allowed) == 4, "natural stop masked");
    require(thinking_summary_argmax(logits, [](int32_t) { return false; }) == -1, "no-candidate not surfaced");
    std::string reason;
    require(normalize_thinking_summary("  Read src/cache.py next; no tests run.\n", true, &reason)
        == "Read src/cache.py next; no tests run.", "valid memory note rejected");
    require(normalize_thinking_summary("Read next. <too", true, &reason).empty() && reason == "partial-marker",
        "partial marker accepted at EOS");
    require(normalize_thinking_summary("Read next.", false, &reason).empty() && reason == "unterminated",
        "truncated prefix accepted as summary");
    require(normalize_thinking_summary("Done <tool_call>bad</tool_call>", true, &reason).empty(),
        "validator silently salvaged a tool invocation");
    std::cout << "thinking summary constraints passed\n";
}
