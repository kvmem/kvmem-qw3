#include "qw3/gguf.hpp"
#include "qw3/tokenizer.hpp"
#include "json.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

// Manual host check against the frozen official Transformers-rendered inputs.
int main(int argc, char **argv) try {
    if (argc != 3) throw std::invalid_argument(
        "usage: qw3-dflash-benchmark-inputs TARGET FROZEN_WORKLOAD_DIR");
    std::unique_ptr<qw3::GgufFile> gguf;
    std::unique_ptr<qw3::QwenTokenizer> tokenizer;
    if (std::filesystem::is_directory(argv[1]))
        tokenizer = std::make_unique<qw3::QwenTokenizer>(std::string(argv[1]));
    else {
        gguf = std::make_unique<qw3::GgufFile>(argv[1]);
        tokenizer = std::make_unique<qw3::QwenTokenizer>(*gguf);
    }
    const std::filesystem::path root(argv[2]);
    std::ifstream manifest_file(root / "manifest.json");
    if (!manifest_file) throw std::runtime_error("missing workload manifest");
    nlohmann::json manifest;
    manifest_file >> manifest;
    uint64_t rows = 0, tokens = 0;
    for (const auto &[name, spec] : manifest.at("datasets").items()) {
        std::ifstream input(root / spec.at("file").get<std::string>());
        if (!input) throw std::runtime_error("missing dataset file: " + name);
        std::string line;
        uint64_t count = 0;
        while (std::getline(input, line)) {
            const auto row = nlohmann::json::parse(line);
            const auto actual = tokenizer->encode(row.at("prompt").get<std::string>(), false);
            const auto expected = row.at("input_ids").get<std::vector<int32_t>>();
            if (actual != expected) {
                size_t at = 0;
                while (at < actual.size() && at < expected.size() && actual[at] == expected[at]) ++at;
                const size_t begin = at > 8 ? at - 8 : 0;
                const std::vector<int32_t> actual_context(actual.begin() + begin,
                    actual.begin() + std::min(actual.size(), at + 12));
                const std::vector<int32_t> expected_context(expected.begin() + begin,
                    expected.begin() + std::min(expected.size(), at + 12));
                std::cerr << "first_difference=" << at << " native_context="
                          << nlohmann::json(tokenizer->decode(actual_context)).dump()
                          << " official_context=" << nlohmann::json(tokenizer->decode(expected_context)).dump()
                          << " native_ids=" << nlohmann::json(actual_context).dump()
                          << " official_ids=" << nlohmann::json(expected_context).dump() << '\n';
                throw std::runtime_error(
                    name + " ordinal=" + row.at("ordinal").dump() + " tokenizer mismatch: native=" +
                    std::to_string(actual.size()) + " official=" + std::to_string(expected.size()));
            }
            ++count; tokens += actual.size();
        }
        if (count != spec.at("frozen_rows").get<uint64_t>())
            throw std::runtime_error("truncated dataset: " + name);
        rows += count;
        std::cout << name << " rows=" << count << " token_ids=exact\n";
    }
    std::cout << "public_workload rows=" << rows << " tokens=" << tokens << " token_ids=exact\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
