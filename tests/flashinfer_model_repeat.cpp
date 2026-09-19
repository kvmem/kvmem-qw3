#include "qw3/qw3.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

// Optional model-dependent GPU regression; never download weights from CTest.
// Usage: qw3-flashinfer-model-repeat MODEL CHUNK [PROMPT_REPEAT_COUNT ...]
// Compare full greedy output bytes, not just successful exit or token counts.
int main(int argc, char **argv) try {
    if (argc < 3) throw std::runtime_error("MODEL CHUNK [PROMPT_REPEAT_COUNT ...] required");
    setenv("QW3_PREFIX_CACHE", "0", 1);
    setenv("QW3_KVMEM_PREFIX_CACHE", "0", 1);
    qw3::EngineOptions config;
    config.model_path = argv[1];
    config.ctx_size = 24576;
    config.prefill_chunk = std::stoi(argv[2]);
    config.native_mtp_chain = 0;
    config.native_mtp_chain_set = true;
    qw3::Engine engine(config);
    qw3::GenerationOptions gen;
    gen.max_tokens = 128;
    gen.ignore_eos = true;
    gen.temperature = 0;
    gen.top_p = 1;
    gen.top_k = 1;
    gen.seed = 0;
    bool passed = true;
    const int cases = argc == 3 ? 2 : argc - 3;
    for (int arg = 0; arg < cases; ++arg) {
        const int count = argc == 3 ? (arg == 0 ? 384 : 128) : std::stoi(argv[arg + 3]);
        if (count < 0) throw std::runtime_error("negative prompt repeat count");
        std::string prompt;
        for (int i = 0; i < count; ++i)
            prompt += "Virtual memory gives each process an isolated address space and maps virtual pages to physical frames. ";
        prompt += "Explain the main idea in one short paragraph.";
        std::string expected;
        for (int run = 0; run < 4; ++run) {
            const auto start = std::chrono::steady_clock::now();
            const std::string output = engine.generate(prompt, gen);
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            if (run == 0) expected = output;
            const bool equal = !output.empty() && output == expected;
            passed &= equal;
            std::cout << "REPEAT count=" << count << " chunk=" << config.prefill_chunk
                      << " run=" << run << " equal=" << equal << " seconds=" << seconds
                      << "\nOUTPUT_BEGIN\n" << output << "\nOUTPUT_END\n" << std::flush;
        }
    }
    if (!passed) throw std::runtime_error("greedy repeated-request outputs differ");
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
}
