#include "backend.hpp"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace qw3 {
namespace {

std::string shell_quote(const std::string &s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

std::string make_prompt_file(const std::string &prompt) {
    std::array<char, 64> tmpl{};
    std::snprintf(tmpl.data(), tmpl.size(), "/tmp/qw3-prompt-XXXXXX");
    const int fd = mkstemp(tmpl.data());
    if (fd < 0) {
        throw std::runtime_error(std::string("mkstemp failed: ") + std::strerror(errno));
    }
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        close(fd);
        throw std::runtime_error(std::string("fdopen failed: ") + std::strerror(errno));
    }
    if (!prompt.empty() && fwrite(prompt.data(), 1, prompt.size(), fp) != prompt.size()) {
        fclose(fp);
        throw std::runtime_error("failed to write temporary prompt file");
    }
    if (fclose(fp) != 0) {
        throw std::runtime_error("failed to close temporary prompt file");
    }
    return tmpl.data();
}

bool file_exists(const std::string &path) {
    return access(path.c_str(), X_OK) == 0;
}

std::string resolve_completion_binary(const std::string &configured) {
    const std::string suffix = "llama-cli";
    if (configured.size() >= suffix.size() &&
        configured.compare(configured.size() - suffix.size(), suffix.size(), suffix) == 0) {
        std::string candidate = configured;
        candidate.replace(candidate.size() - suffix.size(), suffix.size(), "llama-completion");
        if (configured.find('/') == std::string::npos || file_exists(candidate)) {
            return candidate;
        }
    }
    return configured;
}

class LlamaCliBackend final : public Backend {
public:
    std::string name() const override {
        return "llama-completion";
    }

    void load(const EngineOptions &options) override {
        if (options.model_path.empty()) {
            throw std::invalid_argument("llama.cpp executable backend requires --model");
        }
        options_ = options;
    }

    std::string generate(const std::string &prompt,
                         const GenerationOptions &gen,
                         const TokenCallback &on_text) override {
        const std::string prompt_file = make_prompt_file(prompt);
        const std::string binary = resolve_completion_binary(options_.llama_cli_path);
        std::ostringstream cmd;
        cmd << shell_quote(binary)
            << " -m " << shell_quote(options_.model_path)
            << " --file " << shell_quote(prompt_file)
            << " -n " << gen.max_tokens
            << " -c " << options_.ctx_size
            << " -b " << options_.batch_size
            << " -ngl " << options_.gpu_layers
            << " --temp " << gen.temperature
            << " --top-p " << gen.top_p
            << " --no-display-prompt"
            << " --simple-io"
            << " --single-turn"
            << " --no-perf";
        if (options_.threads > 0) cmd << " --threads " << options_.threads;
        if (gen.seed != 0) cmd << " --seed " << gen.seed;
        if (!options_.verbose) cmd << " 2>/dev/null";

        FILE *pipe = popen(cmd.str().c_str(), "r");
        if (!pipe) {
            unlink(prompt_file.c_str());
            throw std::runtime_error(std::string("failed to run llama.cpp executable: ") + std::strerror(errno));
        }

        std::string result;
        std::array<char, 4096> buf{};
        while (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
            std::string chunk(buf.data());
            result += chunk;
            if (on_text) on_text(chunk);
        }

        const int status = pclose(pipe);
        unlink(prompt_file.c_str());
        if (status == -1) {
            throw std::runtime_error("failed to close llama.cpp executable process");
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            std::ostringstream err;
            err << "llama.cpp executable failed with status " << status;
            throw std::runtime_error(err.str());
        }
        return result;
    }

private:
    EngineOptions options_;
};

} // namespace

std::unique_ptr<Backend> make_llama_cli_backend() {
    return std::make_unique<LlamaCliBackend>();
}

} // namespace qw3
