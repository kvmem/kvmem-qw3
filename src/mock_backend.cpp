#include "backend.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>

namespace qw3 {
namespace {

class MockBackend final : public Backend {
public:
    std::string name() const override {
        return "mock";
    }

    void load(const EngineOptions &options) override {
        options_ = options;
    }

    std::string generate(const std::string &prompt,
                         const GenerationOptions &options,
                         const TokenCallback &on_text) override {
        (void)options;
        std::ostringstream out;
        out << "[mock backend] prompt_chars=" << prompt.size()
            << " ctx=" << options_.ctx_size
            << " model=" << (options_.model_path.empty() ? "<none>" : options_.model_path)
            << "\n";
        const std::string text = out.str();
        if (on_text) on_text(text);
        return text;
    }

private:
    EngineOptions options_;
};

} // namespace

std::unique_ptr<Backend> make_mock_backend() {
    return std::make_unique<MockBackend>();
}

} // namespace qw3
