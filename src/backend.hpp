#pragma once

#include "qw3/qw3.hpp"

#include <memory>
#include <string>

namespace qw3 {

class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string name() const = 0;
    virtual void load(const EngineOptions &options) = 0;
    virtual std::string generate(const std::string &prompt,
                                 const GenerationOptions &options,
                                 const TokenCallback &on_text) = 0;
};

std::unique_ptr<Backend> make_backend(BackendKind kind);

} // namespace qw3
