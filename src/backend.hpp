#pragma once

#include "qw3/qw3.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace qw3 {

class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string name() const = 0;
    virtual void load(const EngineOptions &options) = 0;
    virtual std::string generate(const std::string &prompt,
                                 const GenerationOptions &options,
                                 const CancellableTokenCallback &on_text) = 0;
    virtual std::string generate_session(const std::string &prompt_fragment,
                                         const GenerationOptions &options,
                                         const TokenCallback &on_text,
                                         bool reset) {
        if (reset) {
            return generate(
                prompt_fragment, options,
                on_text ? CancellableTokenCallback(
                              [on_text](const std::string &piece) {
                                  on_text(piece);
                                  return true;
                              })
                        : CancellableTokenCallback{});
        }
        throw std::runtime_error(
            "persistent session append is unsupported by this backend");
    }
    virtual KvMemLocalCacheInfo kvmem_local_cache_info(
            const std::string &id) {
        (void)id;
        return {};
    }
    virtual bool erase_kvmem_local_cache(const std::string &id) {
        (void)id;
        return false;
    }
};

std::unique_ptr<Backend> make_backend(BackendKind kind);

} // namespace qw3
