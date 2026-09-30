#pragma once

#include "qw3/qw3.hpp"
#include <stdexcept>

namespace qw3::detail {

// Keep legacy flags/environment behavior when no method was selected. An
// explicit method is authoritative over legacy MTP environment switches.
inline bool native_mtp_method_allowed(const EngineOptions &options) {
    return options.speculative_method == "auto" || options.speculative_method == "mtp";
}

inline void normalize_speculative_options(EngineOptions &options) {
    const auto &method = options.speculative_method;
    if (method != "auto" && method != "none" && method != "mtp" && method != "dflash2")
        throw std::invalid_argument("--speculative-method must be none|mtp|dflash2 (or auto for legacy flags)");
    if (options.native_mtp_chain_set && options.native_mtp_chain < 0)
        throw std::invalid_argument("--mtp-chain must be >= 0");

    const bool native_requested = options.native_mtp_trace || options.native_mtp_prefix ||
        options.native_mtp_speculate ||
        (options.native_mtp_chain_set && options.native_mtp_chain > 0);
    if (!options.dflash_model_path.empty() && native_requested)
        throw std::invalid_argument("DFlash 2 and native MTP are alternatives; remove the native MTP flags or --dflash-model");
    if (method == "auto") return;

    if (method == "mtp") {
        if (!options.dflash_model_path.empty())
            throw std::invalid_argument("--speculative-method mtp cannot use --dflash-model");
        if (options.native_mtp_trace)
            throw std::invalid_argument("--speculative-method mtp cannot use --native-mtp-trace; omit the method for diagnostics");
        if (options.native_mtp_chain_set && options.native_mtp_chain == 0)
            throw std::invalid_argument("--speculative-method mtp requires --mtp-chain > 0");
        if (!options.native_mtp_chain_set) options.native_mtp_chain = 4;
        options.native_mtp_chain_set = true;
        options.native_mtp_speculate = true;
        return;
    }

    if (native_requested)
        throw std::invalid_argument("--speculative-method " + method + " cannot use native MTP flags");
    if (options.mtp_policy != "fixed" || options.mtp_adaptive_min_chain != 0 ||
        options.mtp_adaptive_max_chain != 0)
        throw std::invalid_argument("MTP depth policy options require --speculative-method mtp");
    if (method == "none" && !options.dflash_model_path.empty())
        throw std::invalid_argument("--speculative-method none cannot use --dflash-model");
    if (method == "dflash2" && options.dflash_model_path.empty())
        throw std::invalid_argument("--speculative-method dflash2 requires --dflash-model PATH");
    options.native_mtp_chain = 0;
    options.native_mtp_chain_set = true;
    options.native_mtp_speculate = false;
}

} // namespace qw3::detail
