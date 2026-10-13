#include "speculative_options.hpp"
#include <iostream>
#include <stdexcept>

using qw3::EngineOptions;
using qw3::detail::normalize_speculative_options;

static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void rejects(EngineOptions options, const char *message) {
    try { normalize_speculative_options(options); }
    catch (const std::invalid_argument &) { return; }
    throw std::runtime_error(message);
}

int main() try {
    EngineOptions options;
    normalize_speculative_options(options);
    require(!options.native_mtp_speculate && !options.native_mtp_chain_set,
            "default must preserve legacy configuration");
    options.speculative_method = "mtp";
    normalize_speculative_options(options);
    require(options.native_mtp_speculate && options.native_mtp_chain == 4 &&
            options.native_mtp_chain_set, "explicit MTP must enable its default chain");
    options.native_mtp_chain = 2;
    normalize_speculative_options(options);
    require(options.native_mtp_chain == 2, "explicit chain must survive normalization");
    options.dflash_model_path = "draft";
    rejects(options, "two draft models must be rejected");

    options = EngineOptions{};
    options.speculative_method = "dflash2";
    rejects(options, "DFlash requires a checkpoint");
    options.dflash_model_path = "draft";
    normalize_speculative_options(options);
    normalize_speculative_options(options); // CLI -> server -> engine -> backend
    require(options.native_mtp_chain == 0 && !options.native_mtp_speculate &&
            !qw3::detail::native_mtp_method_allowed(options),
            "DFlash must not allocate/enable native MTP through legacy environment flags");
    options.native_mtp_prefix = true;
    rejects(options, "DFlash cannot populate native MTP prefixes");

    options = EngineOptions{};
    options.speculative_method = "none";
    normalize_speculative_options(options);
    normalize_speculative_options(options);
    require(options.native_mtp_chain_set && options.native_mtp_chain == 0 &&
            !qw3::detail::native_mtp_method_allowed(options), "none must disable legacy MTP");
    options.dflash_model_path = "draft";
    rejects(options, "none cannot load a draft");
    options = EngineOptions{};
    options.speculative_method = "mtp";
    options.native_mtp_chain_set = true;
    options.native_mtp_chain = 0;
    rejects(options, "explicit MTP cannot silently become plain decode");
    options.native_mtp_chain = -1;
    rejects(options, "negative chains must fail");

    options = EngineOptions{};
    options.dflash_model_path = "draft";
    normalize_speculative_options(options);
    require(options.dflash_model_path == "draft" && options.speculative_method == "auto",
            "legacy DFlash flag must remain supported");
    options.native_mtp_speculate = true;
    rejects(options, "legacy incompatible draft flags must fail");
    options = EngineOptions{};
    options.speculative_method = "typo";
    rejects(options, "unknown method must not silently fall back");
    std::cout << "speculative option selection passed\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
