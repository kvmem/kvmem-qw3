#pragma once

#include "qw3/kvmem_mandatory.hpp"
#include "json.hpp"

namespace qw3::detail {

// Lists/strings replace the selection; an object overrides individual parts
// from the server default. Empty arrays/objects have distinct, explicit meaning.
inline KvMemMandatoryConfig parse_kvmem_mandatory_parts(
        const nlohmann::json &value, const KvMemMandatoryConfig &fallback) {
    if (value.is_string()) {
        const auto text = value.get<std::string>();
        const auto first = text.find_first_not_of(" \t\r\n");
        const auto last = text.find_last_not_of(" \t\r\n");
        const auto trimmed = first == std::string::npos ? std::string_view{}
            : std::string_view(text).substr(first, last - first + 1);
        return trimmed == "default" ? fallback : KvMemMandatoryConfig::parse(trimmed);
    }
    if (value.is_array()) {
        if (value.empty()) return KvMemMandatoryConfig::parse("none");
        KvMemMandatoryConfig out = KvMemMandatoryConfig::parse("none");
        for (const auto &name : value) {
            if (!name.is_string())
                throw std::invalid_argument("mandatory parts array must contain strings");
            const auto text = name.get<std::string>();
            bool &enabled = out.part(text);
            if (enabled)
                throw std::invalid_argument("duplicate mandatory part: " + text);
            enabled = true;
        }
        return out;
    }
    if (value.is_object()) {
        auto out = fallback;
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (!it.value().is_boolean())
                throw std::invalid_argument("mandatory part switches must be booleans");
            out.part(it.key()) = it.value().get<bool>();
        }
        return out;
    }
    throw std::invalid_argument("kvmem_mandatory_parts must be a string, array or object");
}

} // namespace qw3::detail
