#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace qw3 {

// Automatic harness lifetimes, independent of fixed sink/recent allocation,
// visual/client pins, and the final KV block needed to continue generation.
struct KvMemMandatoryConfig {
    bool system = true;
    bool root = true;
    bool query = true;
    bool policy = true;
    bool tool_call = false;
    bool tool_result = false;

    bool &part(std::string_view name) {
        if (name == "system") return system;
        if (name == "root") return root;
        if (name == "query") return query;
        if (name == "policy") return policy;
        if (name == "tool-call") return tool_call;
        if (name == "tool-result") return tool_result;
        throw std::invalid_argument("unknown mandatory part: " + std::string(name));
    }

    bool protects_current_query(bool root_is_current) const {
        return query || (root && root_is_current);
    }

    std::string parts() const {
        std::string out;
        const auto add = [&](const char *name, bool enabled) {
            if (!enabled) return;
            if (!out.empty()) out += ',';
            out += name;
        };
        add("system", system); add("root", root); add("query", query);
        add("policy", policy); add("tool-call", tool_call);
        add("tool-result", tool_result);
        return out.empty() ? "none" : out;
    }

    // A list replaces the default selection. Presets must stand alone.
    static KvMemMandatoryConfig parse(std::string_view value) {
        const auto trim = [](std::string_view text) {
            const auto first = text.find_first_not_of(" \t\r\n");
            if (first == std::string_view::npos) return std::string_view{};
            const auto last = text.find_last_not_of(" \t\r\n");
            return text.substr(first, last - first + 1);
        };
        value = trim(value);
        if (value == "default") return {};
        KvMemMandatoryConfig out{false, false, false, false, false, false};
        if (value == "none") return out;
        if (value == "all") return {true, true, true, true, true, true};
        if (value.empty()) throw std::invalid_argument("empty mandatory parts list");
        while (true) {
            const auto comma = value.find(',');
            const auto name = trim(value.substr(0, comma));
            if (name.empty()) throw std::invalid_argument("empty mandatory part");
            bool &enabled = out.part(name);
            if (enabled)
                throw std::invalid_argument("duplicate mandatory part: " + std::string(name));
            enabled = true;
            if (comma == std::string_view::npos) break;
            value.remove_prefix(comma + 1);
        }
        return out;
    }
};

} // namespace qw3
