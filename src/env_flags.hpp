#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

namespace qw3 {

inline std::string env_lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline const char *env_value(const char *name) {
    const char *raw = std::getenv(name);
    return raw ? raw : "";
}

inline bool env_value_is_ci(const char *name, const char *expected_lower) {
    return env_lower_ascii(env_value(name)) == expected_lower;
}

inline bool env_disabled_value(const std::string &value) {
    return value == "0" || value == "off" || value == "false" ||
           value == "none" || value == "n" || value == "no";
}

inline bool env_flag_enabled(const char *name, bool default_value = false) {
    const char *raw = std::getenv(name);
    if (!raw || !*raw) return default_value;
    return !env_disabled_value(env_lower_ascii(raw));
}

inline uint32_t env_uint32_or(const char *name, uint32_t default_value) {
    const char *raw = std::getenv(name);
    if (!raw || !*raw) return default_value;

    char *end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (*end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        return default_value;
    }
    return static_cast<uint32_t>(parsed);
}

inline uint64_t env_uint64_or(const char *name, uint64_t default_value) {
    const char *raw = std::getenv(name);
    if (!raw || !*raw) return default_value;

    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (*end != '\0' || parsed > std::numeric_limits<uint64_t>::max()) {
        return default_value;
    }
    return static_cast<uint64_t>(parsed);
}

} // namespace qw3
