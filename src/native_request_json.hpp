#pragma once
#include "json.hpp"
#include <string>
#include <memory>
#include <vector>

namespace qw3::detail {
// Parse strict JSON while retaining nlohmann number types and error behavior.
// The server opts in for large native requests; callers can force the original
// path for correctness and performance comparisons.
nlohmann::json parse_native_request_json(const std::string &wire, bool allow_fast = true);

// Raw source messages are immutable and shared between bounded prepared-cache
// entries. Internal message counts are filled by the Anthropic converter.
struct NativeMessageSnapshot {
    std::vector<std::shared_ptr<const std::string>> messages;
    std::vector<size_t> converted_counts;
    size_t system_count = 0;
    size_t bytes() const {
        size_t result = sizeof(*this) + messages.capacity() * sizeof(messages[0]) +
                        converted_counts.capacity() * sizeof(size_t);
        for (const auto &message : messages) result += sizeof(*message) + message->capacity() + 64;
        return result;
    }
};

struct NativeRequestParse {
    nlohmann::json request;
    std::shared_ptr<NativeMessageSnapshot> snapshot;
    size_t reused_messages = 0, reused_bytes = 0;
};

// Only byte-identical, previously validated complete messages are omitted from
// request["messages"]. All other fields remain present and are parsed normally.
// Unsupported layouts and malformed inputs retain the original parser behavior.
NativeRequestParse parse_native_request_prefix(
    const std::string &wire, const NativeMessageSnapshot *previous = nullptr,
    bool allow_fast = true);
}
