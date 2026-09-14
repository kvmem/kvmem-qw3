#pragma once

#include "anthropic_adapter.hpp"
#include "prepared_message_cache.hpp"
#include <chrono>

namespace qw3::detail {

struct NativePreparedRequest {
    nlohmann::json request, openai;
    std::shared_ptr<const PreparedMessageCache::Entry> previous;
    std::shared_ptr<const NativeMessageSnapshot> snapshot;
    size_t prefix_count = 0, wire_prefix_messages = 0, wire_prefix_bytes = 0;
    double parse_ms = 0, convert_ms = 0;
    bool system_fallback = false;
};

// The route passes this result directly to the common prepared-message path.
// The client continues to send an ordinary, complete Anthropic request.
inline bool prepare_native_messages_request(
        const std::string &wire, NativePreparedRequest &out, std::string &error,
        std::shared_ptr<const PreparedMessageCache::Entry> previous = {},
        bool allow_prefix = true, bool allow_fast_json = true) {
    using Clock = std::chrono::steady_clock;
    const auto elapsed = [](Clock::time_point begin) {
        return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    };
    out = {};
    if (previous && (!previous->native || previous->native->messages.size() !=
                     previous->native->converted_counts.size())) previous.reset();
    for (;;) {
        auto begin = Clock::now();
        NativeRequestParse parsed;
        if (allow_prefix) parsed = parse_native_request_prefix(
            wire, previous ? previous->native.get() : nullptr, allow_fast_json);
        else parsed.request = parse_native_request_json(wire, allow_fast_json);
        out.parse_ms += elapsed(begin);
        begin = Clock::now();
        AnthropicMessageMapping mapping;
        nlohmann::json converted;
        const bool ok = anthropic_request_to_openai(parsed.request, converted, error, true, &mapping);
        out.convert_ms += elapsed(begin);
        if (!ok) return false;

        size_t prefix_count = 0;
        if (parsed.reused_messages) {
            const auto &old = *previous->native;
            bool safe = mapping.system_count == old.system_count &&
                        old.system_count <= previous->messages.size();
            for (size_t i = 0; safe && i < mapping.system_count; ++i)
                safe = converted["messages"][i] == previous->messages[i]->raw;
            prefix_count = old.system_count;
            for (size_t i = 0; safe && i < parsed.reused_messages; ++i) {
                safe = old.converted_counts[i] <= previous->messages.size() - prefix_count;
                if (safe) prefix_count += old.converted_counts[i];
            }
            if (!safe) {
                // A changed system prompt needs a complete logical message
                // list. Reparse this same full request; never ask for retransmit.
                previous.reset();
                out.system_fallback = true;
                continue;
            }
            auto &messages = converted["messages"].get_ref<nlohmann::json::array_t &>();
            messages.erase(messages.begin(), messages.begin() + mapping.system_count);
        }
        if (parsed.snapshot) {
            auto &snapshot = *parsed.snapshot;
            snapshot.system_count = mapping.system_count;
            if (parsed.reused_messages) snapshot.converted_counts.assign(
                previous->native->converted_counts.begin(),
                previous->native->converted_counts.begin() + parsed.reused_messages);
            snapshot.converted_counts.insert(snapshot.converted_counts.end(),
                mapping.message_counts.begin(), mapping.message_counts.end());
        }
        out.request = std::move(parsed.request);
        out.openai = std::move(converted);
        out.previous = parsed.reused_messages ? std::move(previous) : nullptr;
        out.prefix_count = prefix_count;
        out.snapshot = std::move(parsed.snapshot);
        out.wire_prefix_messages = parsed.reused_messages;
        out.wire_prefix_bytes = parsed.reused_bytes;
        return true;
    }
}
} // namespace qw3::detail
