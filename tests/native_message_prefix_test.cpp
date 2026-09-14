#include "native_message_preparation.hpp"
#include <iostream>
#include <random>
#include <stdexcept>

using namespace qw3::detail;
using J = nlohmann::json;
static void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

static J request() {
    return {{"model", "qw3"}, {"max_tokens", 64}, {"stream", true},
            {"system", "Keep the transcript exact."},
            {"messages", J::array({J{{"role", "user"}, {"content", "first 中文 😀"}}})}};
}

static std::shared_ptr<const PreparedMessageCache::Entry> verify(
        const std::string &wire, std::shared_ptr<const PreparedMessageCache::Entry> previous,
        size_t *reused = nullptr, bool *fallback = nullptr) {
    NativePreparedRequest result;
    std::string error, expected_error;
    const bool valid = prepare_native_messages_request(wire, result, error, previous);
    J expected;
    const bool expected_valid = anthropic_request_to_openai(J::parse(wire), expected, expected_error);
    check(valid == expected_valid && error == expected_error, "conversion acceptance/error changed");
    if (!valid) return previous;
    J actual = result.openai;
    J messages = J::array();
    auto entry = std::make_shared<PreparedMessageCache::Entry>();
    if (result.previous) {
        for (size_t i = 0; i < result.prefix_count; ++i) {
            const auto &record = result.previous->messages[i];
            messages.push_back(record->raw);
            entry->messages.push_back(record);
        }
    }
    for (const auto &message : actual["messages"]) {
        messages.push_back(message);
        auto record = std::make_shared<PreparedMessage>();
        record->raw = message;
        record->rendered = message;
        record->account();
        entry->messages.push_back(record);
    }
    actual["messages"] = std::move(messages);
    check(actual == expected, "full converted request changed");
    entry->native = result.snapshot;
    if (entry->native) {
        check(entry->native->messages.size() == entry->native->converted_counts.size(), "mapping count changed");
        size_t count = entry->native->system_count;
        for (const size_t n : entry->native->converted_counts) count += n;
        check(count == entry->messages.size(), "expanded Anthropic/internal counts disagree");
    }
    if (reused) *reused = result.wire_prefix_messages;
    if (fallback) *fallback = result.system_fallback;
    return entry;
}

static void malformed(const std::string &wire, const NativeMessageSnapshot *previous) {
    J expected, actual;
    std::string old_error, new_error;
    try { expected = parse_native_request_json(wire); }
    catch (const J::exception &e) { old_error = e.what(); }
    try {
        auto result = parse_native_request_prefix(wire, previous);
        actual = std::move(result.request);
        if (result.reused_messages) {
            J messages = J::array();
            for (size_t i = 0; i < result.reused_messages; ++i)
                messages.push_back(J::parse(*previous->messages[i]));
            for (const auto &message : actual["messages"]) messages.push_back(message);
            actual["messages"] = std::move(messages);
        }
    } catch (const J::exception &e) { new_error = e.what(); }
    check(old_error == new_error, "JSON error or byte position changed");
    if (old_error.empty()) check(actual == expected, "JSON document changed");
}

int main() {
    try {
        J doc = request();
        auto previous = verify(doc.dump(), {});
        size_t reused = 0;
        doc["messages"].push_back({{"role", "assistant"}, {"content", "next"}});
        auto next = verify(doc.dump(), previous, &reused);
        check(reused == 1 && next->native->messages.front() == previous->native->messages.front(), "append did not share source prefix");
        previous = next;
        doc["messages"].push_back({{"role", "user"}, {"content", J::array({
            J{{"type", "text"}, {"text", "before"}},
            J{{"type", "tool_result"}, {"tool_use_id", "call-1"}, {"content", J::array({
                J{{"type", "image"}, {"source", {{"type", "base64"}, {"media_type", "image/png"},
                    {"data", std::string(8192, 'A') + "\\\"}] messages 中文"}}}}})}},
            J{{"type", "text"}, {"text", "after"}}})}});
        previous = verify(doc.dump(), previous, &reused);
        check(reused == 2 && previous->native->converted_counts.back() == 3, "tool expansion mapping failed");
        verify(doc.dump(), previous, &reused);
        check(reused == 3, "identical request did not reuse whole history");
        doc["messages"][1]["content"] = "tail rewrite";
        previous = verify(doc.dump(), previous, &reused);
        check(reused == 1, "rewrite reused changed message");
        doc["system"] = "new system";
        bool fallback = false;
        previous = verify(doc.dump(), previous, &reused, &fallback);
        check(fallback && reused == 0, "system rewrite reused old system");
        doc["tools"] = J::array({J{{"name", "test"}, {"input_schema", {{"type", "object"}}}}});
        doc["temperature"] = 0.3;
        previous = verify(doc.dump(), previous, &reused);
        check(reused == 3, "metadata unnecessarily disabled source reuse");
        doc["messages"] = J::array({J{{"role", "user"}, {"content", "compacted summary"}}});
        previous = verify(doc.dump(), previous, &reused);
        check(reused == 0 && previous->native->messages.size() == 1, "compaction resurrected history");

        std::mt19937 random(19);
        size_t histories = 0;
        for (int i = 0; i < 300; ++i) {
            if (i % 13 == 0) doc["messages"] = request()["messages"];
            if (i % 7 == 0) doc["messages"].back()["content"] = "rewrite " + std::to_string(i);
            if (i % 17 == 0) doc["system"] = "system " + std::to_string(i);
            if (i % 19 == 0) doc.erase("system");
            doc["messages"].push_back({{"role", i % 2 ? "user" : "assistant"},
                {"content", std::string(random() % 1000, 'A') + "中文 \\ \"\n ]},"}});
            previous = verify(doc.dump(i % 11 ? -1 : 2), previous);
            ++histories;
        }
        const std::string good = doc.dump();
        const auto *snapshot = previous->native.get();
        for (size_t i = 0; i < good.size(); i += 37) malformed(good.substr(0, i), snapshot);
        for (int i = 0; i < 2500; ++i) {
            auto wire = good;
            if (i % 2) wire.insert(random() % wire.size(), 1, char(random() % 256));
            else wire[random() % wire.size()] = char(random() % 256);
            malformed(wire, snapshot);
        }
        for (const auto &wire : {"{\"messages\":[],\"messages\":[1]}", "{\"messages\":[1,]}",
            "{\"messages\":[],}", "{\"messages\":[],\"other\":1e999}",
            "{\"mess\\u0061ges\":[],\"model\":\"qw3\",\"max_tokens\":1}"}) malformed(wire, snapshot);
        verify("{\"model\":\"qw3\",\"max_tokens\":1,\"messages\":[],\"messages\":[{\"role\":\"user\",\"content\":\"last wins\"}]}", previous);
        PreparedMessageCache cache(16u << 20, 2);
        check(cache.store("latest", previous) && cache.latest_native() == previous, "native snapshot lookup failed");
        PreparedMessageCache tiny(1, 2);
        check(!tiny.store("oversized", previous) && !tiny.latest_native(), "native snapshot limit ignored");
        std::cout << "PASS: native Claude Messages parity, append/rewrite/compaction/system/tools, "
                  << histories << " histories, malformed byte/error parity, bounded snapshots\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
