#include "native_request_json.hpp"
#include "yyjson/yyjson.h"
#include <memory>
#include <string>
#include <cstring>
#include <string_view>
#include <unordered_set>

namespace qw3::detail {
namespace {
using json=nlohmann::json;
struct LegacyRequired {};
json convert(yyjson_val *value,unsigned depth) {
    // Keep native recursion bounded; the established parser handles deeper
    // documents and every unsupported/error case with its original behavior.
    if (depth>=256) throw LegacyRequired{};
    switch (yyjson_get_type(value)) {
    case YYJSON_TYPE_NULL:return nullptr;
    case YYJSON_TYPE_BOOL:return yyjson_get_bool(value);
    case YYJSON_TYPE_STR:
        return std::string(yyjson_get_str(value),yyjson_get_len(value));
    case YYJSON_TYPE_RAW: {
        // Preserve signed/unsigned integers, negative zero, large-number
        // handling and floating-point rounding using the original parser.
        const char *first=yyjson_get_raw(value);
        return json::parse(first,first+yyjson_get_len(value));
    }
    case YYJSON_TYPE_ARR: {
        json result=json::array();
        auto &items=result.get_ref<json::array_t &>();
        items.reserve(yyjson_get_len(value));
        size_t index,count;yyjson_val *item;
        yyjson_arr_foreach(value,index,count,item) items.push_back(convert(item,depth+1));
        return result;
    }
    case YYJSON_TYPE_OBJ: {
        json result=json::object();
        size_t index,count;yyjson_val *key,*item;
        yyjson_obj_foreach(value,index,count,key,item)
            result[std::string(yyjson_get_str(key),yyjson_get_len(key))]=convert(item,depth+1);
        return result;
    }
    default:throw LegacyRequired{};
    }
}
} // namespace

json parse_native_request_json(const std::string &wire, bool allow_fast) {
    if (!allow_fast) return json::parse(wire);
    std::unique_ptr<yyjson_doc,decltype(&yyjson_doc_free)> doc(
        yyjson_read(wire.data(),wire.size(),YYJSON_READ_NUMBER_AS_RAW),yyjson_doc_free);
    if (doc) {
        try {return convert(yyjson_doc_get_root(doc.get()),0);}
        catch (const LegacyRequired &) {}
        catch (const json::exception &) {}
        doc.reset();
    }
    return json::parse(wire);
}

namespace {
// Locate source values without allocating their strings. This is a boundary
// scanner, not a replacement JSON validator: every uncached byte is subsequently
// validated by the existing parser. Ambiguous/malformed layouts use that parser
// on the original document, including its original error positions.
class MessageBoundaries {
public:
    MessageBoundaries(const std::string &wire, const NativeMessageSnapshot *old)
        : wire_(wire), old_(old) {}

    bool scan() {
        if (!take('{')) return false;
        std::unordered_set<std::string> keys;
        if (take('}')) return false;
        do {
            space();
            const size_t begin = pos_;
            if (!string()) return false;
            const auto key = json::parse(wire_.substr(begin, pos_ - begin)).get<std::string>();
            if (!keys.insert(key).second || !take(':')) return false;
            if (key == "messages") {
                if (!take('[')) return false;
                array_begin = pos_;
                space();
                if (peek() != ']') {
                    bool matching = old_ != nullptr;
                    do {
                        space();
                        const size_t start = pos_, index = spans.size();
                        if (matching && index < old_->messages.size()) {
                            const auto &source = *old_->messages[index];
                            if (source.size() <= wire_.size() - pos_ &&
                                std::memcmp(wire_.data() + pos_, source.data(), source.size()) == 0) {
                                pos_ += source.size();
                                ++reused;
                                reused_bytes += source.size();
                            } else matching = false;
                        } else matching = false;
                        if (!matching && !value()) return false;
                        spans.emplace_back(start, pos_ - start);
                        space();
                        if (peek() == ']') break;
                        if (!take(',')) return false;
                    } while (true);
                }
                array_end = pos_;
                if (!take(']')) return false;
                found = true;
            } else if (!value()) return false;
            space();
            if (take('}')) break;
            if (!take(',')) return false;
        } while (true);
        space();
        return found && pos_ == wire_.size();
    }

    std::vector<std::pair<size_t, size_t>> spans;
    size_t array_begin = 0, array_end = 0, reused = 0, reused_bytes = 0;
private:
    char peek() const { return pos_ < wire_.size() ? wire_[pos_] : '\0'; }
    void space() {
        while (pos_ < wire_.size() && (wire_[pos_] == ' ' || wire_[pos_] == '\n' ||
               wire_[pos_] == '\r' || wire_[pos_] == '\t')) ++pos_;
    }
    bool take(char ch) {
        space();
        if (pos_ == wire_.size() || wire_[pos_] != ch) return false;
        ++pos_;
        return true;
    }
    bool string() {
        if (pos_ == wire_.size() || wire_[pos_++] != '"') return false;
        while (pos_ < wire_.size()) {
            const size_t next = wire_.find_first_of("\\\"", pos_);
            if (next == std::string::npos) return false;
            if (wire_[next] == '"') { pos_ = next + 1; return true; }
            if (next + 1 == wire_.size()) return false;
            pos_ = next + 2;
        }
        return false;
    }
    bool value(size_t depth = 0) {
        space();
        if (depth >= 256 || pos_ == wire_.size()) return false;
        if (peek() == '"') return string();
        if (peek() == '[' || peek() == '{') {
            const bool object = wire_[pos_++] == '{';
            const char end = object ? '}' : ']';
            if (take(end)) return true;
            do {
                if (object) {
                    space();
                    if (!string() || !take(':')) return false;
                }
                if (!value(depth + 1)) return false;
                if (take(end)) return true;
                if (!take(',')) return false;
            } while (true);
        }
        const size_t begin = pos_;
        while (pos_ < wire_.size()) {
            const char ch = wire_[pos_];
            if (ch == ',' || ch == ']' || ch == '}' || ch == ' ' ||
                ch == '\n' || ch == '\r' || ch == '\t') break;
            ++pos_;
        }
        return pos_ > begin;
    }
    const std::string &wire_;
    const NativeMessageSnapshot *old_;
    size_t pos_ = 0;
    bool found = false;
};
} // namespace

NativeRequestParse parse_native_request_prefix(
        const std::string &wire, const NativeMessageSnapshot *previous, bool allow_fast) {
    NativeRequestParse result;
    MessageBoundaries boundaries(wire, previous);
    bool located = false;
    try { located = boundaries.scan(); }
    catch (const json::exception &) { /* Preserve the original parser error. */ }
    if (!located) {
        result.request = parse_native_request_json(wire, allow_fast);
        return result;
    }
    if (boundaries.reused) {
        const size_t suffix = boundaries.reused < boundaries.spans.size()
            ? boundaries.spans[boundaries.reused].first : boundaries.array_end;
        std::string reduced;
        reduced.reserve(boundaries.array_begin + wire.size() - suffix);
        reduced.append(wire.data(), boundaries.array_begin);
        reduced.append(wire.data() + suffix, wire.size() - suffix);
        try { result.request = parse_native_request_json(reduced, allow_fast); }
        catch (const json::exception &) {
            result.request = parse_native_request_json(wire, allow_fast);
            return result;
        }
    } else result.request = parse_native_request_json(wire, allow_fast);
    // Only fully validated source values may become candidates for later reuse.
    result.snapshot = std::make_shared<NativeMessageSnapshot>();
    result.snapshot->messages.reserve(boundaries.spans.size());
    for (size_t i = 0; i < boundaries.spans.size(); ++i) {
        if (i < boundaries.reused) result.snapshot->messages.push_back(previous->messages[i]);
        else {
            const auto span = boundaries.spans[i];
            result.snapshot->messages.push_back(
                std::make_shared<const std::string>(wire.substr(span.first, span.second)));
        }
    }
    result.reused_messages = boundaries.reused;
    result.reused_bytes = boundaries.reused_bytes;
    return result;
}
}
