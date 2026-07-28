#include "tool_call_stream.hpp"

#include <algorithm>
#include <cctype>
#include <utility>

namespace qw3 {
namespace detail {
namespace {

bool identifier_char(unsigned char c) {
    return std::isalnum(c) || c == '_' || c == '-';
}

bool is_utf8_continuation(unsigned char c) {
    return (c & 0xC0u) == 0x80u;
}

size_t start_of_last_utf8_codepoints(const std::string &text, size_t count) {
    size_t pos = text.size();
    while (count > 0 && pos > 0) {
        --pos;
        while (pos > 0 &&
               is_utf8_continuation(static_cast<unsigned char>(text[pos]))) {
            --pos;
        }
        --count;
    }
    return pos;
}

size_t longest_suffix_prefix(const std::string &text,
                             const std::string &marker) {
    const size_t limit = std::min(text.size(), marker.size() - 1);
    for (size_t size = limit; size > 0; --size) {
        if (text.compare(text.size() - size, size, marker, 0, size) == 0) {
            return size;
        }
    }
    return 0;
}

bool is_prefix_of(const std::string &text, const std::string &candidate) {
    return text.size() <= candidate.size() &&
           candidate.compare(0, text.size(), text) == 0;
}

} // namespace

std::string json_string_fragment(const std::string &text) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(text.size());
    for (unsigned char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20u) {
                    out += "\\u00";
                    out.push_back(hex[(c >> 4) & 0x0Fu]);
                    out.push_back(hex[c & 0x0Fu]);
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    return out;
}

bool CanonicalToolCallStreamParser::feed(
        const std::string &text,
        std::vector<ToolCallStreamEvent> &events) {
    if (failed()) return false;
    if (complete()) return true;
    pending_ += text;
    return parse(events);
}

bool CanonicalToolCallStreamParser::finish(
        std::vector<ToolCallStreamEvent> &events) {
    if (!feed({}, events)) return false;
    if (!complete()) {
        fail("incomplete canonical tool call");
        return false;
    }
    return true;
}

void CanonicalToolCallStreamParser::consume_whitespace() {
    size_t count = 0;
    while (count < pending_.size() &&
           std::isspace(static_cast<unsigned char>(pending_[count]))) {
        ++count;
    }
    pending_.erase(0, count);
}

void CanonicalToolCallStreamParser::fail(std::string message) {
    state_ = State::Failed;
    error_ = std::move(message);
}

bool CanonicalToolCallStreamParser::consume_exact(const std::string &tag,
                                                   const char *label) {
    if (pending_.size() < tag.size()) {
        if (!is_prefix_of(pending_, tag)) {
            fail(std::string("expected ") + label);
        }
        return false;
    }
    if (pending_.compare(0, tag.size(), tag) != 0) {
        fail(std::string("expected ") + label);
        return false;
    }
    pending_.erase(0, tag.size());
    return true;
}

void CanonicalToolCallStreamParser::emit_parameter_prefix(bool final) {
    while (leading_trim_ != LeadingTrim::Done) {
        if (parameter_tail_.empty()) {
            if (!final) return;
            leading_trim_ = LeadingTrim::Done;
            break;
        }
        if (leading_trim_ == LeadingTrim::OptionalLf) {
            if (parameter_tail_.front() == '\n') {
                parameter_tail_.erase(parameter_tail_.begin());
            }
            leading_trim_ = LeadingTrim::OptionalCr;
            continue;
        }
        if (parameter_tail_.front() == '\r') {
            parameter_tail_.erase(parameter_tail_.begin());
        }
        leading_trim_ = LeadingTrim::Done;
    }
}

void CanonicalToolCallStreamParser::emit_safe_parameter_tail(
        std::vector<ToolCallStreamEvent> &events, bool final) {
    if (leading_trim_ != LeadingTrim::Done) return;
    if (final) {
        if (!parameter_tail_.empty() && parameter_tail_.back() == '\n') {
            parameter_tail_.pop_back();
        }
        if (!parameter_tail_.empty() && parameter_tail_.back() == '\r') {
            parameter_tail_.pop_back();
        }
        if (!parameter_tail_.empty()) {
            events.push_back(
                {ToolCallStreamEventKind::ParameterData, parameter_tail_});
            parameter_tail_.clear();
        }
        return;
    }

    const size_t keep_from =
        start_of_last_utf8_codepoints(parameter_tail_, 2);
    if (keep_from == 0) return;
    events.push_back(
        {ToolCallStreamEventKind::ParameterData,
         parameter_tail_.substr(0, keep_from)});
    parameter_tail_.erase(0, keep_from);
}

void CanonicalToolCallStreamParser::process_parameter_data(
        const std::string &text, bool final,
        std::vector<ToolCallStreamEvent> &events) {
    parameter_tail_ += text;
    emit_parameter_prefix(final);
    emit_safe_parameter_tail(events, final);
}

bool CanonicalToolCallStreamParser::parse(
        std::vector<ToolCallStreamEvent> &events) {
    static const std::string tool_open = "<tool_call>";
    static const std::string function_open = "<function=";
    static const std::string parameter_open = "<parameter=";
    static const std::string parameter_close = "</parameter>";
    static const std::string function_close = "</function>";
    static const std::string tool_close = "</tool_call>";

    while (!failed() && !complete()) {
        const size_t before = pending_.size();
        const State before_state = state_;

        if (state_ == State::ToolOpen) {
            consume_whitespace();
            if (!consume_exact(tool_open, "<tool_call>")) break;
            state_ = State::FunctionOpen;
        } else if (state_ == State::FunctionOpen) {
            consume_whitespace();
            if (pending_.size() < function_open.size()) {
                if (!is_prefix_of(pending_, function_open)) {
                    fail("expected <function=...>");
                }
                break;
            }
            if (pending_.compare(0, function_open.size(), function_open) != 0) {
                fail("expected <function=...>");
                break;
            }
            const size_t end = pending_.find('>', function_open.size());
            if (end == std::string::npos) break;
            const std::string name =
                pending_.substr(function_open.size(),
                                end - function_open.size());
            if (name.empty() ||
                !std::all_of(name.begin(), name.end(), [](unsigned char c) {
                    return identifier_char(c);
                })) {
                fail("invalid canonical tool function name");
                break;
            }
            pending_.erase(0, end + 1);
            started_ = true;
            events.push_back({ToolCallStreamEventKind::ToolStart, name});
            state_ = State::ParameterOrFunctionEnd;
        } else if (state_ == State::ParameterOrFunctionEnd) {
            consume_whitespace();
            if (pending_.empty()) break;
            if (pending_.size() >= function_close.size() &&
                pending_.compare(0, function_close.size(), function_close) == 0) {
                pending_.erase(0, function_close.size());
                state_ = State::ToolEnd;
            } else if (pending_.size() >= parameter_open.size() &&
                       pending_.compare(0, parameter_open.size(),
                                        parameter_open) == 0) {
                const size_t end = pending_.find('>', parameter_open.size());
                if (end == std::string::npos) break;
                const std::string name =
                    pending_.substr(parameter_open.size(),
                                    end - parameter_open.size());
                if (name.empty()) {
                    fail("empty canonical tool parameter name");
                    break;
                }
                pending_.erase(0, end + 1);
                parameter_tail_.clear();
                leading_trim_ = LeadingTrim::OptionalLf;
                events.push_back(
                    {ToolCallStreamEventKind::ParameterStart, name});
                state_ = State::ParameterValue;
            } else if (is_prefix_of(pending_, function_close) ||
                       is_prefix_of(pending_, parameter_open)) {
                break;
            } else {
                fail("expected canonical parameter or </function>");
                break;
            }
        } else if (state_ == State::ParameterValue) {
            const size_t close = pending_.find(parameter_close);
            if (close != std::string::npos) {
                process_parameter_data(pending_.substr(0, close), true, events);
                pending_.erase(0, close + parameter_close.size());
                events.push_back(
                    {ToolCallStreamEventKind::ParameterEnd, {}});
                state_ = State::ParameterOrFunctionEnd;
            } else {
                const size_t held =
                    longest_suffix_prefix(pending_, parameter_close);
                const size_t safe = pending_.size() - held;
                if (safe == 0) break;
                process_parameter_data(pending_.substr(0, safe), false, events);
                pending_.erase(0, safe);
            }
        } else if (state_ == State::ToolEnd) {
            consume_whitespace();
            if (!consume_exact(tool_close, "</tool_call>")) break;
            events.push_back({ToolCallStreamEventKind::ToolEnd, {}});
            state_ = State::Complete;
        }

        if (pending_.size() == before && state_ == before_state) break;
    }
    return !failed();
}

} // namespace detail
} // namespace qw3
