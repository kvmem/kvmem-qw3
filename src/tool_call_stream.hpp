#pragma once

#include <string>
#include <vector>

namespace qw3 {
namespace detail {

enum class ToolCallStreamEventKind {
    ToolStart,
    ParameterStart,
    ParameterData,
    ParameterEnd,
    ToolEnd,
};

struct ToolCallStreamEvent {
    ToolCallStreamEventKind kind;
    std::string value;
};

// Escape a complete UTF-8 fragment for placement inside an already-open JSON
// string. Fragments can be concatenated directly before the closing quote.
std::string json_string_fragment(const std::string &text);

// Incremental parser for the canonical Qwen tool-call form:
//
//   <tool_call><function=name>
//   <parameter=key>value</parameter>
//   </function></tool_call>
//
// Recovery variants remain handled by qw3_server's existing full-response
// parser. This parser deliberately fails closed when the framing is malformed,
// because bytes already emitted as OpenAI argument deltas cannot be retracted.
class CanonicalToolCallStreamParser {
public:
    bool feed(const std::string &text,
              std::vector<ToolCallStreamEvent> &events);
    bool finish(std::vector<ToolCallStreamEvent> &events);
    bool finish_with_closure_repair(
        std::vector<ToolCallStreamEvent> &events,
        std::string &synthesized_suffix);
    bool finish_with_structural_closure_repair(
        std::vector<ToolCallStreamEvent> &events,
        std::string &synthesized_suffix);

    bool started() const { return started_; }
    bool complete() const { return state_ == State::Complete; }
    bool failed() const { return state_ == State::Failed; }
    const std::string &error() const { return error_; }

private:
    enum class State {
        ToolOpen,
        FunctionOpen,
        ParameterOrFunctionEnd,
        ParameterValue,
        ToolEnd,
        Complete,
        Failed,
    };

    enum class LeadingTrim {
        OptionalLf,
        OptionalCr,
        Done,
    };

    bool parse(std::vector<ToolCallStreamEvent> &events);
    bool finish_with_closure_repair_impl(
        std::vector<ToolCallStreamEvent> &events,
        std::string &synthesized_suffix,
        bool allow_parameter_value_close);
    bool consume_exact(const std::string &tag, const char *label);
    void consume_whitespace();
    void fail(std::string message);
    void process_parameter_data(const std::string &text, bool final,
                                std::vector<ToolCallStreamEvent> &events);
    void emit_parameter_prefix(bool final);
    void emit_safe_parameter_tail(std::vector<ToolCallStreamEvent> &events,
                                  bool final);

    State state_ = State::ToolOpen;
    LeadingTrim leading_trim_ = LeadingTrim::OptionalLf;
    std::string pending_;
    std::string parameter_tail_;
    std::string error_;
    bool started_ = false;
};

} // namespace detail
} // namespace qw3
