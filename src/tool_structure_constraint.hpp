#pragma once

#include "qw3/qw3.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace qw3 {
namespace detail {

struct ToolStructureLegalTokens {
    std::vector<uint32_t> ids;
    double build_ms = 0.0;
};

// Byte-level state machine for the canonical Qwen tool-call form. It is
// tokenizer-independent: callers test a decoded token piece on a copy, then
// commit the piece only after the token is actually accepted.
class ToolStructureConstraint {
public:
    explicit ToolStructureConstraint(
        std::shared_ptr<const ToolStructureConstraintSpec> spec);

    bool enabled() const;
    bool enforcing() const;
    bool requires_full_mask() const;

    bool allows_piece(const std::string &piece) const;
    bool commit_piece(const std::string &piece);

    const std::string &error() const { return error_; }
    std::string state_name() const;
    std::string diagnostic_summary() const;
    const ToolStructureLegalTokens &legal_tokens(
        const std::vector<std::string> &token_pieces,
        bool *cache_hit = nullptr) const;

private:
    enum class State {
        Outside,
        FunctionPrefix,
        FunctionName,
        ParameterOrFunctionEnd,
        ParameterName,
        ParameterValue,
        ToolEnd,
        Failed,
    };

    bool feed(const std::string &piece);
    bool feed_byte(unsigned char byte);
    bool consume_fixed_byte(unsigned char byte, const std::string &target,
                            State next);
    bool update_marker_suffix(unsigned char byte, const std::string &marker,
                              std::string &suffix);
    bool required_parameters_complete() const;
    bool function_name_has_prefix(const std::string &prefix) const;
    bool select_function(const std::string &name);
    bool parameter_name_has_prefix(const std::string &prefix) const;
    bool parameter_name_can_close() const;
    bool select_parameter(const std::string &name);
    bool has_legal_completion() const;
    void invalidate_legal_tokens();
    void begin_tool_call();
    void fail(std::string message);

    std::shared_ptr<const ToolStructureConstraintSpec> spec_;
    State state_ = State::Outside;
    size_t fixed_pos_ = 0;
    size_t function_index_ = static_cast<size_t>(-1);
    std::string outside_suffix_;
    std::string name_buffer_;
    std::string structural_buffer_;
    std::string value_close_suffix_;
    std::vector<bool> seen_parameters_;
    std::shared_ptr<const std::unordered_set<std::string>>
        seen_additional_parameters_ =
            std::make_shared<const std::unordered_set<std::string>>();
    std::string error_;
    mutable const std::vector<std::string> *legal_token_source_ = nullptr;
    mutable std::shared_ptr<const ToolStructureLegalTokens>
        legal_token_cache_;
};

} // namespace detail
} // namespace qw3
