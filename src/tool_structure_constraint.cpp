#include "tool_structure_constraint.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <utility>

namespace qw3 {
namespace detail {
namespace {

constexpr const char *kToolOpen = "<tool_call>";
constexpr const char *kFunctionOpen = "<function=";
constexpr const char *kParameterOpen = "<parameter=";
constexpr const char *kParameterClose = "</parameter>";
constexpr const char *kFunctionClose = "</function>";
constexpr const char *kToolClose = "</tool_call>";

bool is_space(unsigned char c) {
    return std::isspace(c) != 0;
}

bool is_prefix_of(const std::string &prefix, const std::string &value) {
    return prefix.size() <= value.size() &&
           value.compare(0, prefix.size(), prefix) == 0;
}

size_t longest_suffix_prefix(const std::string &text,
                             const std::string &marker) {
    const size_t limit = std::min(text.size(), marker.size() - 1);
    for (size_t n = limit; n > 0; --n) {
        if (text.compare(text.size() - n, n, marker, 0, n) == 0) {
            return n;
        }
    }
    return 0;
}

} // namespace

ToolStructureConstraint::ToolStructureConstraint(
        std::shared_ptr<const ToolStructureConstraintSpec> spec)
    : spec_(std::move(spec)) {}

bool ToolStructureConstraint::enabled() const {
    return spec_ && !spec_->functions.empty();
}

bool ToolStructureConstraint::enforcing() const {
    return enabled() && state_ != State::Outside && state_ != State::Failed;
}

bool ToolStructureConstraint::requires_full_mask() const {
    if (!enabled() || state_ == State::Failed) return false;
    if (state_ == State::Outside) return !outside_suffix_.empty();
    return state_ != State::ParameterValue;
}

bool ToolStructureConstraint::allows_piece(const std::string &piece) const {
    if (!enabled()) return true;
    if (piece.empty()) return !enforcing();
    ToolStructureConstraint probe(*this);
    return probe.feed(piece);
}

bool ToolStructureConstraint::commit_piece(const std::string &piece) {
    if (!enabled()) return true;
    if (piece.empty() && enforcing()) {
        fail("empty token inside canonical tool call");
        return false;
    }
    return feed(piece);
}

void ToolStructureConstraint::fail(std::string message) {
    state_ = State::Failed;
    error_ = std::move(message);
}

void ToolStructureConstraint::begin_tool_call() {
    state_ = State::FunctionPrefix;
    fixed_pos_ = 0;
    function_index_ = static_cast<size_t>(-1);
    name_buffer_.clear();
    structural_buffer_.clear();
    value_close_suffix_.clear();
    seen_parameters_.clear();
    seen_additional_parameters_.clear();
}

bool ToolStructureConstraint::update_marker_suffix(
        unsigned char byte, const std::string &marker, std::string &suffix) {
    suffix.push_back(static_cast<char>(byte));
    if (suffix == marker) {
        suffix.clear();
        return true;
    }
    const size_t keep = longest_suffix_prefix(suffix, marker);
    if (keep == 0) {
        suffix.clear();
    } else if (keep < suffix.size()) {
        suffix.erase(0, suffix.size() - keep);
    }
    return false;
}

bool ToolStructureConstraint::consume_fixed_byte(
        unsigned char byte, const std::string &target, State next) {
    if (fixed_pos_ == 0 && is_space(byte)) return true;
    if (fixed_pos_ >= target.size() ||
        byte != static_cast<unsigned char>(target[fixed_pos_])) {
        fail("unexpected byte while generating canonical tool tag");
        return false;
    }
    ++fixed_pos_;
    if (fixed_pos_ == target.size()) {
        fixed_pos_ = 0;
        state_ = next;
        name_buffer_.clear();
    }
    return true;
}

bool ToolStructureConstraint::function_name_has_prefix(
        const std::string &prefix) const {
    if (!spec_) return false;
    for (const ToolStructureFunction &function : spec_->functions) {
        if (is_prefix_of(prefix, function.name)) return true;
    }
    return false;
}

bool ToolStructureConstraint::select_function(const std::string &name) {
    if (!spec_) return false;
    for (size_t i = 0; i < spec_->functions.size(); ++i) {
        if (spec_->functions[i].name != name) continue;
        function_index_ = i;
        seen_parameters_.assign(
            spec_->functions[i].parameters.size(), false);
        seen_additional_parameters_.clear();
        state_ = State::ParameterOrFunctionEnd;
        structural_buffer_.clear();
        name_buffer_.clear();
        return true;
    }
    fail("tool function is not present in the request schema");
    return false;
}

bool ToolStructureConstraint::required_parameters_complete() const {
    if (!spec_ || function_index_ >= spec_->functions.size()) return false;
    const auto &parameters = spec_->functions[function_index_].parameters;
    for (size_t i = 0; i < parameters.size(); ++i) {
        if (parameters[i].required &&
            (i >= seen_parameters_.size() || !seen_parameters_[i])) {
            return false;
        }
    }
    return true;
}

bool ToolStructureConstraint::parameter_name_has_prefix(
        const std::string &prefix) const {
    if (!spec_ || function_index_ >= spec_->functions.size()) return false;
    const ToolStructureFunction &function = spec_->functions[function_index_];
    if (function.allow_additional_parameters) {
        return prefix.size() <= 128 &&
               std::none_of(prefix.begin(), prefix.end(), [](unsigned char c) {
                   return is_space(c) || c == '<' || c == '>';
               });
    }
    for (size_t i = 0; i < function.parameters.size(); ++i) {
        if (i < seen_parameters_.size() && seen_parameters_[i]) continue;
        if (is_prefix_of(prefix, function.parameters[i].name)) return true;
    }
    return false;
}

bool ToolStructureConstraint::select_parameter(const std::string &name) {
    if (!spec_ || function_index_ >= spec_->functions.size() || name.empty()) {
        fail("invalid canonical tool parameter name");
        return false;
    }
    const ToolStructureFunction &function = spec_->functions[function_index_];
    for (size_t i = 0; i < function.parameters.size(); ++i) {
        if (function.parameters[i].name != name) continue;
        if (i < seen_parameters_.size() && seen_parameters_[i]) {
            fail("duplicate canonical tool parameter");
            return false;
        }
        seen_parameters_[i] = true;
        state_ = State::ParameterValue;
        value_close_suffix_.clear();
        name_buffer_.clear();
        return true;
    }
    if (!function.allow_additional_parameters ||
        seen_additional_parameters_.count(name) != 0) {
        fail(function.allow_additional_parameters
                 ? "duplicate additional canonical tool parameter"
                 : "tool parameter is not present in the request schema");
        return false;
    }
    seen_additional_parameters_.insert(name);
    state_ = State::ParameterValue;
    value_close_suffix_.clear();
    name_buffer_.clear();
    return true;
}

bool ToolStructureConstraint::feed_byte(unsigned char byte) {
    if (state_ == State::Failed) return false;
    if (state_ == State::Outside) {
        if (update_marker_suffix(byte, kToolOpen, outside_suffix_)) {
            begin_tool_call();
        }
        return true;
    }
    if (state_ == State::FunctionPrefix) {
        return consume_fixed_byte(byte, kFunctionOpen, State::FunctionName);
    }
    if (state_ == State::FunctionName) {
        if (byte == '>') {
            if (name_buffer_.empty() ||
                !function_name_has_prefix(name_buffer_)) {
                fail("invalid canonical tool function name");
                return false;
            }
            return select_function(name_buffer_);
        }
        name_buffer_.push_back(static_cast<char>(byte));
        if (!function_name_has_prefix(name_buffer_)) {
            fail("tool function is not present in the request schema");
            return false;
        }
        return true;
    }
    if (state_ == State::ParameterOrFunctionEnd) {
        if (structural_buffer_.empty() && is_space(byte)) return true;
        structural_buffer_.push_back(static_cast<char>(byte));
        const bool parameter_prefix =
            is_prefix_of(structural_buffer_, kParameterOpen);
        const bool close_prefix = required_parameters_complete() &&
            is_prefix_of(structural_buffer_, kFunctionClose);
        if (structural_buffer_ == kParameterOpen) {
            state_ = State::ParameterName;
            structural_buffer_.clear();
            name_buffer_.clear();
            return true;
        }
        if (required_parameters_complete() &&
            structural_buffer_ == kFunctionClose) {
            state_ = State::ToolEnd;
            structural_buffer_.clear();
            fixed_pos_ = 0;
            return true;
        }
        if (!parameter_prefix && !close_prefix) {
            fail(required_parameters_complete()
                     ? "expected canonical parameter or </function>"
                     : "required tool parameter is missing");
            return false;
        }
        return true;
    }
    if (state_ == State::ParameterName) {
        if (byte == '>') return select_parameter(name_buffer_);
        name_buffer_.push_back(static_cast<char>(byte));
        if (!parameter_name_has_prefix(name_buffer_)) {
            fail("tool parameter is not present in the request schema");
            return false;
        }
        return true;
    }
    if (state_ == State::ParameterValue) {
        if (update_marker_suffix(byte, kParameterClose,
                                 value_close_suffix_)) {
            state_ = State::ParameterOrFunctionEnd;
            structural_buffer_.clear();
        }
        return true;
    }
    if (state_ == State::ToolEnd) {
        if (!consume_fixed_byte(byte, kToolClose, State::Outside)) {
            return false;
        }
        if (state_ == State::Outside) outside_suffix_.clear();
        return true;
    }
    return false;
}

bool ToolStructureConstraint::feed(const std::string &piece) {
    if (!enabled()) return true;
    for (unsigned char byte : piece) {
        if (!feed_byte(byte)) return false;
    }
    return state_ != State::Failed;
}

std::string ToolStructureConstraint::state_name() const {
    switch (state_) {
        case State::Outside: return "outside";
        case State::FunctionPrefix: return "function_prefix";
        case State::FunctionName: return "function_name";
        case State::ParameterOrFunctionEnd: return "parameter_or_function_end";
        case State::ParameterName: return "parameter_name";
        case State::ParameterValue: return "parameter_value";
        case State::ToolEnd: return "tool_end";
        case State::Failed: return "failed";
    }
    return "unknown";
}

} // namespace detail
} // namespace qw3
