#include "tool_structure_constraint.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace {

using qw3::ToolStructureConstraintSpec;
using qw3::ToolStructureFunction;
using qw3::ToolStructureParameter;
using qw3::detail::ToolStructureConstraint;

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "tool_structure_constraint_test: " << message << "\n";
    std::exit(1);
}

std::shared_ptr<const ToolStructureConstraintSpec> edit_spec(
        bool allow_additional = false) {
    auto spec = std::make_shared<ToolStructureConstraintSpec>();
    ToolStructureFunction edit;
    edit.name = "edit";
    edit.allow_additional_parameters = allow_additional;
    edit.parameters = {
        ToolStructureParameter{"filePath", true},
        ToolStructureParameter{"oldString", true},
        ToolStructureParameter{"newString", true},
    };
    spec->functions.push_back(std::move(edit));
    return spec;
}

void require_commit(ToolStructureConstraint &constraint,
                    const std::string &piece) {
    if (!constraint.commit_piece(piece)) {
        fail("commit failed in state " + constraint.state_name() +
             ": " + constraint.error());
    }
}

void test_valid_edit_and_one_byte_chunks() {
    ToolStructureConstraint constraint(edit_spec());
    require_commit(constraint, "ordinary reasoning\n");
    const std::string call =
        "<tool_call>\n<function=edit>\n"
        "<parameter=filePath>/tmp/a</parameter>\n"
        "<parameter=oldString>old</parameter>\n"
        "<parameter=newString>new</parameter>\n"
        "</function>\n</tool_call>";
    for (char c : call) require_commit(constraint, std::string(1, c));
    if (constraint.enforcing() || constraint.state_name() != "outside") {
        fail("complete edit did not leave constrained mode");
    }
}

void test_unknown_parameter_and_missing_required_are_blocked() {
    ToolStructureConstraint unknown(edit_spec());
    require_commit(unknown, "<tool_call><function=edit>");
    if (unknown.allows_piece("<parameter=content>")) {
        fail("unknown edit parameter was allowed");
    }
    if (!unknown.allows_piece("<parameter=filePath>")) {
        fail("declared edit parameter was blocked");
    }

    ToolStructureConstraint missing(edit_spec());
    require_commit(missing,
                   "<tool_call><function=edit>"
                   "<parameter=filePath>/tmp/a</parameter>");
    if (missing.allows_piece("</function>")) {
        fail("function closed before required edit fields were present");
    }
}

void test_duplicate_and_cross_token_boundaries() {
    ToolStructureConstraint constraint(edit_spec());
    require_commit(constraint, "prefix <tool_");
    require_commit(constraint, "call><func");
    require_commit(constraint, "tion=edit><parameter=file");
    require_commit(constraint, "Path>/tmp/a</para");
    require_commit(constraint, "meter>");
    if (constraint.allows_piece("<parameter=filePath>")) {
        fail("duplicate parameter was allowed");
    }
    require_commit(constraint,
                   "<parameter=oldString>x</parameter>"
                   "<parameter=newString>y</parameter>"
                   "</function></tool_call>");
}

void test_value_marker_lookalike_and_additional_properties() {
    ToolStructureConstraint value(edit_spec());
    require_commit(value,
                   "<tool_call><function=edit>"
                   "<parameter=filePath>/tmp/a</parameter>"
                   "<parameter=oldString>literal </parameterX> text</parameter>"
                   "<parameter=newString>ok</parameter>"
                   "</function></tool_call>");

    ToolStructureConstraint additional(edit_spec(true));
    require_commit(additional, "<tool_call><function=edit>");
    if (!additional.allows_piece("<parameter=content>")) {
        fail("additional parameter was blocked when schema permits it");
    }
}

void test_probe_does_not_mutate_and_mask_states() {
    ToolStructureConstraint constraint(edit_spec());
    if (!constraint.allows_piece("<tool_call>")) {
        fail("valid opener probe failed");
    }
    if (constraint.enforcing()) fail("allows_piece mutated live state");
    require_commit(constraint, "prefix <tool_");
    if (!constraint.requires_full_mask()) {
        fail("partial tool opener did not request guarded token selection");
    }
    require_commit(constraint, "not-a-call ");
    if (constraint.requires_full_mask()) {
        fail("diverging from a partial tool opener did not return to fast mode");
    }
    require_commit(constraint, "<tool_call>");
    if (!constraint.enforcing() || !constraint.requires_full_mask()) {
        fail("function prefix does not request a full mask");
    }
    require_commit(constraint, "<function=edit><parameter=filePath>");
    if (constraint.requires_full_mask()) {
        fail("parameter value should use the fast candidate-validation path");
    }
}

} // namespace

int main() {
    test_valid_edit_and_one_byte_chunks();
    test_unknown_parameter_and_missing_required_are_blocked();
    test_duplicate_and_cross_token_boundaries();
    test_value_marker_lookalike_and_additional_properties();
    test_probe_does_not_mutate_and_mask_states();
    std::cout << "tool structure constraint tests passed\n";
    return 0;
}
