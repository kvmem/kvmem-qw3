"""Explain literal observation comparisons without changing their predicates.

Feedback describes available evidence, not input delivery, persisted business
results or inferred app/field identity. This module sends no device/model calls.
"""


def foreground_evidence(observation):
    front = observation.get("foreground", {})
    result, truncated = {}, []
    for key in ("handle", "process", "title"):
        value = front.get(key)
        if isinstance(value, str):
            result[key] = value[:512]
            if len(value) > 512:
                truncated.append(key)
    if truncated:
        result["truncated_fields"] = truncated
    return result


def explain_condition(evidence, observation):
    """Keep the original condition, pass/fail and ambiguity feedback intact."""
    condition = evidence["condition"]
    kind = condition["kind"]
    accessibility = observation.get("accessibility", {})
    available = accessibility.get("status") == "ok"
    complete = available and accessibility.get("truncated") is False
    diagnostic = {"input_delivery_inferred": False,
                  "independent_business_result_verified": False}
    hint = None
    if kind == "foreground_app":
        actual = observation.get("foreground", {}).get("process")
        aliases = observation.get("capabilities", {}).get("app_processes", {})
        aliases = aliases if isinstance(aliases, dict) else {}
        known = [alias for alias, processes in aliases.items()
                 if isinstance(alias, str) and 0 < len(alias) <= 128
                 and isinstance(processes, list) and isinstance(actual, str)
                 and actual in processes]
        diagnostic.update(scope="foreground_application_identity_only",
                          foreground=foreground_evidence(observation),
                          observed_process_capability_aliases=sorted(known)[:16],
                          comparison="exact_observed_process_or_explicit_capability_alias")
        if not evidence["passed"]:
            hint = ("The expected application condition does not match this observation. "
                    "Inspect the actual foreground and its advertised aliases; use an exact observed process "
                    "or a declared app alias only if it is the intended application. "
                    "To revise a pending criterion declare a recovery reason. This does not show that input failed.")
    elif kind == "foreground_title":
        diagnostic.update(scope="visible_window_title_only",
                          foreground=foreground_evidence(observation), comparison="literal_title_contains_text")
    elif kind.startswith("control_"):
        diagnostic.update(accessibility_status=accessibility.get("status"),
                          accessibility_complete=complete,
                          matching_controls=evidence.get("matching_controls"))
        if kind == "control_absent":
            diagnostic["scope"] = "absence_requires_complete_accessibility_read"
            if not evidence["passed"]:
                if evidence.get("matching_controls", 0):
                    diagnostic["reason"] = "matching_control_is_observed"
                else:
                    diagnostic["reason"] = "complete_read_not_established"
                hint = ("Absence is not established. A partial or unavailable control tree cannot prove absence. "
                        "Inspect a positive visible result or obtain a complete read; do not infer failed input or repeat it.")
        elif kind == "control_text":
            diagnostic["scope"] = "positive_visible_control_name_only"
            if not evidence["passed"]:
                hint = ("The requested text is not established in the available control tree. "
                        "It may be visible in the screenshot outside that tree. Inspect the current screen; "
                        "do not treat missing control evidence as failed input.")
        else:
            diagnostic.update(scope="unique_exact_control_name_and_literal_value",
                              comparison="exact_name_optional_exact_kind_and_exact_value")
            if not evidence["passed"]:
                hint = ("The exact named field value is not established. A field name may change with its content. "
                        "Use an observed native field_ref when available for that field, or inspect current controls. "
                        "Coordinates and resource IDs alone do not identify the old field. Do not resend delivered input.")
    elif kind == "screen_changed":
        diagnostic["scope"] = "visual_transition_only_no_focus_or_save_proof"
    elif kind == "field_value":
        diagnostic["scope"] = "visible_native_field_literal_value_only"
    result = evidence | {"comparison_evidence": diagnostic}
    if hint and not result.get("hint"):
        result["hint"] = hint
    return result
