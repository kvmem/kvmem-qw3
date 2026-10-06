"""Versioned device facts and feature admission; no model or OS input code."""
import hashlib
import json


FEATURES = ("observe", "pointer_input", "keyboard_input", "ensure_app",
            "select_window", "owned_modal", "focus_control", "task_resume")


def capability_manifest(capabilities):
    raw = dict(capabilities)
    actions = raw.get("actions", [])
    apps = raw.get("open_apps", [])
    processes = raw.get("app_processes", {})
    available = raw.get("available_now")
    if available is not None and type(available) is not bool:
        raise ValueError("Device availability must be true, false or unknown")
    implemented = {
        "observe": raw.get("implemented") is True,
        "pointer_input": "click" in actions,
        "keyboard_input": "key" in actions or "type" in actions,
        # A launch alias without an observed process mapping cannot be verified.
        "ensure_app": "open_app" in actions and any(processes.get(app) for app in apps),
        "select_window": "activate_window" in actions,
        "owned_modal": raw.get("owned_modal_evidence") is True,
        "focus_control": "focus_control" in actions,
        "task_resume": raw.get("checked_task_resume") is True,
    }
    supplied = raw.get("features", {})
    features = {}
    for name in FEATURES:
        declared = supplied.get(name, {})
        evidence = declared.get("qualification_evidence", [])
        if not isinstance(evidence, list) or any(not isinstance(e, str) or not e for e in evidence):
            raise ValueError("Invalid qualification evidence for " + name)
        supported = implemented[name]
        enabled = supported and declared.get("enabled_for_task", True) is True
        features[name] = {
            "implemented": supported, "enabled_for_task": enabled,
            "available_now": available if enabled else False,
            "qualified_on_platform": supported and declared.get("qualified_on_platform") is True and bool(evidence),
            "qualification_evidence": evidence,
            "limitations": declared.get("limitations", []),
        }
    raw["features"] = features
    # Exclude transient version values; include permissions, scope and features.
    raw.pop("capability_version", None)
    raw["contract_version"] = "qw3-desktop-contract/1"
    raw["capability_version"] = "cap_" + hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    return raw


def admit_task(manifest, required_features, *, qualification_mode=False):
    """Qualification runs may exercise candidates, never unavailable features."""
    for name in required_features:
        if name not in FEATURES:
            raise ValueError("Unknown required desktop feature: " + name)
        feature = manifest["features"][name]
        if not feature["implemented"] or not feature["enabled_for_task"] or feature["available_now"] is False:
            raise ValueError("Required desktop feature unavailable: " + name)
        if not qualification_mode and not feature["qualified_on_platform"]:
            raise ValueError("Required desktop feature not qualified: " + name)
        if not qualification_mode and feature['available_now'] is not True:
            raise ValueError('Required desktop feature availability unknown: ' + name)
    return {"admitted": True, "qualification_mode": qualification_mode,
            "required_features": list(required_features), "capability_version": manifest["capability_version"]}


def target_facts(observation):
    """Expose evidence and its limits without inventing a window or field ID."""
    foreground = observation.get("foreground", {})
    parent = foreground.get("scope_parent")
    return {
        "device_session": {k: observation.get(k) for k in ("device_id", "session_id", "desktop_id")},
        "application": {k: foreground.get(k) for k in ("process", "pid", "instance_id")},
        "window": {k: foreground.get(k) for k in ("handle", "window_identity", "title", "bounds")},
        "owned_modal_parent": parent,
        "focused_control": observation.get("focused_control"),
        "scope": "outside_task_scope" if observation.get("foreground_in_scope") is False or observation.get("accessibility", {}).get("status") == "outside_task_scope" else "driver_evidence_only",
        "identity_scope": "Observation-bound facts; absent native instance or focus evidence is unknown, not verified",
    }
