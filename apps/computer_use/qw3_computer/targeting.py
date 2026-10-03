"""Resolve observation-bound accessibility targets without guessing coordinates."""
import hashlib
import json


class TargetError(ValueError):
    def __init__(self, message):
        self.problem = {"code": "invalid_control_target", "path": "$.action.control_id",
            "message": message,
            "hint": "Observe again, check the screenshot, and copy a matching control_id from that observation. "
                    "If no matching control is exposed, use an explicit normalized_1000 visual action."}
        super().__init__(message)


def safe_center(control, width, height):
    fields = ("x", "y", "width", "height", "center_x", "center_y")
    if not isinstance(control, dict) or any(type(control.get(k)) is not int for k in fields):
        return False
    x, y, w, h, cx, cy = (control[k] for k in fields)
    return (w > 0 and h > 0 and 0 <= cx < width and 0 <= cy < height
            and x <= cx < x + w and y <= cy < y + h)


def identify_controls(observation):
    """IDs identify one control in one exact observation, never a name match."""
    accessibility = observation.get("accessibility")
    if not isinstance(accessibility, dict):
        return observation
    width, height = observation.get("width"), observation.get("height")
    controls = []
    raw_controls = accessibility.get("controls", [])
    for index, original in enumerate(raw_controls if isinstance(raw_controls, list) else []):
        if not isinstance(original, dict):
            continue
        control = {k: v for k, v in original.items() if k != "control_id"}
        if (accessibility.get("status") == "ok" and type(width) is int and type(height) is int
                and safe_center(control, width, height)):
            identity = json.dumps([observation["desktop_id"], observation["observation_id"], index], separators=(",", ":"))
            control["control_id"] = "ctl_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        controls.append(control)
    return observation | {"accessibility": accessibility | {"controls": controls}}


def resolve_control(arguments, observation):
    if arguments.get("coordinate_space") not in (None, "pixels"):
        raise TargetError("click_control uses the observed control's pixel bounds; do not supply normalized coordinates.")
    accessibility = observation.get("accessibility", {})
    if not isinstance(accessibility, dict) or accessibility.get("status") != "ok":
        raise TargetError("Accessibility controls are unavailable for this observation.")
    control_id = arguments["action"]["control_id"]
    matches = [c for c in accessibility.get("controls", []) if c.get("control_id") == control_id]
    if len(matches) != 1:
        raise TargetError("The control_id is not uniquely present in the latest observation.")
    control = matches[0]
    if not safe_center(control, observation["width"], observation["height"]):
        raise TargetError("The observed control has invalid bounds; no input was sent.")
    action = {"kind": "click", "x": control["center_x"], "y": control["center_y"], "button": "left", "clicks": 1}
    dispatched = {k: v for k, v in arguments.items() if k != "coordinate_space"} | {"action": action}
    mapping = {"source": "accessibility_control", "coordinate_space": "pixels",
        "observation_id": observation["observation_id"], "control_id": control_id,
        "control": control, "input_action": arguments["action"], "dispatched_action": action,
        "width": observation["width"], "height": observation["height"]}
    return dispatched, mapping
