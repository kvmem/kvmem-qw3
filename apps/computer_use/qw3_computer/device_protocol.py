"""Capabilities of local device adapters; agent runtime is intentionally absent."""
PLATFORMS = ("windows", "linux", "android", "macos", "ios")
PROTOCOL_VERSION = "qw3-device/1"


def capabilities(platform):
    if platform not in PLATFORMS:
        raise ValueError("Unsupported device platform")
    supported = platform in {"windows", "linux", "android"}
    actions = ["click", "type", "key", "drag", "scroll", "open_app"] if supported else []
    return {"protocol_version": PROTOCOL_VERSION, "platform": platform, "implemented": supported,
            "actions": actions, "accessibility": {"windows": "UIA", "linux": "AT-SPI", "android": "UiAutomator2"}.get(platform),
            "coordinate_spaces": ["pixels", "normalized_1000"], "condition_basis": "visible_observation",
            "software_workflows": platform == "windows", "input_owner_limit": 1}
