"""Explicit image-coordinate conversion; never infer units from the numbers."""


def normalized_point(x, y, width, height):
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("Positive integer screenshot dimensions are required")
    if any(type(v) is not int or not 0 <= v <= 1000 for v in (x, y)):
        raise ValueError("normalized_1000 coordinates must be integers from 0 to 1000")
    # 1000 denotes the far image boundary, whose last clickable pixel is size-1.
    return {"x": min(width - 1, (x * width + 500) // 1000),
            "y": min(height - 1, (y * height + 500) // 1000)}


def dispatch_arguments(arguments, observation):
    space = arguments.get("coordinate_space", "pixels")
    if space not in ("pixels", "normalized_1000"):
        raise ValueError("Unknown coordinate_space; use pixels or normalized_1000")
    result = {k: v for k, v in arguments.items() if k != "coordinate_space"}
    if space == "pixels":
        return result, None
    action = dict(arguments["action"])
    if action.get("kind") not in ("click", "scroll", "drag"):
        raise ValueError("normalized_1000 applies only to click, scroll and drag")
    pairs = [("from_x", "from_y"), ("to_x", "to_y")] if action["kind"] == "drag" else [("x", "y")]
    for x, y in pairs:
        point = normalized_point(action[x], action[y], observation["width"], observation["height"])
        action.update({x: point["x"], y: point["y"]})
    result["action"] = action
    mapping = {"coordinate_space": space, "width": observation["width"], "height": observation["height"],
               "observation_id": observation["observation_id"], "input_action": arguments["action"],
               "dispatched_action": action}
    return result, mapping
