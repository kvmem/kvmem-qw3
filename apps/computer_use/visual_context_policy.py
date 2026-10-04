"""Matched full/recent presentation with identical history tools and guidance."""
from .visual_working_context import working_context, eligible_observation


def select(body, pins=(), policy="recent-observations"):
    if policy not in ("recent-observations", "full-retrievable"):
        raise ValueError("Declare the retrievable image comparison policy")
    view, report = working_context(body, pins)
    report = report | {"policy": policy, "comparison_tools_and_guidance_matched": True}
    if not report["applied"]:
        return view, report
    # Both arms perform identical source validation and carry the same guidance.
    # Run the same identity/image validation pass in both arms; only the full
    # arm retains every original content part, including all images.
    calls, selected = {}, []
    for message in body["messages"]:
        for call in message.get("tool_calls", []):
            calls[call.get("id")] = call.get("function", {}).get("name")
        item = eligible_observation(message, calls)
        if item:
            selected.append(item["reference"])
    if policy == "recent-observations":
        return view, report
    return body | {"messages": [view["messages"][0], *body["messages"]]}, report | {
        "dropped_images": 0, "retained_image_count": report["source_image_count"],
        "working_selection_preview": report["selected"], "selected": selected}
