"""Opt-in image working set; immutable source history remains retrievable."""
import base64
import hashlib
import io
import json

from PIL import Image

from .observation_history import observation_reference, IDENTITY_KEYS


COMPUTER_TOOLS = {"mcp__computer__observe", "mcp__computer__act", "mcp__computer__workflow", "mcp__computer__history"}
INSTRUCTION = ("Older desktop images may be represented by read-only observation references. "
    "Use history list/read to inspect the original observation when needed, and pin up to four task reference images "
    "before leaving them. User-provided images remain present. A historical observation proves only what was visible then; "
    "it does not prove current focus, state, completion or delivery. Use the latest live observation for all device input.")


def eligible_observation(message, calls):
    if message.get("role") != "tool" or calls.get(message.get("tool_call_id")) not in COMPUTER_TOOLS:
        return None
    parts = message.get("content")
    if not isinstance(parts, list):
        return None
    images = [(i,p) for i,p in enumerate(parts) if isinstance(p,dict) and p.get("type") == "image_url"]
    if len(images) != 1:
        return None
    observations = []
    for part in parts:
        value = part.get("text") if isinstance(part, dict) and part.get("type") == "text" else None
        if not isinstance(value, str) or len(value) > 256*1024 or not value.lstrip().startswith("{"):
            continue
        try: payload = json.loads(value)
        except ValueError: continue
        if isinstance(payload, dict) and isinstance(payload.get("observation"), dict):
            observations.append(payload["observation"])
    if len(observations) != 1:
        return None
    metadata = observations[0]
    reference = observation_reference(metadata)
    if reference is None:
        return None
    index, image = images[0]
    image_url = image.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else None
    if not isinstance(url,str) or not url.startswith("data:image/png;base64,") or len(url)>24*1024*1024:
        return None
    try:
        pixels = base64.b64decode(url.split(",",1)[1],validate=True)
        if hashlib.sha256(pixels).hexdigest() != reference["sha256"]:
            return None
        with Image.open(io.BytesIO(pixels)) as decoded:
            if decoded.format != "PNG" or decoded.size != (metadata.get("width"),metadata.get("height")):
                return None
            decoded.verify()
    except (ValueError, OSError, Image.DecompressionBombError):
        return None
    return {"part_index": index, "reference": reference,
            "historical_read_only": metadata.get("historical_read_only") is True}


def working_context(body, pins=()):
    """Keep two latest live observations and one explicit historical read per
    device/session/display context, plus the newest occurrence of each pin.
    Unknown, ambiguous, foreign-tool and user images are never removed.
    """
    if not isinstance(pins,(list,tuple)) or len(pins)>4 or any(
            not isinstance(p,dict) or set(p)!=set(IDENTITY_KEYS) or observation_reference(p)!=p for p in pins):
        raise ValueError("Use at most four validated task-owned observation references")
    offered = {tool.get("function",{}).get("name") for tool in body.get("tools",[])}
    report = {"policy":"recent-observations-v1", "source_changed":False, "dropped_images":0,
              "retained_image_count":0, "eligible_observation_images":0, "missing_pins":[], "selected":[]}
    report["source_image_count"] = sum(isinstance(p,dict) and p.get("type")=="image_url"
        for m in body["messages"] if isinstance(m.get("content"),list) for p in m["content"])
    report["retained_image_count"] = report["source_image_count"]
    if "mcp__computer__history" not in offered:
        return body, report | {"applied":False,"reason":"history_retrieval_tool_is_unavailable"}
    calls, found, groups = {}, {}, {}
    for index,message in enumerate(body["messages"]):
        for call in message.get("tool_calls",[]):
            calls[call.get("id")] = call.get("function",{}).get("name")
        item = eligible_observation(message,calls)
        if item:
            found[index] = item
            ref = item["reference"]
            scope = tuple(ref[k] for k in IDENTITY_KEYS[:4])
            groups.setdefault((scope,item["historical_read_only"]),[]).append(index)
    report["eligible_observation_images"] = len(found)
    report["missing_pins"] = [p for p in pins if not any(item["reference"]==p for item in found.values())]
    if report["missing_pins"]:
        return body, report | {"applied":False,"reason":"a_pinned_reference_is_missing_from_source_history"}
    keep = set()
    for (_,historical),indices in groups.items():
        keep.update(indices[-(1 if historical else 2):])
    for pin in pins:
        keep.add(max(index for index,item in found.items() if item["reference"]==pin))
    messages = []
    for index,message in enumerate(body["messages"]):
        item = found.get(index)
        if item and index not in keep:
            parts = list(message["content"])
            parts[item["part_index"]] = {"type":"text", "text":json.dumps({
                "archived_desktop_image":item["reference"], "historical_read_only":True,
                "input_target_eligible":False, "retrieve":{"tool":"history","operation":"read",
                    "observation_id":item["reference"]["observation_id"]}},separators=(",",":"))}
            messages.append(message | {"content":parts})
            report["dropped_images"] += 1
        else:
            messages.append(message)
        if item and index in keep:
            report["selected"].append(item["reference"])
    report["retained_image_count"] = sum(isinstance(p,dict) and p.get("type")=="image_url"
        for m in messages if isinstance(m.get("content"),list) for p in m["content"])
    messages.insert(0,{"role":"system","content":INSTRUCTION})
    return body | {"messages":messages}, report | {"applied":True}
