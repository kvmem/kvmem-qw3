"""Positive, session-scoped field references; names and values are not identity.

The first provider is Linux AT-SPI. Resource IDs, coordinates and tree positions
alone cannot mint references. This module sends no input and reads no graders.
"""
from collections import Counter
import hashlib
import json
import re


FIELD_PATTERN = r"^fld_[a-f0-9]{32}$"
PATH_PATTERN = r"/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+"


def provider_identity_from_attrs(attrs):
    bus = attrs.get("qw3-bus-name")
    path = attrs.get("qw3-object-path")
    frame = attrs.get("qw3-window-path")
    pid = attrs.get("qw3-owner-pid")
    if (not isinstance(bus, str) or len(bus) > 64 or not re.fullmatch(r":\d+\.\d+", bus)
            or not isinstance(pid, str) or not pid.isascii() or not pid.isdecimal()
            or len(pid) > 10
            or not 0 < int(pid) < 2**31
            or any(not isinstance(v, str) or len(v) > 1024
                   or not re.fullmatch(PATH_PATTERN, v) for v in (path, frame))):
        return None
    return {"provider": "atspi", "bus_name": bus, "object_path": path,
            "window_path": frame, "owner_pid": int(pid)}


def identity_signature(control):
    identity = control.get("provider_identity")
    required = {"provider", "bus_name", "object_path", "window_path", "owner_pid"}
    if not isinstance(identity, dict) or set(identity) != required or identity["provider"] != "atspi":
        return None
    pid = identity["owner_pid"]
    if type(pid) is not int:
        return None
    validated = provider_identity_from_attrs({"qw3-bus-name": identity["bus_name"],
        "qw3-object-path": identity["object_path"], "qw3-window-path": identity["window_path"],
        "qw3-owner-pid": str(pid)})
    if validated != identity or not isinstance(control.get("kind"), str) or not control["kind"]:
        return None
    return json.dumps([identity, control["kind"]], sort_keys=True, separators=(",", ":"))


def identify_fields(observation, task_id):
    """Only positive native object identity mints a field reference.

    A partial tree may prove an observed object's value. It cannot prove its
    absence. References expire when the task, gateway session, display or native
    window/object identity changes. Mutable names, values and bounds are omitted.
    """
    accessibility = observation.get("accessibility", {})
    raw = accessibility.get("controls", [])
    if not isinstance(raw, list):
        return observation
    controls = [{k: v for k, v in c.items() if k != "field_ref"} for c in raw if isinstance(c, dict)]
    front = observation.get("foreground", {})
    context = [task_id, observation.get("desktop_id"), observation.get("device_id"),
               observation.get("session_id"), observation.get("display_id"), front.get("handle"),
               front.get("process")]
    allowed = (accessibility.get("status") == "ok"
               and observation.get("capabilities", {}).get("platform") == "linux"
               and all(isinstance(v, str) and v for v in context))
    signatures = [identity_signature(c) for c in controls]
    objects = [json.dumps(c["provider_identity"], sort_keys=True) if s is not None else None
               for c, s in zip(controls, signatures)]
    counts = Counter(s for s in objects if s is not None)
    for control, signature, native_object in zip(controls, signatures, objects):
        if (allowed and signature is not None and counts[native_object] == 1
                and control.get("editable") is True and isinstance(control.get("value"), str)):
            encoded = json.dumps([context, signature], separators=(",", ":"), ensure_ascii=True)
            control["field_ref"] = "fld_" + hashlib.sha256(encoded.encode()).hexdigest()[:32]
    return observation | {"accessibility": accessibility | {"controls": controls}}


def field_value_evidence(check, observation):
    accessibility = observation.get("accessibility", {})
    exact = [c for c in accessibility.get("controls", [])
             if c.get("field_ref") == check["field_ref"] and c.get("editable") is True
             and identity_signature(c) is not None]
    passed = (accessibility.get("status") == "ok" and len(exact) == 1
              and exact[0].get("value") == check["value"])
    return {"condition": check, "passed": passed, "observation_id": observation.get("observation_id"),
            "basis": "visible_native_object_value_in_bound_task_and_device_window_session",
            "matching_controls": len(exact),
            "hint": None if passed else
                "The referenced field value is not established in this observation. Observe and inspect the current field; do not infer that delivered input failed or resend it."}


def atspi_identity_script(source):
    """Extend the pinned official reader locally; do not edit the guest server."""
    anchor = " xml=a.lxml.etree.tostring(a.create_atspi_node(frames[0]),encoding='unicode')"
    if source.count(anchor) != 1:
        raise ValueError("Qualified AT-SPI traversal anchor changed")
    injection = '''
 identity_original=a.create_atspi_node
 identity_frame=frames[0]
 try:
  identity_bus=identity_frame.app.bus_name
  identity_pid=identity_frame.get_process_id()
  identity_window=identity_frame.path
 except Exception:
  identity_bus=identity_pid=identity_window=None
 def with_native_identity(node,depth=0,flag=None):
  xml=identity_original(node,depth,flag)
  try:
   if identity_bus and identity_pid and identity_window and node.app.bus_name==identity_bus:
    xml.set('qw3-bus-name',identity_bus)
    xml.set('qw3-owner-pid',str(identity_pid))
    xml.set('qw3-window-path',identity_window)
    xml.set('qw3-object-path',node.path)
  except Exception:
   pass
  return xml
 a.create_atspi_node=with_native_identity
'''
    return source.replace(anchor, injection + anchor, 1)
