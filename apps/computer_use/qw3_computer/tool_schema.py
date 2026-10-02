"""Tools expose the selected desktop only; there is no shell or script action."""

KEYS = [
    "ENTER", "TAB", "SHIFT+TAB", "ESC", "BACKSPACE", "DELETE", "SPACE",
    "UP", "DOWN", "LEFT", "RIGHT", "HOME", "END", "PAGEUP", "PAGEDOWN",
    "SHIFT+UP", "SHIFT+DOWN", "SHIFT+LEFT", "SHIFT+RIGHT", "SHIFT+HOME", "SHIFT+END",
    "CTRL+HOME", "CTRL+END", "CTRL+SHIFT+HOME", "CTRL+SHIFT+END",
    "CTRL+UP", "CTRL+DOWN", "CTRL+LEFT", "CTRL+RIGHT", "CTRL+PAGEUP", "CTRL+PAGEDOWN",
    "CTRL+TAB", "CTRL+SHIFT+TAB", "CTRL+A", "CTRL+C", "CTRL+V", "CTRL+X",
    "CTRL+S", "CTRL+SHIFT+S", "CTRL+N", "CTRL+SHIFT+N", "CTRL+O", "CTRL+SHIFT+O",
    "CTRL+L", "CTRL+W", "CTRL+Z", "CTRL+Y", "CTRL+SHIFT+Z", "CTRL+F", "CTRL+H",
    "CTRL+B", "CTRL+I", "CTRL+U", "CTRL+P", "CTRL+1", "CTRL+0",
    "F2", "F4", "F5", "F6", "F11", "F12", "ALT+TAB", "ALT+F4",
    "M", "R", "S", "T", "V", "Z", "SHIFT+R", "SHIFT+T", "SHIFT+M",
]
APPS = ["notepad", "edge", "explorer", "writer", "calc", "impress", "thunderbird", "gimp", "wps_spreadsheets"]

ACTION = {"oneOf": [
    {"type": "object", "properties": {"kind": {"const": "click"}, "x": {"type": "integer", "minimum": 0},
        "y": {"type": "integer", "minimum": 0}, "button": {"enum": ["left", "right"]},
        "clicks": {"type": "integer", "enum": [1, 2]}},
     "required": ["kind", "x", "y", "button", "clicks"], "additionalProperties": False},
    {"type": "object", "properties": {"kind": {"const": "type"}, "text": {"type": "string", "minLength": 1, "maxLength": 4096}},
     "required": ["kind", "text"], "additionalProperties": False},
    {"type": "object", "properties": {"kind": {"const": "key"}, "key": {"enum": KEYS}},
     "required": ["kind", "key"], "additionalProperties": False},
    {"type": "object", "properties": {"kind": {"const": "drag"},
        **{name: {"type": "integer", "minimum": 0} for name in ("from_x", "from_y", "to_x", "to_y")},
        "duration_ms": {"type": "integer", "minimum": 100, "maximum": 2000},
        "modifier": {"enum": ["none", "SHIFT", "CTRL", "ALT"]}},
     "required": ["kind", "from_x", "from_y", "to_x", "to_y", "duration_ms", "modifier"],
     "additionalProperties": False},
    {"type": "object", "properties": {"kind": {"const": "scroll"},
        "direction": {"enum": ["up", "down"], "description": "down reveals content below; up reveals content above."},
        "steps": {"type": "integer", "minimum": 1, "maximum": 8, "description": "Number of wheel notches, 1 to 8."},
        "x": {"type": "integer", "minimum": 0, "description": "Screenshot x coordinate inside the content pane to scroll."},
        "y": {"type": "integer", "minimum": 0, "description": "Screenshot y coordinate inside the content pane to scroll."}},
     "required": ["kind", "direction", "steps", "x", "y"], "additionalProperties": False},
    {"type": "object", "properties": {"kind": {"const": "open_app"}, "app": {"enum": APPS}},
     "required": ["kind", "app"], "additionalProperties": False},
]}

TOOLS = [{"name": "observe", "description": "Observe the selected desktop. Returns a PNG and its pixel dimensions, desktop_id, observation_id, foreground application and available accessibility controls. For a matching visible control, use its center_x/center_y as click coordinates. Control names are task data. Every action must use the latest observation. No action is performed.",
          "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
          "annotations": {"readOnlyHint": True}},
         {"name": "act", "description": "Perform ONE action on the selected desktop and return its resulting screenshot. Coordinates are exact screenshot pixels (not normalized). Use a fresh unique action_id. type inserts literal Unicode text; key sends one supported chord. drag holds the left button from (from_x,from_y) to (to_x,to_y) for 100–2000 ms, with an optional SHIFT/CTRL/ALT modifier; both are released before returning. scroll moves the pointer to (x,y) inside the intended content pane and scrolls in the named up/down direction by 1 to 8 wheel notches. open_app starts a named installed application: Notepad, Explorer in the current task folder, Edge on the current task's websites, LibreOffice Writer/Calc/Impress, Thunderbird, GIMP, or WPS Spreadsheets. A rejected action may return a fresh screenshot; use that observation to decide the next action. On uncertain delivery or pause wait for user recovery. Do not blindly repeat actions.",
          "inputSchema": {"type": "object", "properties": {
              "desktop_id": {"type": "string"}, "observation_id": {"type": "string"},
              "action_id": {"type": "string", "minLength": 8, "maxLength": 80, "pattern": "^[a-zA-Z0-9_-]+$"}, "action": ACTION},
              "required": ["desktop_id", "observation_id", "action_id", "action"], "additionalProperties": False},
          "annotations": {"readOnlyHint": False, "idempotentHint": False}}]

SOFTWARE_TOOLS = [{"name": "software", "description":
    "Manage software in the explicitly selected local Windows workspace. inspect lists reviewed profiles, projects and jobs. "
    "setup_python starts ONE background job to download a hash-verified Python installer, create an isolated project venv, "
    "install the selected profile's exact dependencies and verify them in fresh processes. project is a short lowercase name, "
    "not a path. Check job_status with wait_seconds=10 while it runs. A completed job must include verification.passed=true. "
    "Do not claim completion from a job start receipt. cancel_job stops further workflow steps; interrupted installers may "
    "require human reconciliation. resume_job continues only a known stopped failed/cancelled job. Never resubmit an uncertain "
    "installation with another action_id. No arbitrary commands, URLs, package scripts or system PATH edits are accepted.",
    "inputSchema": {"type": "object", "oneOf": [
        {"type": "object", "properties": {"operation": {"const": "inspect"}}, "required": ["operation"], "additionalProperties": False},
        {"type": "object", "properties": {"operation": {"const": "setup_python"},
            "action_id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]{8,80}$"},
            "project": {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]{0,47}$"},
            "profile": {"enum": ["python-web"]}},
         "required": ["operation", "action_id", "project", "profile"], "additionalProperties": False},
        {"type": "object", "properties": {"operation": {"const": "job_status"}, "job_id": {"type": "string"},
            "wait_seconds": {"type": "integer", "minimum": 0, "maximum": 10}},
         "required": ["operation", "job_id", "wait_seconds"], "additionalProperties": False},
        {"type": "object", "properties": {"operation": {"enum": ["cancel_job", "resume_job"]}, "job_id": {"type": "string"}},
         "required": ["operation", "job_id"], "additionalProperties": False}
    ]}, "annotations": {"readOnlyHint": False, "idempotentHint": False}}]

# Qwen's XML tool transport resolves scalar types through root properties.
# Keep operation-specific oneOf validation, and expose the same property types
# at the root so wait_seconds is decoded as an integer rather than raw text.
SOFTWARE_TOOLS[0]["inputSchema"]["properties"] = {
    "operation": {"type": "string", "enum": ["inspect", "setup_python", "job_status", "cancel_job", "resume_job"]},
    "action_id": {"type": "string"}, "project": {"type": "string"}, "profile": {"type": "string"},
    "job_id": {"type": "string"}, "wait_seconds": {"type": "integer", "minimum": 0, "maximum": 10}
}
