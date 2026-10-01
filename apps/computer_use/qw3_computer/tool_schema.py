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
         {"name": "act", "description": "Perform ONE action on the selected desktop and return its resulting screenshot. Coordinates are exact screenshot pixels (not normalized). Use a fresh unique action_id. type inserts literal Unicode text; key sends one supported chord. drag holds the left button from (from_x,from_y) to (to_x,to_y) for 100–2000 ms, with an optional SHIFT/CTRL/ALT modifier; both are released before returning. scroll moves the pointer to (x,y) inside the intended content pane and scrolls in the named up/down direction by 1 to 8 wheel notches. open_app starts a named installed application: Notepad, Explorer in the output folder, Edge on the local test form, LibreOffice Writer/Calc/Impress, Thunderbird, GIMP, or WPS Spreadsheets. A rejected action may return a fresh screenshot; use that observation to decide the next action. On uncertain delivery or pause wait for user recovery. Do not blindly repeat actions.",
          "inputSchema": {"type": "object", "properties": {
              "desktop_id": {"type": "string"}, "observation_id": {"type": "string"},
              "action_id": {"type": "string", "minLength": 8, "maxLength": 80, "pattern": "^[a-zA-Z0-9_-]+$"}, "action": ACTION},
              "required": ["desktop_id", "observation_id", "action_id", "action"], "additionalProperties": False},
          "annotations": {"readOnlyHint": False, "idempotentHint": False}}]
