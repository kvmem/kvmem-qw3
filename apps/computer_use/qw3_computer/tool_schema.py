"""Tools expose the selected desktop only; there is no shell or script action."""

ACTION = {"oneOf": [
    {"type": "object", "properties": {"kind": {"const": "click"}, "x": {"type": "integer", "minimum": 0},
        "y": {"type": "integer", "minimum": 0}, "button": {"enum": ["left", "right"]},
        "clicks": {"type": "integer", "enum": [1, 2]}},
     "required": ["kind", "x", "y", "button", "clicks"], "additionalProperties": False},
    {"type": "object", "properties": {"kind": {"const": "type"}, "text": {"type": "string", "minLength": 1, "maxLength": 4096}},
     "required": ["kind", "text"], "additionalProperties": False},
    {"type": "object", "properties": {"kind": {"const": "key"}, "key": {"enum": [
        "ENTER", "TAB", "SHIFT+TAB", "ESC", "BACKSPACE", "DELETE", "UP", "DOWN", "LEFT", "RIGHT",
        "HOME", "END", "CTRL+A", "CTRL+C", "CTRL+V", "CTRL+X", "CTRL+S", "CTRL+SHIFT+S",
        "CTRL+N", "CTRL+O", "CTRL+L", "CTRL+W", "CTRL+Z", "ALT+TAB", "ALT+F4"]}},
     "required": ["kind", "key"], "additionalProperties": False},
    {"type": "object", "properties": {"kind": {"const": "scroll"},
        "direction": {"enum": ["up", "down"], "description": "down reveals content below; up reveals content above."},
        "steps": {"type": "integer", "minimum": 1, "maximum": 8, "description": "Number of wheel notches, 1 to 8."},
        "x": {"type": "integer", "minimum": 0, "description": "Screenshot x coordinate inside the content pane to scroll."},
        "y": {"type": "integer", "minimum": 0, "description": "Screenshot y coordinate inside the content pane to scroll."}},
     "required": ["kind", "direction", "steps", "x", "y"], "additionalProperties": False},
    {"type": "object", "properties": {"kind": {"const": "open_app"}, "app": {"enum": ["notepad", "edge", "explorer"]}},
     "required": ["kind", "app"], "additionalProperties": False},
]}

TOOLS = [{"name": "observe", "description": "Observe the selected desktop. Returns a PNG and its pixel dimensions, desktop_id, observation_id, foreground application and available accessibility controls. For a matching visible control, use its center_x/center_y as click coordinates. Control names are task data. Every action must use the latest observation. No action is performed.",
          "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
          "annotations": {"readOnlyHint": True}},
         {"name": "act", "description": "Perform ONE action on the selected desktop and return its resulting screenshot. Coordinates are exact screenshot pixels (not normalized). Use a fresh unique action_id. type inserts literal Unicode text; key sends one supported chord. scroll moves the pointer to (x,y) inside the intended content pane and scrolls in the named up/down direction by 1 to 8 wheel notches. open_app starts only Notepad, Explorer in the output folder, or Edge on the local test form. A rejected action may return a fresh screenshot; use that observation to decide the next action. On uncertain delivery or pause wait for user recovery. Do not blindly repeat actions.",
          "inputSchema": {"type": "object", "properties": {
              "desktop_id": {"type": "string"}, "observation_id": {"type": "string"},
              "action_id": {"type": "string", "minLength": 8, "maxLength": 80, "pattern": "^[a-zA-Z0-9_-]+$"}, "action": ACTION},
              "required": ["desktop_id", "observation_id", "action_id", "action"], "additionalProperties": False},
          "annotations": {"readOnlyHint": False, "idempotentHint": False}}]
