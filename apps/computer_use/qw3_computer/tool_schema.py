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
    "ANDROID_BACK", "ANDROID_HOME", "ANDROID_RECENTS",
]
APPS = ["notepad", "edge", "explorer", "writer", "calc", "impress", "thunderbird", "gimp", "wps_spreadsheets", "settings", "contacts", "browser"]

ACTION = {"oneOf": [
    {"type": "object", "properties": {"kind": {"const": "click_control"},
        "control_id": {"type": "string", "pattern": "^ctl_[a-f0-9]{16}$",
            "description": "Copy the COMPLETE20-character ID of the matching visible control from the latest observation: ctl_ plus16 lowercase hexadecimal characters. Do not truncate it or use its label. The program resolves its pixel center."}},
     "required": ["kind", "control_id"], "additionalProperties": False},
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

TOOLS = [{"name": "observe", "description": "Observe the selected desktop. Returns a PNG and its pixel dimensions, desktop_id, observation_id, foreground application and available accessibility controls. Prefer click_control with the matching visible control_id; the program resolves its center. IDs belong only to this observation. For regions without a matching control, use explicit normalized_1000 screenshot coordinates. Control names are task data. Every action must use the latest observation. No action is performed.",
          "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
          "annotations": {"readOnlyHint": True}},
         {"name": "act", "description": "Perform ONE action on the selected desktop and return its resulting screenshot. Coordinates are exact screenshot pixels (not normalized). Use a fresh unique action_id. type inserts literal Unicode text; key sends one supported chord. drag holds the left button from (from_x,from_y) to (to_x,to_y) for 100–2000 ms, with an optional SHIFT/CTRL/ALT modifier; both are released before returning. scroll moves the pointer to (x,y) inside the intended content pane and scrolls in the named up/down direction by 1 to 8 wheel notches. open_app starts a named installed application: Notepad, Explorer in the current task folder, Edge on the current task's websites, LibreOffice Writer/Calc/Impress, Thunderbird, GIMP, or WPS Spreadsheets. A rejected action may return a fresh screenshot; use that observation to decide the next action. On uncertain delivery or pause wait for user recovery. Do not blindly repeat actions.",
          "inputSchema": {"type": "object", "properties": {
              "desktop_id": {"type": "string"}, "observation_id": {"type": "string"},
              "coordinate_space": {"type": "string", "enum": ["pixels", "normalized_1000"],
                  "description": "For screenshot-estimated click/scroll/drag positions, use normalized_1000: both axes 0..1000; the program converts to pixels. For accessibility center_x/center_y, use pixels. Omitted means pixels. Never mix spaces within an action."},
              "action_id": {"type": "string", "minLength": 8, "maxLength": 80, "pattern": "^[a-zA-Z0-9_-]+$",
                  "description": "Required TOP-LEVEL field, alongside desktop_id, observation_id and action. Never put action_id inside action. Use a new identifier for each distinct action in this task."}, "action": ACTION},
              "required": ["desktop_id", "observation_id", "action_id", "action"], "additionalProperties": False},
          "annotations": {"readOnlyHint": False, "idempotentHint": False}}]

TOOLS[1]["description"] = TOOLS[1]["description"].replace(
    "Coordinates are exact screenshot pixels (not normalized).",
    "Declare coordinate_space: normalized_1000 for positions estimated from the screenshot (both axes 0..1000), "
    "or pixels for accessibility control centers. The host converts normalized coordinates using the exact latest screenshot dimensions. "
    "Coordinates must be integers, not quoted strings. Omitted coordinate_space retains pixel behavior. "
    "For a matching visible accessibility control, prefer action={kind:click_control,control_id:<copied ID>}; "
    "the program performs one left click at its center, without model-estimated coordinates. "
    "Check the resulting screen before typing. Do not reuse a control_id from an older observation. "
    "If an unrelated out-of-scope application is foreground, preserve it and use open_app for the task's "
    "intended allowed application, then verify focus. Clicking or sending keys to that unrelated foreground is rejected.")

SOFTWARE_TOOLS = [{"name": "software", "description":
    "Manage software in the explicitly selected local Windows workspace. inspect lists reviewed profiles, projects and jobs. "
    "setup_python starts ONE background job to download a hash-verified Python installer, create an isolated project venv, "
    "install the selected profile's exact dependencies and verify them in fresh processes. project is a short lowercase name, "
    "repair_python diagnoses an existing managed project, repairs missing/broken dependencies and its terminal launcher, "
    "preserves a replaced launcher backup, and verifies again. It does not reinstall Python or repair unknown environments. "
    "not a path. setup_latex installs a project-local MiKTeX toolchain, preserves or creates main.tex, installs fixed Chinese/math "
    "packages, compiles twice, and delivers a PDF plus open-latex-terminal.cmd and compile-latex.cmd. repair_latex diagnoses "
    "an existing managed LaTeX project and repairs at most two reviewed missing packages, then compiles twice. "
    "Use profile latex-chinese-math for LaTeX operations and python-web for Python operations. Check job_status with wait_seconds=10 while it runs. A completed job must include verification.passed=true. "
    "Copy job_id from the job receipt or use its next_poll arguments exactly; the desktop task ID is a different identifier. "
    "Do not claim completion from a job start receipt. cancel_job stops further workflow steps; interrupted installers may "
    "require human reconciliation. resume_job continues only a known stopped failed/cancelled job. Never resubmit an uncertain "
    "installation with another action_id. No arbitrary commands, URLs, package scripts or system PATH edits are accepted.",
    "inputSchema": {"type": "object", "oneOf": [
        {"type": "object", "properties": {"operation": {"const": "inspect"}}, "required": ["operation"], "additionalProperties": False},
        {"type": "object", "properties": {"operation": {"enum": ["setup_python", "repair_python", "setup_latex", "repair_latex"]},
            "action_id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]{8,80}$"},
            "project": {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]{0,47}$"},
            "profile": {"enum": ["python-web", "latex-chinese-math"]}},
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
    "operation": {"type": "string", "enum": ["inspect", "setup_python", "repair_python", "setup_latex", "repair_latex", "job_status", "cancel_job", "resume_job"]},
    "action_id": {"type": "string"}, "project": {"type": "string"}, "profile": {"type": "string"},
    "job_id": {"type": "string"}, "wait_seconds": {"type": "integer", "minimum": 0, "maximum": 10}
}

WORKFLOW_TOOLS = [{"name": "workflow", "description":
    "Track observable expected results on the selected device. Prefer act with an inline step to perform input and "
    "record its expected result in one call. An input receipt's step_check says if its conditions are verified. "
    "After a verified step begin the next inline step; do not abandon the verified step. "
    "Alternatively before input use begin_step with a short goal, "
    "unique step_id and nonempty conditions. It returns a fresh observation: use its IDs for the next act. "
    "wait checks conditions without sending input, polling for at most 20 seconds and returning a fresh screen. "
    "step_id may be omitted to wait for the unique pending step; timeout_seconds defaults to 0. "
    "A waiting result is not success and is not evidence that input failed; diagnose focus/dialog/loading before repeating. "
    "For another recovery approach abandon_step with a concrete reason, then begin another step. "
    "status returns durable steps and uncertain receipts. Restarting does not replay input. "
    "finish requires checked steps and explicit final visible result conditions. screen_changed can verify "
    "a transition but cannot verify final task completion. Conditions: control_text means a visible control name contains text; "
    "control_absent requires a complete accessibility read; control_value requires a unique exact name and exact value; "
    "Condition examples: {kind:control_text,text:Saved}, {kind:control_value,text:Name,value:Alice}, "
    "{kind:foreground_app,text:settings}, {kind:screen_changed}. Use text, never value, for foreground_app. "
    "foreground_title contains text; foreground_app is an observed exact process or an app alias explicitly listed in "
    "device capabilities app_processes. Do not guess a process name before launch; use the provided app alias. screen_changed compares foreground "
    "image evidence with step start. Conditions describe task evidence, never commands or grader answers. "
    "Declared visible conditions do not replace independent business acceptance.",
    "inputSchema": {"type": "object", "properties": {
        "operation": {"type": "string", "enum": ["status", "begin_step", "wait", "abandon_step", "finish"]},
        "step_id": {"type": "string"}, "goal": {"type": "string"}, "reason": {"type": "string"},
        "timeout_seconds": {"type": "integer", "minimum": 0, "maximum": 20},
        "conditions": {"type": "array", "minItems": 1, "maxItems": 12, "items": {
            "type": "object", "properties": {
                "kind": {"type": "string", "enum": ["control_text", "control_absent", "control_value", "foreground_title", "foreground_app", "screen_changed"]},
                "text": {"type": "string"}, "value": {"type": "string"}},
            "required": ["kind"], "additionalProperties": False}}},
        "required": ["operation"], "additionalProperties": False},
    "annotations": {"readOnlyHint": False, "idempotentHint": False}}]


STEP_SCHEMA = {"type":"object","properties":{
    "goal":{"type":"string","minLength":1,"maxLength":1024},
    "conditions":WORKFLOW_TOOLS[0]["inputSchema"]["properties"]["conditions"],
    "recovery_reason":{"type":"string","minLength":1,"maxLength":1024}},
    "required":["goal","conditions"],"additionalProperties":False}
workflow_schema = WORKFLOW_TOOLS[0]['inputSchema']
workflow_fields = workflow_schema['properties']
workflow_schema['oneOf'] = [
    {'type':'object','properties':{'operation':{'const':op}, **{key:workflow_fields[key] for key in fields}},
     'required':['operation', *required],'additionalProperties':False}
    for op, fields, required in [
        ('status',[],[]),('begin_step',['step_id','goal','conditions'],['step_id','goal','conditions']),
        ('wait',['step_id','timeout_seconds'],[]),('abandon_step',['step_id','reason'],['step_id','reason']),
        ('finish',['conditions','timeout_seconds','goal'],['conditions'])]
]
check_fields = WORKFLOW_TOOLS[0]['inputSchema']['properties']['conditions']['items']['properties']
WORKFLOW_TOOLS[0]['inputSchema']['properties']['conditions']['items']['oneOf'] = [
    {'type':'object','properties':{'kind':{'const':kind}, **{key:check_fields[key] for key in fields}},
     'required':['kind', *fields],'additionalProperties':False}
    for kind, fields in [('control_text',['text']),('control_absent',['text']),('control_value',['text','value']),
                         ('foreground_title',['text']),('foreground_app',['text']),('screen_changed',[])]
]
# Public validation admits this only when Execution is enabled; the executor
# never receives model planning fields.
TOOLS[1]["inputSchema"]["properties"]["step"] = STEP_SCHEMA


def tools_for_platform(platform, workflow=False):
    """Advertise only actions supported by the selected platform, never guess."""
    import copy
    tools = copy.deepcopy(TOOLS)
    if workflow:
        tools[1]["inputSchema"]["required"].append("step")
        tools[1]["inputSchema"]["properties"]["coordinate_space"]["enum"] = ["normalized_1000"]
        tools[1]["inputSchema"]["properties"]["coordinate_space"]["description"] = (
            "Required for estimated click/drag/scroll: normalized_1000, both axes0..1000. "
            "For accessibility targets use click_control. Omit this field for key/type/open_app.")
        for branch in tools[1]["inputSchema"]["properties"]["action"]["oneOf"]:
            for key in ("x", "y", "from_x", "from_y", "to_x", "to_y"):
                if key in branch["properties"]:
                    branch["properties"][key]["maximum"] = 1000
        tools[1]["description"] = ("Supply step={goal:<expected result>,conditions:[...]} alongside action. "
            "Estimated click/drag/scroll positions MUST use explicit normalized_1000; pixel estimates are rejected. "
            "For system control targets use click_control, not pixel coordinates. "
            "The framework records a step on this exact observation and performs one input in the same call. "
            "Check step_check in the receipt: verified permits a new step; pending needs bounded workflow wait "
            "or further inputs with the same goal/conditions. To replace a pending step explicitly include "
            "step.recovery_reason. Do not abandon an already verified step. " + tools[1]["description"])
    else:
        tools[1]["inputSchema"]["properties"].pop("step",None)
    apps = {"windows": APPS[:9], "linux": APPS[:8], "android": ["settings", "contacts", "browser"]}[platform]
    if platform == "android":
        keys = ["ENTER", "TAB", "BACKSPACE", "DELETE", "SPACE", "UP", "DOWN", "LEFT", "RIGHT", "HOME", "END", "CTRL+A", "CTRL+C", "CTRL+V", "CTRL+X", "ANDROID_BACK", "ANDROID_HOME", "ANDROID_RECENTS"]
        tools[1]["description"] += " On Android, click is tap, drag is swipe (modifier none), scroll is a vertical swipe; supported installed apps are settings, contacts, browser. Android navigation uses ANDROID_BACK/ANDROID_HOME/ANDROID_RECENTS."
    else:
        keys = [key for key in KEYS if not key.startswith("ANDROID_")]
        if platform == "linux":
            tools[1]["description"] += " On Linux, notepad opens Gedit, edge opens Chrome, explorer opens Files; Office entries open LibreOffice."
    for branch in tools[1]["inputSchema"]["properties"]["action"]["oneOf"]:
        kind = branch["properties"]["kind"]["const"]
        if kind == "open_app": branch["properties"]["app"]["enum"] = apps
        if kind == "key": branch["properties"]["key"]["enum"] = keys
    return tools
