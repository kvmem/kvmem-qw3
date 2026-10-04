"""Build the opt-in image-history experiment in a private immutable package."""
import argparse
import hashlib
import json
from pathlib import Path

from post_sdk51_candidate import replace_once


SOURCES = {
    "app.py": "0df0b0d94c66d30695d7c226450ffc1b205e580aa5ff056da2a94d038b9063a1",
    "bridge.py": "0361fd940a1a4f8dcd98bc091ac9b6d3d5c96e683ac39cd142ca6bc505917a57",
    "codex.py": "43a7f0c1a6f1abe949f15422bd3819dd88dd5a6a856a5b445f98cf8a09ca3de3",
    "mcp_proxy.py": "936473e44986b16d14d0d69d27cff0009cce0b160c15c0a85d0a5f22dd0abe68",
    "tool_schema.py": "254041c37a4aea67e687104d7a9715669d71b3faff0f4521c337f37696d9a92d",
}

HISTORY_TOOLS = [{"name": "history", "description":
    "Read this task's previously captured selected-device observations without capturing or sending input. "
    "List uses bounded cursor pages. Read returns the original image as historical read-only evidence. "
    "Pin up to four reference observations to retain in image context; unpin frees a reference slot. "
    "None of these operations promotes an archived observation to current or proves current state.",
    "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    "inputSchema": {"type": "object", "properties": {
        "operation": {"type": "string", "enum": ["list", "read", "pin", "unpin"]},
        "observation_id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,128}$"},
        "before_rowid": {"type": "integer", "minimum": 1, "maximum": 2**63-1}},
        "required": ["operation"], "additionalProperties": False,
        "oneOf": [
            {"properties": {"operation": {"const": "list"}}, "not": {"required": ["observation_id"]}},
            {"properties": {"operation": {"enum": ["read", "pin", "unpin"]}},
             "required": ["observation_id"], "not": {"required": ["before_rowid"]}}]}}]


def patch(name, original):
    if hashlib.sha256(original).hexdigest() != SOURCES[name]:
        raise ValueError("Use the declared completion-feedback source: " + name)
    text = original.decode()
    def sub(before, after):
        nonlocal text
        text = replace_once(text, before, after)
    if name == "app.py":
        sub("from .execution import Execution\n", "from .execution import Execution\nfrom .observation_history import ObservationHistory\n")
        sub('workflow=False, device_platform="windows", model_output_tokens=4096):',
            'workflow=False, device_platform="windows", model_output_tokens=4096, image_history="full"):' )
        sub('        self.thinking = thinking\n',
            '        self.thinking = thinking\n        if image_history not in ("full", "recent-observations") or (image_history != "full" and not workflow):\n            raise ValueError("Recent observations require verified workflow mode")\n        self.image_history = image_history\n')
        sub('        self.workflow = Execution(desktop, self.runtime) if workflow else None\n',
            '        self.workflow = Execution(desktop, self.runtime) if workflow else None\n        self.history = ObservationHistory(desktop, self.runtime) if image_history != "full" else None\n')
        sub('    def require_saved_inference_mode(self):\n',
            '    def require_saved_inference_mode(self):\n        if self.state.get("image_history", "full") != self.image_history:\n            raise DesktopError("任务图像历史策略与检查点不同，请用原策略继续或新建任务。")\n')
        sub('                                              max_tokens=self.model_output_tokens,\n',
            '                                              max_tokens=self.model_output_tokens,\n                                              image_context=self.history.reference_pins if self.history else None,\n')
        sub('| ({"mcp__computer__software"} if self.software else set())),',
            '| ({"mcp__computer__software"} if self.software else set())\n                                              | ({"mcp__computer__history"} if self.history else set())),')
        sub('        if self.workflow:\n            proxy_args.append("--workflow")\n',
            '        if self.workflow:\n            proxy_args.append("--workflow")\n        if self.history:\n            proxy_args.append("--history")\n')
        sub('                     context_window=self.model_context_window)',
            '                     context_window=self.model_context_window, history=self.history is not None)')
        sub('            self.state["execution_mode"] = "verified" if self.workflow else "legacy"\n',
            '            self.state["execution_mode"] = "verified" if self.workflow else "legacy"\n            self.state["image_history"] = self.image_history\n')
        sub('and not (name == "workflow" and self.workflow):',
            'and not (name == "workflow" and self.workflow) and not (name == "history" and self.history):')
        sub('        if name == "workflow":\n',
            '        if name == "history":\n            try:\n                result = self.history.invoke(arguments)\n            except (ValueError, OSError) as exc:\n                raise DesktopError(str(exc)) from exc\n            # Archived reads do not touch latest, GUI progress or pending steps.\n            if "observation" in result:\n                return mcp_observation(result["observation"], result)\n            return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}\n        if name == "workflow":\n')
        sub('        if self.workflow:\n            self.workflow.close()\n',
            '        if self.workflow:\n            self.workflow.close()\n        if self.history:\n            self.history.close()\n')
        sub("                            model_output_tokens=getattr(args,'model_output_tokens',4096),\n",
            "                            model_output_tokens=getattr(args,'model_output_tokens',4096),\n                            image_history=getattr(args, 'image_history', 'full'),\n")
        sub('    parser.add_argument("--runtime", type=Path, required=True)\n',
            '    parser.add_argument("--image-history", choices=("full", "recent-observations"), default="full",\n                        help="Opt-in retained recent images with task-owned historical retrieval; requires verified mode")\n    parser.add_argument("--runtime", type=Path, required=True)\n')
    elif name == "bridge.py":
        # Add a source-preserving selector before the existing exact dispatched trace.
        sub('from pathlib import Path\n', 'from pathlib import Path\nfrom .visual_request_archive import present_request\n')
        sub('allowed_tools=None):', 'allowed_tools=None, image_context=None):')
        sub('    app = web.Application(client_max_size=model_request_limit(max_request_bytes))',
            '    if image_context is not None and (not callable(image_context) or trace_dir is None):\n        raise ValueError("Image context requires a task-owned callback and private trace archive")\n    app = web.Application(client_max_size=model_request_limit(max_request_bytes))')
        sub('                trace.mkdir(parents=True)\n',
            '                trace.mkdir(parents=True, mode=0o700)\n                if image_context is not None:\n                    converted.body, _ = present_request(converted.body, image_context(), trace)\n')
    elif name == "codex.py":
        sub('workflow=False, context_window=1048576):', 'workflow=False, context_window=1048576, history=False):')
        sub('    if software:\n        instructions =',
            '    if history:\n        instructions += ("\\nUse history list/read to inspect task-owned archived screenshots, "\n                         "and pin up to four important visual references. Historical images cannot "\n                         "establish current state or be used as current input targets.\\n")\n    if software:\n        instructions =')
        sub('+ (["workflow"] if workflow else []))}', '+ (["workflow"] if workflow else []) + (["history"] if history else []))}')
    elif name == "mcp_proxy.py":
        sub('TOOLS, SOFTWARE_TOOLS, WORKFLOW_TOOLS, tools_for_platform', 'TOOLS, SOFTWARE_TOOLS, WORKFLOW_TOOLS, HISTORY_TOOLS, tools_for_platform')
        sub('    parser.add_argument("--workflow", action="store_true")\n',
            '    parser.add_argument("--workflow", action="store_true")\n    parser.add_argument("--history", action="store_true")\n')
        sub('+ (WORKFLOW_TOOLS if args.workflow else [])}',
            '+ (WORKFLOW_TOOLS if args.workflow else []) + (HISTORY_TOOLS if args.history else [])}')
    else:
        text += '\nHISTORY_TOOLS = ' + repr(HISTORY_TOOLS) + '\n'
    compile(text, 'isolated-image-history-' + name, 'exec')
    return text.encode()


def build_candidate(source_package, source_build, output):
    source_package, source_build, output = map(lambda p: Path(p).resolve(), (source_package, source_build, output))
    manifest = json.loads(source_build.read_text())
    inventory = manifest.get("candidate_files_sha256", {})
    if manifest.get("kind") != "isolated_observation_comparison_feedback_candidate" or any(
            inventory.get(name) != digest for name,digest in SOURCES.items()):
        raise ValueError("Declare the isolated completion-feedback source inventory")
    contents = {}
    for name,digest in inventory.items():
        if Path(name).name != name or Path(name).suffix not in {".py", ".html", ".json", ".tex"}:
            raise ValueError("Invalid package filename")
        data = (source_package/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Private source changed: " + name)
        contents[name] = patch(name,data) if name in SOURCES else data
    helpers = {}
    for name in ("observation_history.py", "visual_working_context.py", "visual_request_archive.py"):
        contents[name] = Path(__file__).with_name(name).read_bytes()
        compile(contents[name], name, "exec")
        helpers[name] = hashlib.sha256(contents[name]).hexdigest()
    archive = Path(__file__).parent/'osworld_linux/archive.py'
    contents['request_archive.py'] = archive.read_bytes()
    helpers['request_archive.py'] = hashlib.sha256(contents['request_archive.py']).hexdigest()
    if output == source_package or output.is_relative_to(source_package):
        raise ValueError("Keep the experiment outside its immutable source")
    output.mkdir(parents=True, exist_ok=False)
    destination = output/'framework/qw3_computer'; destination.mkdir(parents=True)
    for name,data in contents.items(): (destination/name).write_bytes(data)
    result = {"kind": "isolated_visual_working_context_candidate", "qualified_for_production": False,
        "real_model_retest": False, "native_windows_qualified": False, "model_calls": 0, "device_inputs": 0,
        "source_package": str(source_package), "source_files_sha256": inventory,
        "source_build_sha256": hashlib.sha256(source_build.read_bytes()).hexdigest(),
        "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "helper_sha256": helpers,
        "candidate_files_sha256": {name:hashlib.sha256(data).hexdigest() for name,data in contents.items()},
        "changed_package_files": [name for name,data in contents.items() if hashlib.sha256(data).hexdigest()!=inventory.get(name)],
        "default_policy": "full", "experimental_policy": "recent-observations",
        "scope": "Opt-in image presentation plus task-owned archive retrieval. Full converted source preserved losslessly; user/unknown images remain. Latest, receipts, conditions, input, lifetime budget, frozen queues and QW3/KVMem configuration unchanged."}
    (output/'build.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-package','source-build','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    result=build_candidate(args.source_package,args.source_build,args.output)
    print(json.dumps({k:result[k] for k in ('kind','changed_package_files','real_model_retest')}))


if __name__=='__main__':main()
