"""Actual pinned Codex/MCP feedback transport, synthetic desktop/backend only."""
import asyncio
import base64
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import AsyncMock

from aiohttp import web
from PIL import Image

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--fixture-tests", type=Path, required=True)
parser.add_argument("--build-sha256", required=True)
args = parser.parse_args()
CANDIDATE = args.candidate.resolve()
DIRECTORY = args.output.resolve()
if hashlib.sha256((CANDIDATE / "build.json").read_bytes()).hexdigest() != args.build_sha256:
    raise ValueError("Declared candidate build changed")
BUILD = json.loads((CANDIDATE / "build.json").read_text())
if BUILD.get("kind") != "isolated_sdk51_literal_completion_feedback_candidate":
    raise ValueError("Use the declared literal completion feedback candidate")
fixture_tests = args.fixture_tests.resolve()
sys.path.insert(0, str(fixture_tests.parent))
sys.path.insert(0, str(fixture_tests))
from test_bridge import chunk, serve
from test_controller import FixtureDesktop
DIRECTORY.mkdir(parents=True, exist_ok=False)
PACKAGE = CANDIDATE / "framework/qw3_computer"
for name, sha in BUILD["candidate_files_sha256"].items():
    if hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() != sha:
        raise ValueError("Candidate source changed")
spec = importlib.util.spec_from_file_location("feedback_transport_candidate", PACKAGE / "__init__.py",
                                            submodule_search_locations=[str(PACKAGE)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
app_module = importlib.import_module(spec.name + ".app")


def find_feedback(value):
    result = []
    if isinstance(value, dict):
        if "comparison_evidence" in value and "condition" in value:
            result.append(value)
        for part in value.values(): result.extend(find_feedback(part))
    elif isinstance(value, list):
        for part in value: result.extend(find_feedback(part))
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try: result.extend(find_feedback(json.loads(value)))
        except json.JSONDecodeError: pass
    return result


async def qualify(platform):
    directory = DIRECTORY / platform
    directory.mkdir(exist_ok=False)
    bodies = []
    feedback = []
    async def backend(request):
        body = await request.json(); bodies.append(body)
        (directory / f"backend-{len(bodies):02}.json").write_text(json.dumps(body, ensure_ascii=False))
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await chunk(response, {"reasoning_content": "Synthetic fixture decision; no QW3 inference."})
        if len(bodies) <= 2:
            name = next(t["function"]["name"] for t in body["tools"] if t["function"]["name"].endswith("__workflow"))
            arguments = ({"operation": "begin_step", "step_id": "read-page", "goal": "Inspect document title",
                          "conditions": [{"kind": "control_text", "text": "Prize Items"}]}
                         if len(bodies) == 1 else {"operation": "wait", "timeout_seconds": 0})
            await chunk(response, {"tool_calls": [{"index": 0, "id": f"fixture_{len(bodies)}",
                "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]})
            await chunk(response, finish="tool_calls")
        else:
            for message in body.get("messages", []):
                if message.get("role") == "tool": feedback.extend(find_feedback(message.get("content")))
            await chunk(response, {"content": "Criterion remains unverified; synthetic fixture only."})
            await chunk(response, finish="stop")
        await response.write(b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120}}\n\n')
        await response.write(b"data: [DONE]\n\n")
        return response
    backend_app = web.Application(); backend_app.router.add_post("/v1/chat/completions", backend)
    backend_runner, url = await serve(backend_app)
    desktop = FixtureDesktop()
    desktop.latest.update(progress_sha256="fixture", foreground={"process": "Evince", "title": "fixture.pdf"},
        accessibility={"status": "ok", "truncated": False, "controls": [
            {"name": "page-label-entry", "kind": "text", "value": "2"}]})
    pixels = io.BytesIO(); Image.new("RGB", (128, 96), "blue").save(pixels, format="PNG")
    desktop.observe.return_value = desktop.latest | {"image": base64.b64encode(pixels.getvalue()).decode()}
    desktop.start = AsyncMock(return_value=desktop.latest)
    controller = app_module.Controller(directory / "app", desktop, url + "/v1", 0, workflow=True,
        device_platform=platform, thinking=True, model_output_tokens=16384,
        tool_call_limit=10, trace_backend=True)
    app_runner, app_url = await serve(app_module.web_app(controller))
    controller.port = int(app_url.rsplit(":", 1)[1])
    try:
        await controller.start()
        await controller.start_task("Qualify literal completion feedback transport only; send no desktop input.")
        async with asyncio.timeout(45):
            while controller.state["status"] in ("starting", "running", "verifying"):
                await asyncio.sleep(.02)
        evidence = controller.workflow.steps()[0]["evidence"][0]
        act_schema = next(t["function"]["parameters"] for t in bodies[0]["tools"]
                          if t["function"]["name"].endswith("__act"))
        checks = {
            "three_backend_requests_traversed_real_codex": len(bodies) == 3,
            "two_mcp_calls_consumed_original_cumulative_budget": controller.state["tool_calls_used"] == 2,
            "original_act_schema_is_present": "action" in act_schema.get("properties", {}),
            "thinking_and_output_budget_preserved": all(b.get("enable_thinking") is True and b.get("max_tokens") == 16384 for b in bodies),
            "feedback_reached_actual_next_backend_request": bool(feedback),
            "original_condition_remains_failed_on_wire": bool(feedback) and all(f["passed"] is False for f in feedback),
            "literal_control_scope_reached_backend": bool(feedback) and all(f["comparison_evidence"]["scope"] == "positive_visible_control_name_only" for f in feedback),
            "no_delivery_or_business_inference_on_wire": bool(feedback) and all(not f["comparison_evidence"]["input_delivery_inferred"] and not f["comparison_evidence"]["independent_business_result_verified"] for f in feedback),
            "pending_step_persisted": controller.workflow.steps()[0]["status"] == "pending" and evidence["passed"] is False,
            "no_workflow_completion_record_created": controller.workflow.completion() is None,
            "no_input_endpoint_called": all("/act" not in str(call.args) for call in desktop.request.await_args_list),
            "desktop_paused_after_model_fixture": desktop.paused is True,
        }
        result = {"platform_tool_catalogue": platform, "passed": all(checks.values()), "checks": checks,
                  "feedback_messages_received": len(feedback), "task_status": controller.state["status"],
                  "synthetic_backend_requests": len(bodies), "qw3_model_calls": 0, "native_device_inputs": 0,
                  "native_windows_qualified": sys.platform == "win32", "business_success_improvement_proven": False}
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        await controller.close(); await app_runner.cleanup(); await backend_runner.cleanup()


async def main():
    results = []
    for platform in ("windows", "linux", "android"):
        result = await qualify(platform); results.append(result)
        if not result["passed"]: break
    report = {"kind": "actual_codex_mcp_sdk51_literal_completion_feedback_transport_qualification",
              "at": time.time(), "passed": len(results) == 3 and all(r["passed"] for r in results),
              "candidate_build_sha256": hashlib.sha256((CANDIDATE / "build.json").read_bytes()).hexdigest(),
              "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "platforms": results, "checks_passed": sum(sum(r["checks"].values()) for r in results),
              "qw3_model_calls": 0, "native_device_inputs": 0, "native_windows_qualified": sys.platform == "win32",
              "business_success_improvement_proven": False, "advisor_enabled": False,
              "runtime_platform": sys.platform, "python": sys.version,
              "desktop_and_backend_are_synthetic": True,
              "source_qualification_recipe_sha256": "d14dcdfa905d447e2e3ec7eec0941f1995d3758a7a9255c42bc2e883e349eecd"}
    (DIRECTORY / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "checks_passed", "qw3_model_calls", "native_device_inputs")}))
    if not report["passed"]: raise ValueError("Actual Codex/MCP qualification failed")


asyncio.run(main())
