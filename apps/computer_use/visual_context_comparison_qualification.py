"""Reuse the frozen real-Codex fixture to qualify both matched image policies.

Only the fixture's declared candidate kind, cohort loop and policy assertions
change. Its synthetic desktop, scripted model and real MCP operations are reused.
"""
import argparse
import asyncio
import hashlib
import importlib
import inspect
import itertools
import json
from pathlib import Path

import visual_context_qualification as original
from post_sdk51_candidate import replace_once


ORIGINAL_SHA = "6d5f5dcf08d6d279a5ef11d30d71b748d8e4749aeccb68b7fa3c41e11d28be30"


def fixture():
    path = Path(original.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_SHA:
        raise ValueError("Keep the qualified native-Codex fixture unchanged")
    source = inspect.getsource(original.qualify)
    changes = [
        ("'isolated_visual_working_context_candidate'", "'isolated_matched_visual_context_comparison_candidate'"),
        ("for platform in ('windows','linux','android'):",
         "for platform,policy in itertools.product(('windows','linux','android'),('full-retrievable','recent-observations')):"),
        ("runtime=output/platform", "runtime=output/(platform+'-'+policy)"),
        ("image_history='recent-observations',tool_call_limit=12", "image_history=policy,tool_call_limit=12"),
        ("controller.state['image_history']=='recent-observations'", "controller.state['image_history']==policy"),
        ("max(t['report']['dropped_images'] for t in traces)>=3",
         "(max(t['report']['dropped_images'] for t in traces)>=3 if policy=='recent-observations' else all(t['report']['dropped_images']==0 for t in traces))"),
        ("{'platform_tool_catalogue':platform,'passed':True", "{'platform_tool_catalogue':platform,'image_policy':policy,'passed':True"),
        ("'actual_codex_mcp_image_context_transport_qualification'", "'actual_codex_mcp_matched_image_context_transport_qualification'")]
    for before, after in changes:
        source = replace_once(source, before, after)
    namespace = vars(original).copy()
    namespace.update(itertools=itertools, __file__=__file__)
    exec(compile(source, __file__ + ":frozen_fixture", "exec"), namespace)
    return namespace["qualify"], source


async def qualify(candidate, output):
    run, source = fixture()
    result = await run(candidate, output)
    policy_module = importlib.import_module("qw3_computer.visual_context_policy")
    codec = importlib.import_module("qw3_computer.request_archive")
    checks = []
    for cohort in result["platforms"]:
        path = Path(output) / (cohort["platform_tool_catalogue"] + "-" + cohort["image_policy"]) / "app/inference"
        exact = common = True
        for record in cohort["traces"]:
            trace = path / record["trace"]
            meta = json.loads((trace / "request.full.archive.json").read_text())
            body = codec.decode(meta["tree"], trace.parent / "full-history-blobs")
            # Read the last completed pin/unpin receipt from this exact source.
            # Do not infer pins from the image selector's own selected report.
            pins, calls = [], {}
            for message in body["messages"]:
                for call in message.get("tool_calls", []):
                    calls[call["id"]] = call["function"]
                function = calls.get(message.get("tool_call_id"), {})
                if message.get("role") != "tool" or function.get("name") != "mcp__computer__history": continue
                args = json.loads(function["arguments"])
                if args.get("operation") not in ("pin", "unpin"): continue
                receipts = []
                for part in message["content"]:
                    text = part.get("text", "")
                    if part.get("type") == "text" and text.lstrip().startswith("{"):
                        payload = json.loads(text)
                        if payload.get("operation") == args["operation"] and isinstance(payload.get("pins"), list): receipts.append(payload)
                if len(receipts) != 1 or receipts[0].get("input_dispatched") is not False:
                    raise AssertionError("The exact source lacks an unambiguous pin receipt")
                pins = receipts[0]["pins"]
            actual = json.loads((trace / "request.json").read_text())
            full, _ = policy_module.select(body, pins, "full-retrievable")
            recent, _ = policy_module.select(body, pins, "recent-observations")
            common = common and (full["tools"] == recent["tools"] and full["messages"][0] == recent["messages"][0])
            exact = exact and actual == (full if cohort["image_policy"] == "full-retrievable" else recent)
        checks += [{"name":cohort["platform_tool_catalogue"]+"/"+cohort["image_policy"]+" exact source presentation", "passed":exact},
                   {"name":cohort["platform_tool_catalogue"]+"/"+cohort["image_policy"]+" common tools and guidance", "passed":common}]
    result.update(comparison_checks=checks, original_fixture_sha256=ORIGINAL_SHA,
                  derived_fixture_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  qw3_task_latency_or_success_improvement_proven=False,
                  passed=all(c["passed"] for c in checks))
    (Path(output) / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]: raise AssertionError("Actual dispatched inputs do not match the declared image comparison")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = asyncio.run(qualify(args.candidate.resolve(), args.output.resolve()))
    print(json.dumps({"passed":result["passed"], "transport_checks":sum(len(c["checks"]) for c in result["platforms"]),
        "comparison_checks":len(result["comparison_checks"]), "qw3_model_calls":0, "native_device_inputs":0}))


if __name__ == "__main__": main()
