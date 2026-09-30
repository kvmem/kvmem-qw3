"""Qualify a real desktop task whose action history crosses the KVMem budget.

Requires a clean guest with vm/long_fixture.html staged at its dedicated path,
an already-running app, and a fresh engine using retrievable visual history.
This is a synthetic archive workflow operated through actual Windows input.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import aiohttp


REFERENCE = "ARCHIVE-T7M4-92QX"
PAGES = 36
PROMPT = (
    "Open Edge using computer open_app edge. Navigate its address bar to "
    "file:///C:/Qw3Computer/long_fixture.html. Complete this local archive review. "
    "Remember the opening archive reference shown on page 1, without transcribing it into an intermediate message "
    "or a document and without using the clipboard. On every page, check I have reviewed this page, then click "
    "Confirm and continue. Continue through all 36 pages. On the final screen, TYPE the opening archive reference "
    "from the earlier screenshot into Opening archive reference, then click Save archive review. "
    "Verify the visible saved status. Do not reload the page, navigate backward, skip pages, use the clipboard, "
    "or open source files. Use only computer tools and do not stop before the final save."
)


def inspect_history(runtime):
    """Locate the model request that first produced the final reference typing."""
    candidates = []
    requests = []
    for directory in (runtime / "inference").iterdir():
        metrics = json.loads((directory / "metrics.json").read_text())
        requests.append((metrics["started_at"], directory))
        calls = {}
        usage = None
        for line in (directory / "backend.jsonl").read_text().splitlines():
            if line == "[DONE]":
                continue
            chunk = json.loads(line)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                for call in choice.get("delta", {}).get("tool_calls", []):
                    index = call.get("index", 0)
                    calls[index] = calls.get(index, "") + call.get("function", {}).get("arguments", "")
        for arguments in calls.values():
            try:
                action = json.loads(arguments).get("action", {})
            except ValueError:
                continue
            if action.get("kind") == "type" and REFERENCE in action.get("text", ""):
                candidates.append((metrics["started_at"], directory, usage))
    assert candidates, "No direct reference typing was found in the model trace"
    answer_started, directory, usage = min(candidates)
    body = json.loads((directory / "request.json").read_text())
    images = []
    text_messages = json.loads(json.dumps(body["messages"]))
    for message in text_messages:
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    images.append(hashlib.sha256(url.encode()).hexdigest())
                    part["image_url"]["url"] = "[image removed for text-only evidence audit]"
    assert REFERENCE not in json.dumps(text_messages), "The reference already appeared as text before the final answer"
    assert usage and usage["prompt_tokens"] > 131072, "The reference query did not cross the retrieval budget"
    historical_images = set()
    for request_started, request_dir in requests:
        if request_started >= answer_started:
            continue
        earlier = json.loads((request_dir / "request.json").read_text())
        for message in earlier["messages"]:
            if isinstance(message.get("content"), list):
                for part in message["content"]:
                    if part.get("type") == "image_url":
                        historical_images.add(hashlib.sha256(part["image_url"]["url"].encode()).hexdigest())
    assert len(historical_images) > 1 and historical_images.issubset(set(images)), "Earlier observation images were discarded"
    return {"request_directory": str(directory), "prompt_tokens": usage["prompt_tokens"],
            "image_count": len(images), "unique_image_url_hashes": len(set(images)),
            "all_prior_images_preserved": True, "unique_prior_images": len(historical_images),
            "reference_absent_from_prior_text": True}


async def run(args):
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    engine = json.loads((args.engine_dir / "manifest.json").read_text())
    assert engine["visual_history"] == "retrievable"
    result = {"kind": "live_windows_synthetic_archive_workflow", "passed": False, "prompt": PROMPT,
              "page_count": PAGES, "engine": engine,
              "fixture_sha256": hashlib.sha256(Path(__file__).with_name("vm").joinpath("long_fixture.html").read_bytes()).hexdigest()}
    staged = json.loads(args.fixture_ready.read_text(encoding="utf-8-sig"))
    assert staged["sha256"] == result["fixture_sha256"], "Guest setup reported a different fixture"
    result["guest_fixture"] = staged
    started = time.monotonic()
    token = args.desktop_token.read_text().strip()
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=45)) as client:
        async def api(route, body=None):
            async with client.request("GET" if body is None else "POST", args.app + route, json=body) as response:
                data = await response.json()
                response.raise_for_status()
                return data

        async def files():
            async with client.get(args.desktop + "/files", headers={"Authorization": "Bearer " + token}) as response:
                data = await response.json()
                response.raise_for_status()
                return data["files"]

        try:
            before = await api("/api/state")
            assert before["status"] == "idle" and await files() == [], "Restore a clean VM and start a fresh app"
            task = await api("/api/task", {"prompt": PROMPT})
            result["task_id"] = task["id"]
            last_status = None
            while time.monotonic() - started < args.timeout:
                state = await api("/api/state")
                if state["status"] != last_status:
                    print(state["status"], flush=True)
                    last_status = state["status"]
                if state["status"] in ("completed", "failed", "paused", "stopped"):
                    result["state"] = state
                    assert state["status"] == "completed", "Archive task did not finish"
                    actual = await files()
                    assert len(actual) == 1 and actual[0]["name"] == "archive-review.json"
                    expected = {"reference": REFERENCE, "reviewed_pages": list(range(1, PAGES + 1)), "page_count": PAGES}
                    assert json.loads(actual[0]["content"]) == expected, "Saved archive record differs"
                    assert actual[0] in state["files"], "App and guest evidence differ"
                    result["verified_file"] = actual[0]
                    with sqlite3.connect((args.app_runtime.resolve() / "desktop/desktop.sqlite").as_uri() + "?mode=ro", uri=True) as db:
                        actions = [json.loads(row[0])["action"] for row in db.execute(
                            "SELECT request FROM actions WHERE task=? AND status='completed' ORDER BY created", (task["id"],))]
                    forbidden = {"CTRL+C", "CTRL+V", "CTRL+R", "F5", "ALT+LEFT"}
                    assert not any(a.get("kind") == "key" and a.get("key") in forbidden for a in actions), "Clipboard or page reset shortcut used"
                    result["completed_actions"] = len(actions)
                    result["history"] = inspect_history(args.app_runtime)
                    if engine["kvmem"] == "on":
                        rows = [json.loads(line) for line in (args.engine_dir / "retrieval.jsonl").read_text().splitlines()]
                        candidates = [r for r in rows if r.get("type") == "meta"
                                      and r["prompt_tokens"] == result["history"]["prompt_tokens"]]
                        assert candidates, "Missing retrieval evidence for the reference-generating request"
                        meta = candidates[-1]
                        assert meta["selected"] < meta["block_count"]
                        assert meta["scorer_fallback_reason"] == "none" and meta["query_ready"] and meta["index_ready"]
                        result["retrieval"] = meta
                    result["passed"] = True
                    break
                await asyncio.sleep(1)
            else:
                await api("/api/control", {"command": "stop"})
                raise TimeoutError("Live archive task deadline exceeded")
        except Exception as exc:
            result["error"] = repr(exc)
        finally:
            if not result["passed"]:
                try:
                    current = await api("/api/state")
                    if current["status"] in ("starting", "running", "verifying"):
                        await api("/api/control", {"command": "stop"})
                except Exception as exc:
                    result["cleanup_error"] = repr(exc)
            result["elapsed_seconds"] = time.monotonic() - started
            (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(json.dumps({k: result.get(k) for k in ("passed", "elapsed_seconds", "error")}), flush=True)
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--app-runtime", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--app", default="http://127.0.0.1:18083")
    parser.add_argument("--desktop", default="http://127.0.0.1:18082")
    parser.add_argument("--desktop-token", type=Path, default=Path("apps/computer_use/.runtime/windows/shared/executor-token.txt"))
    parser.add_argument("--fixture-ready", type=Path, default=Path("apps/computer_use/.runtime/windows/shared/long-task-ready.json"))
    parser.add_argument("--timeout", type=float, default=900)
    raise SystemExit(0 if asyncio.run(run(parser.parse_args())) else 1)
