"""Run a frozen Windows task through the application and verify its saved file.

This is a smoke runner. Repeated benchmark acceptance additionally requires
restoring the same VM snapshot before every trial.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time

import aiohttp


CASES = {
    "notepad": {
        "prompt": "Open Notepad and create a NEW document. Write exactly these two lines, with a final newline:\n"
                  "QW3 computer use\nLocal Windows test passed.\n"
                  "Save it as C:\\Qw3Computer\\outputs\\codex-note.txt. Verify the saved filename and visible text. "
                  "Do not modify any existing document. Use only the computer tools.",
        "filename": "codex-note.txt", "content": "QW3 computer use\r\nLocal Windows test passed.\r\n",
    },
    "browser": {
        "prompt": "Open the Edge browser fixture using computer open_app edge. Fill the Request form with Name Alice, "
                  "Reference QC-001, Priority High, and check I confirm the details. Click Save request to download "
                  "browser-result.json. Verify the visible Saved status. Do not edit any other files. Use only the computer tools.",
        "filename": "browser-result.json", "json": {"name": "Alice", "reference": "QC-001", "priority": "High", "confirmed": True},
    },
    "cross_app": {
        "prompt": "Open the Edge browser fixture using computer open_app edge. Find the reference code in its Copy task "
                  "section. Select and COPY the code using Ctrl+C. Open Notepad, create a NEW document, and PASTE it using "
                  "Ctrl+V. Do not type the code yourself. Save it as C:\\Qw3Computer\\outputs\\copied-reference.txt. "
                  "Verify the saved filename and visible reference code. Do not change existing files. Use only the computer tools.",
        "filename": "copied-reference.txt", "content": "QW3-WINDOWS-2026-ALPHA",
    },
}


async def run(args):
    args.run_dir.mkdir(parents=True, exist_ok=False)
    case = CASES[args.case]
    result = {"case": args.case, "definition": case, "passed": False, "kind": "real_model_windows_smoke"}
    started = time.monotonic()
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=45)) as client:
        async def api(route, body=None):
            async with client.request("POST" if body is not None else "GET", args.app + route, json=body) as response:
                data = await response.json()
                if response.status != 200:
                    raise RuntimeError(data)
                return data
        try:
            before = await api("/api/state")
            if any(f["name"] == case["filename"] for f in before.get("files", [])):
                raise RuntimeError("The expected output already exists; restore a clean VM state before this trial")
            state = await api("/api/task", {"prompt": case["prompt"]})
            result["task_id"] = state["id"]
            paused = False
            last_status = None
            while time.monotonic() - started < args.timeout:
                state = await api("/api/state")
                if state["status"] != last_status:
                    print(state["status"], flush=True)
                    last_status = state["status"]
                if args.pause_after_seconds and not paused and time.monotonic() - started >= args.pause_after_seconds and state["status"] == "running":
                    pause_started = time.monotonic()
                    stopped = await api("/api/control", {"command": "pause"})
                    result["pause_latency_seconds"] = time.monotonic() - pause_started
                    assert stopped["status"] == "paused" and stopped["desktop_paused"]
                    await asyncio.sleep(2)
                    still = await api("/api/state")
                    assert still["status"] == "paused" and still["desktop_paused"]
                    result["paused_state"] = still
                    await api("/api/control", {"command": "resume"})
                    paused = True
                elif state["status"] in ("completed", "failed", "stopped", "paused"):
                    result["state"] = state
                    if state["status"] != "completed":
                        raise RuntimeError("Task did not complete: " + state["status"])
                    files = [f for f in state["files"] if f["name"] == case["filename"]]
                    assert len(files) == 1, "Expected output file is missing"
                    if "json" in case:
                        assert json.loads(files[0]["content"]) == case["json"], "Saved JSON differs"
                    else:
                        assert files[0]["content"] == case["content"], "Saved text differs"
                    result.update(passed=True, verified_file=files[0], pause_resume_exercised=paused)
                    break
                await asyncio.sleep(0.5)
            else:
                await api("/api/control", {"command": "stop"})
                raise TimeoutError("Task deadline exceeded")
        except Exception as exc:
            result["error"] = repr(exc)
            raise
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            (args.run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(json.dumps({k: result.get(k) for k in ("passed", "elapsed_seconds", "error")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", default="http://127.0.0.1:18083")
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--pause-after-seconds", type=float, default=0)
    asyncio.run(run(parser.parse_args()))
