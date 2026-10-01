"""Verify literal desktop text input by saving actual Notepad files in the test VM.

No model, UIA writing, or programmatic document edits are used.
The executor inserts literal text into the focused edit control; the probe saves
through the ordinary UI and independently verifies the resulting files.
The focus point is for the reviewed default VM window geometry; Save As uses
visible accessibility control centers. Run only while other VM tasks are idle.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import uuid

from qw3_computer.desktop import Desktop


CASES = {
    "english": "QW3 computer use\nLocal Windows test passed.\n",
    "unicode": "中文输入检查：你好，世界。🙂\nCafé naïve — 日本語 — 한국어\r\n终行。\n",
    "unicode_boundary": "中文🙂abc\n" * 512,
    "boundary": ("0123456789 abcDEF !?\n" * 216)[:4096],
}
# The executor's bound is 4096 UTF-16 code units. Use an exact boundary payload
# without splitting a surrogate pair; the shorter case separately tests emoji.


async def run(args):
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    desktop = Desktop(args.desktop, args.token.read_text().strip(), output / "desktop")
    result = {"kind": "real_windows_literal_typing_contract", "passed": False, "trials": [],
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "executor_sha256": hashlib.sha256(Path(__file__).with_name("vm").joinpath("executor.ps1").read_bytes()).hexdigest()}
    prefix = "type-" + uuid.uuid4().hex[:8]
    action_number = 0
    try:
        await desktop.start()
        await desktop.control("resume")
        await desktop.observe()

        async def act(action):
            nonlocal action_number
            action_number += 1
            return await desktop.act({"desktop_id": desktop.latest["desktop_id"],
                "observation_id": desktop.latest["observation_id"], "action_id": f"{prefix}-{action_number}", "action": action})

        async def key(name):
            return await act({"kind": "key", "key": name})

        async def click(x, y):
            return await act({"kind": "click", "x": x, "y": y, "button": "left", "clicks": 1})

        async def control(name, kind):
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                observation = await desktop.observe()
                matches = [c for c in observation.get("accessibility", {}).get("controls", [])
                           if c["name"] == name and c["kind"] == kind]
                if matches:
                    return matches[0]
                await asyncio.sleep(0.2)
            raise TimeoutError(f"Visible control {name!r}/{kind} unavailable")

        for repetition in range(args.repetitions):
            for case in args.cases:
                text = CASES[case]
                filename = f"{prefix}-{case}-{repetition}.txt"
                trial = {"case": case, "repetition": repetition, "filename": filename, "passed": False}
                result["trials"].append(trial)
                await act({"kind": "open_app", "app": "notepad"})
                await asyncio.sleep(0.5)
                await desktop.observe()
                if desktop.latest["foreground"]["process"].lower() == "msedge":
                    # Windows can keep the previous test's Edge window in
                    # front when Notepad opens behind it. Switch normally.
                    await key("ALT+TAB")
                await click(500, 400)
                assert desktop.latest["foreground"]["process"].lower() == "notepad", "Notepad did not gain focus"
                await key("CTRL+N")
                began = time.monotonic()
                receipt = await act({"kind": "type", "text": text})
                trial["type_seconds"] = time.monotonic() - began
                trial["type_observation_sha256"] = receipt["observation"]["sha256"]
                # A later observation distinguishes transient paint from an
                # incorrect document; file verification is authoritative.
                await asyncio.sleep(1)
                later = await desktop.observe()
                trial["settled_observation_sha256"] = later["sha256"]
                await key("CTRL+S")
                field = await control("File name:", "Edit")
                await click(field["center_x"], field["center_y"])
                await key("CTRL+A")
                await act({"kind": "type", "text": "C:\\Qw3Computer\\outputs\\" + filename})
                button = await control("Save", "Button")
                await click(button["center_x"], button["center_y"])
                await asyncio.sleep(0.3)
                files = (await desktop.request("GET", "/files"))["files"]
                actual = next(f for f in files if f["name"] == filename)
                expected = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
                raw = expected.encode()
                trial.update(expected=expected, actual=actual)
                trial["passed"] = (actual["content"] == expected and actual["size"] == len(raw)
                                   and actual["sha256"] == hashlib.sha256(raw).hexdigest())
                print(case, repetition, trial["passed"], round(trial["type_seconds"], 3), flush=True)
                if not trial["passed"] and args.stop_on_failure:
                    raise AssertionError("Saved text differs from the requested literal input")
                await desktop.observe()
                await key("CTRL+W")
        if args.browser:
            trial = {"case": "browser_unicode_and_boundary", "passed": False}
            result["trials"].append(trial)
            existing = {f["name"] for f in (await desktop.request("GET", "/files"))["files"]}
            await act({"kind": "open_app", "app": "edge"})
            await asyncio.sleep(1)
            name = "中文🙂 Café — 한국어 {[]}%+^"
            reference = ("Az09{}[]()%+^~!/\\;" * 250)[:4096]
            assert len(reference.encode("utf-16-le")) == 8192
            for label, value in (("Name", name), ("Reference", reference)):
                field = await control(label, "Edit")
                await click(field["center_x"], field["center_y"])
                await key("CTRL+A")
                await act({"kind": "type", "text": value})
            checkbox = await control("I have checked the information", "CheckBox")
            await click(checkbox["center_x"], checkbox["center_y"])
            button = await control("Save request", "Button")
            await click(button["center_x"], button["center_y"])
            for attempt in range(40):
                files = (await desktop.request("GET", "/files"))["files"]
                new = [f for f in files if f["name"] not in existing and f["name"].startswith("browser-result")]
                if new:
                    break
                await asyncio.sleep(0.2)
            if len(new) != 1:
                raise AssertionError("Expected one newly downloaded browser fixture file")
            actual = new[0]
            expected = {"name": name, "reference": reference, "priority": "Normal", "confirmed": True}
            trial.update(actual=actual, expected=expected)
            trial["passed"] = (json.loads(actual["content"]) == expected
                               and hashlib.sha256(actual["content"].encode()).hexdigest() == actual["sha256"])
            print(trial["case"], trial["passed"], flush=True)
        result["passed"] = all(t["passed"] for t in result["trials"])
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        try:
            if desktop.session:
                await desktop.control("pause")
        finally:
            await desktop.close()
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({k: result.get(k) for k in ("passed", "error")}), flush=True)
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--desktop", default="http://127.0.0.1:18082")
    parser.add_argument("--token", type=Path, default=Path("apps/computer_use/.runtime/windows/shared/executor-token.txt"))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--browser", action="store_true", help="Also verify Unicode and 4096-character Edge form input")
    raise SystemExit(0 if asyncio.run(run(parser.parse_args())) else 1)
