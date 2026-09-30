"""Probe named wheel directions and bounds on the dedicated Windows form.

Requires an idle app and the standard VM window geometry. The focus and wheel
points below were checked against the visible form; this is a test driver,
not an agent tool or a general window locator.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import uuid

from qw3_computer.desktop import Desktop, DesktopRejected


async def run(args):
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    desktop = Desktop(args.desktop, args.token.read_text().strip(), output / "desktop")
    result = {"kind": "real_windows_named_scroll_contract", "passed": False}
    prefix = "scroll-" + uuid.uuid4().hex[:12]
    try:
        await desktop.start()
        await desktop.control("resume")
        await desktop.observe()

        async def act(label, action):
            return await desktop.act({"desktop_id": desktop.latest["desktop_id"],
                "observation_id": desktop.latest["observation_id"], "action_id": prefix + "-" + label, "action": action})

        await act("open", {"kind": "open_app", "app": "edge"})
        await asyncio.sleep(1)
        await desktop.observe()
        if desktop.latest["foreground"]["process"] != "msedge":
            await act("focus", {"kind": "click", "x": 400, "y": 200, "button": "left", "clicks": 1})
        assert desktop.latest["foreground"]["process"] == "msedge", "Expected visible Edge form"
        await act("top", {"kind": "scroll", "direction": "up", "steps": 8, "x": 950, "y": 400})
        before = await desktop.observe()
        down = (await act("down", {"kind": "scroll", "direction": "down", "steps": 4, "x": 950, "y": 400}))["observation"]
        up = (await act("up", {"kind": "scroll", "direction": "up", "steps": 4, "x": 950, "y": 400}))["observation"]

        def position(observation):
            return next(c["center_y"] for c in observation["accessibility"]["controls"] if c["name"] == "Name" and c["kind"] == "Edit")

        result["name_control_y"] = {"before": position(before), "down": position(down), "up": position(up)}
        assert position(down) < position(before) and position(up) == position(before), "Returned frames did not reflect settled wheel directions"
        result["rejected"] = []
        for i, change in enumerate(({"direction": "sideways"}, {"steps": 0}, {"steps": 9}, {"x": 1280}, {"y": -1})):
            try:
                await act(f"invalid-{i}", {"kind": "scroll", "direction": "down", "steps": 1, "x": 950, "y": 400} | change)
            except DesktopRejected as exc:
                result["rejected"].append(str(exc))
            else:
                raise AssertionError("Invalid scroll was dispatched")
        assert position(await desktop.observe()) == position(before)
        result["executor_sha256"] = hashlib.sha256(Path(__file__).with_name("vm").joinpath("executor.ps1").read_bytes()).hexdigest()
        result["passed"] = True
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        try:
            if desktop.session:
                await desktop.control("pause")
        finally:
            await desktop.close()
        (output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--desktop", default="http://127.0.0.1:18082")
    parser.add_argument("--token", type=Path, default=Path("apps/computer_use/.runtime/windows/shared/executor-token.txt"))
    raise SystemExit(0 if asyncio.run(run(parser.parse_args())) else 1)
