"""Live bounded-drag and office-key contracts on a local synthetic Windows page."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import uuid

from qw3_computer.desktop import Desktop, DesktopRejected


async def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    desktop = Desktop(args.desktop, args.token.read_text().strip(), output / "desktop")
    result = {"kind": "windows_office_input_contracts", "passed": False, "run": args.run}
    prefix = "office-controls-" + uuid.uuid4().hex[:12]
    try:
        await desktop.start()
        await desktop.control("resume")
        await desktop.observe()
        assert "Office controls qualification" in desktop.latest["foreground"]["title"]

        async def act(label, action):
            return await desktop.act({"desktop_id": desktop.latest["desktop_id"],
                "observation_id": desktop.latest["observation_id"], "action_id": prefix + "-" + label,
                "action": action})

        def point(name):
            control = next(c for c in desktop.latest["accessibility"]["controls"] if c["name"] == name)
            return {"x": control["center_x"], "y": control["center_y"]}

        target = point("Drag surface")
        action = {"kind": "drag", "from_x": target["x"]-150, "from_y": target["y"],
                  "to_x": target["x"]+150, "to_y": target["y"], "duration_ms": 400, "modifier": "none"}
        result["rejected"] = []
        for i, change in enumerate(({"from_x": -1}, {"to_y": desktop.latest["height"]},
                                    {"duration_ms": 99}, {"duration_ms": 2001},
                                    {"from_x": True}, {"modifier": "WIN"}, {"extra": 1})):
            try:
                await act(f"invalid-drag-{i}", action | change)
            except DesktopRejected as exc:
                result["rejected"].append(str(exc))
            else:
                raise AssertionError("Invalid drag was dispatched")
        for label, invalid in (("app", {"kind": "open_app", "app": "powershell"}),
                               ("key", {"kind": "key", "key": "WIN+R"})):
            try:
                await act("invalid-"+label, invalid)
            except DesktopRejected as exc:
                result["rejected"].append(str(exc))
            else:
                raise AssertionError("Out-of-scope action was dispatched")
        for modifier in ("none", "SHIFT", "CTRL", "ALT"):
            await act("drag-"+modifier, action | {"modifier": modifier})
            # The browser records this real key event, proving the drag's
            # modifier did not remain pressed after the receipt returned.
            await act("released-"+modifier, {"kind": "key", "key": "M"})
        await act("focus-text", {"kind": "click", **point("Keyboard check"), "button": "left", "clicks": 1})
        for i, key in enumerate(("CTRL+HOME", "SHIFT+END", "CTRL+C", "END", "SPACE", "CTRL+V")):
            await act(f"edit-{i}", {"kind": "key", "key": key})
        await act("export", {"kind": "click", **point("Save control evidence"), "button": "left", "clicks": 1})
        evidence = None
        for _ in range(20):
            files = (await desktop.request("GET", "/files"))["files"]
            matches = [f for f in files if f["name"] == f"office-controls-{args.run}.json"]
            if matches:
                evidence = json.loads(matches[0]["content"])
                break
            await asyncio.sleep(0.25)
        assert evidence is not None, "GUI evidence download missing"
        (output / "browser-events.json").write_text(json.dumps(evidence, indent=2))
        downs = [e for e in evidence["events"] if e["kind"] == "pointerdown" and e["target"] == "surface"]
        ups = [e for e in evidence["events"] if e["kind"] == "pointerup" and e["target"] == "surface"]
        releases = [e for e in evidence["events"] if e["kind"] == "keydown" and e["key"].lower() == "m"]
        assert len(downs) == len(ups) == len(releases) == 4, "Unexpected or duplicate drag delivery"
        for i, modifier in enumerate((None, "shift", "ctrl", "alt")):
            assert ups[i]["x"] - downs[i]["x"] == 300
            assert ups[i]["buttons"] == 0
            assert 300 <= ups[i]["time"] - downs[i]["time"] <= 1500
            assert [bool(downs[i][k]) for k in ("shift", "ctrl", "alt")] == [modifier == k for k in ("shift", "ctrl", "alt")]
            assert not any(releases[i][k] for k in ("shift", "ctrl", "alt")), "Drag modifier remained held"
        assert evidence["text"] == "alpha beta gamma alpha beta gamma", "Office selection/clipboard keys failed"
        result.update(passed=True, verified_drags=4, verified_rejections=len(result["rejected"]),
                      keyboard_result=evidence["text"])
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        try:
            if desktop.session:
                await desktop.control("pause")
        finally:
            await desktop.close()
        result["executor_sha256"] = hashlib.sha256(Path(__file__).with_name("vm").joinpath("executor.ps1").read_bytes()).hexdigest()
        (output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", default="01")
    parser.add_argument("--desktop", default="http://127.0.0.1:18082")
    parser.add_argument("--token", type=Path, default=Path("apps/computer_use/.runtime/windows/shared/executor-token.txt"))
    raise SystemExit(0 if asyncio.run(run(parser.parse_args())) else 1)
