"""Qualify real Windows drag interruption, without an agent or inference."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import sys
import uuid

from qw3_computer.desktop import Desktop, DesktopError


async def run(args):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", args.run):
        raise ValueError("Invalid run identity")
    args.output.mkdir(parents=True, exist_ok=False)
    share = Path("apps/computer_use/.runtime/windows/shared").resolve()
    prefix = f"drag-interruption-{args.mode}-{args.run}"
    action_id = "drag-fault-" + uuid.uuid4().hex
    result = {"kind": "windows_drag_interruption_contract", "mode": args.mode, "passed": False}
    desktop = Desktop("http://127.0.0.1:18082", (share/"executor-token.txt").read_text().strip(), args.output/"desktop")
    # Never overwrite a previous fault probe or confuse its ready/result files.
    request_file = share/(prefix+"-request.json")
    with request_file.open("x") as handle:
        json.dump({"action_id": action_id}, handle)
    worker_source = Path(__file__).with_name("vm").joinpath("test_drag_interruption.ps1").read_bytes()
    worker_name = prefix+"-worker.ps1"
    with (share/worker_name).open("xb") as handle:
        handle.write(worker_source)
    (args.output/"worker.ps1").write_bytes(worker_source)
    result["worker_sha256"] = hashlib.sha256(worker_source).hexdigest()
    wrapper = args.output/"arm.ps1"
    wrapper.write_text("& '\\\\host.lan\\Data\\" + worker_name + "' -Mode '" + args.mode + "' -Run '" + args.run + "'\n")
    try:
        process = await asyncio.create_subprocess_exec(sys.executable, str(Path(__file__).with_name("vm")/"provision.py"), str(wrapper))
        if await process.wait():
            raise RuntimeError("Fault helper did not launch")
        async with asyncio.timeout(25):
            while not (share/(prefix+"-ready.json")).exists():
                await asyncio.sleep(.2)
        ready = json.loads((share/(prefix+"-ready.json")).read_text(encoding="utf-8-sig"))
        assert ready["action_id"] == action_id
        await desktop.start()
        await desktop.control("resume")
        await desktop.observe()
        async with asyncio.timeout(10):
            # The administrative launcher must have closed before the probe
            # can use the intended test page. This never activates a window.
            while "Office controls qualification" not in desktop.latest["foreground"]["title"]:
                await asyncio.sleep(.3)
                await desktop.observe()
        c = next(c for c in desktop.latest["accessibility"]["controls"] if c["name"] == "Drag surface")
        request = {"desktop_id": desktop.latest["desktop_id"], "observation_id": desktop.latest["observation_id"],
            "action_id": action_id, "action": {"kind": "drag", "from_x": c["center_x"]-150,
            "from_y": c["center_y"], "to_x": c["center_x"]+150, "to_y": c["center_y"],
            "modifier": "ALT", "duration_ms": 2000}}
        try:
            await desktop.act(request)
            raise AssertionError("Interrupted drag unexpectedly completed")
        except DesktopError as exc:
            result["delivery_error"] = str(exc)
        async with asyncio.timeout(40):
            while not (share/(prefix+"-result.json")).exists():
                await asyncio.sleep(.2)
        guest = json.loads((share/(prefix+"-result.json")).read_text(encoding="utf-8-sig"))
        result["guest"] = guest
        assert guest["passed"], guest.get("error", "Guest fault contract failed")
        async with asyncio.timeout(20):
            while True:
                try:
                    health = await desktop.request("GET", "/health")
                    break
                except DesktopError:
                    await asyncio.sleep(.2)
        assert health["paused"]
        receipt = await desktop.request("GET", "/actions/"+action_id)
        assert receipt["status"] in ("dispatching", "uncertain")
        # Bypass the client cache only to prove that the remote durable ledger
        # itself rejects duplicate delivery after a restart or focus loss.
        try:
            await desktop.request("POST", "/act", request)
            raise AssertionError("Executor redelivered uncertain drag")
        except DesktopError as exc:
            assert "uncertain" in str(exc)
            result["duplicate_rejection"] = str(exc)
        assert desktop.unresolved()
        result.update(passed=True, health=health, receipt_status=receipt["status"], action_id=action_id)
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        try:
            if desktop.session:
                await desktop.control("pause")
        finally:
            await desktop.close()
        (args.output/"result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k:v for k,v in result.items() if k != "guest"}))
    return result["passed"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("kill", "focus"), required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if asyncio.run(run(p.parse_args())) else 1)
