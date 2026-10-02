"""Run predeclared product repetitions serially, with a fresh owned engine each time."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "apps/computer_use"


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def stop_engine(process, engine_dir):
    if process and process.poll() is None:
        manifest = json.loads((engine_dir / "manifest.json").read_text())
        actual = [p.decode() for p in Path(f"/proc/{process.pid}/cmdline").read_bytes().split(b"\0") if p]
        if manifest["pid"] != process.pid or actual != manifest["command"]:
            raise ValueError("Owned engine identity changed; refusing a signal")
        os.kill(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
        save(engine_dir / "shutdown.json", {"owned_pid": process.pid, "stopped_at": time.time(), "exit_code": process.returncode})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()
    output = args.acceptance.resolve().parent
    path = output / "campaign.json"
    if path.exists():
        raise ValueError("Campaign already has a receipt; inspect its recorded handles before taking further action")
    frozen_bytes = args.acceptance.read_bytes()
    frozen = json.loads(frozen_bytes)
    for relative, expected in frozen["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen campaign source changed: " + relative)
    report = {"kind": "serial_formal_product_campaign", "status": "running", "started_at": time.time(),
              "acceptance_sha256": hashlib.sha256(frozen_bytes).hexdigest(), "planned": len(frozen["trial_ids"]),
              "campaign_pid": os.getpid(), "trials": [], "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(path, report)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    engine = None
    engine_dir = None
    try:
        for identity in frozen["trial_ids"]:
            if args.acceptance.read_bytes() != frozen_bytes:
                raise ValueError("Acceptance definition changed during the campaign")
            engine_dir = APP / ".runtime" / (identity + "-engine")
            trial_dir = APP / ".runtime" / identity
            if engine_dir.exists() or trial_dir.exists():
                raise ValueError("Trial identity already exists; no automatic retry is permitted")
            repair_args = []
            if frozen["task"] == "product-python-repair":
                definition = frozen["repair_fixture"]
                original = ROOT / definition["path"]
                if hashlib.sha256(original.read_bytes()).hexdigest() != definition["sha256"]:
                    raise ValueError("Frozen repair inputs changed")
                fixture_dir = APP / ".runtime" / (identity + "-fixture")
                report.update(stage="resetting_repair_faults", current_trial=identity)
                save(path, report)
                with (output / (identity + "-fixture.log")).open("xb") as log:
                    worker = subprocess.Popen([sys.executable, str(APP / "prepare_python_repair.py"),
                        "--bundle", str(args.bundle.resolve()), "--output", str(fixture_dir), "--reset-from", str(original)],
                        cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(APP)), stdout=log, stderr=log)
                    report["fixture_worker_pid"] = worker.pid; save(path, report)
                    worker.wait()
                if worker.returncode != 0:
                    raise ValueError("Repair input preparation failed; inspect the recorded worker before retry")
                repair_args = ["--repair-fixture", str(fixture_dir / "result.json")]
            if frozen.get("thinking"):
                repair_args.append("--thinking")
            engine = subprocess.Popen([sys.executable, str(APP / "run_engine.py"), "--method", "dflash2", "--kvmem", "off",
                                       "--run-dir", str(engine_dir), "--lock-timeout", "15"], cwd=ROOT)
            report.update(stage="starting_engine", current_trial=identity, engine_pid=engine.pid)
            save(path, report)
            deadline = time.monotonic() + 180
            while True:
                if engine.poll() is not None:
                    raise RuntimeError("Owned engine exited during startup; inspect its log")
                try:
                    with opener.open("http://127.0.0.1:18080/v1/models", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Owned engine readiness deadline expired")
                time.sleep(.5)
            manifest = json.loads((engine_dir / "manifest.json").read_text())
            if manifest["pid"] != engine.pid or manifest["command"] != frozen["engine_command"]:
                raise ValueError("Started engine differs from acceptance profile")
            report["stage"] = "running_trial"
            save(path, report)
            log_path = output / (identity + ".log")
            with log_path.open("xb") as log:
                worker = subprocess.Popen([sys.executable, str(APP / "software_product_run.py"), "--output", str(trial_dir),
                    "--bundle", str(args.bundle.resolve()), "--engine", str(engine_dir), "--ffmpeg", str(args.ffmpeg.resolve()),
                    "--timeout", str(frozen["timeout_seconds"]), "--acceptance-manifest", str(args.acceptance.resolve()), *repair_args],
                    cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(APP)), stdout=log, stderr=log)
                report["worker_pid"] = worker.pid
                save(path, report)
                # An observation timeout is not a reason to start another copy.
                worker.wait()
            result_path = trial_dir / "result.json"
            result = json.loads(result_path.read_text())
            report["trials"].append({"identity": identity, "status": result["status"], "passed": result["passed"],
                "worker_exit_code": worker.returncode, "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()})
            stop_engine(engine, engine_dir)
            engine = None
            if (worker.returncode != 0 or result["status"] != "complete" or not result["passed"]
                    or result.get("guest_result", {}).get("needs_reconciliation")):
                report["status"] = "halted_for_first_error_analysis"
                break
        else:
            report["status"] = "complete"
    except Exception as exc:
        report.update(status="halted", error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            stop_engine(engine, engine_dir)
        except Exception as exc:
            report.update(status="halted", engine_stop_error=str(exc), needs_reconciliation=True)
        report["finished_at"] = time.time()
        save(path, report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    result = main()
    raise SystemExit(0 if result["status"] == "complete" else 1)
