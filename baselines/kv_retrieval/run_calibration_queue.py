"""Sequential bounded GPU calibration; stop on any failure for inspection."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    queue = json.loads(a.queue.read_text())
    (a.out / "queue.json").write_bytes(a.queue.read_bytes())
    state = {"status": "waiting_for_prerequisite", "completed": [], "started": time.time()}
    def save():
        tmp = a.out / "state.tmp"
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(a.out / "state.json")
    child = None
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        save()
        prerequisite = Path(queue["prerequisite"])
        while True:
            previous = json.loads(prerequisite.read_text()) if prerequisite.exists() else {}
            if previous.get("status") == queue["prerequisite_success"]:
                break
            if previous.get("status") == "failed":
                raise RuntimeError("prerequisite failed; inspect its raw report")
            time.sleep(5)
        for step in queue["steps"]:
            while True:
                free = float(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits"], text=True).strip()) / 1024
                if free >= step.get("minimum_free_gpu_gib", 0):
                    break
                state.update(status="waiting_for_free_gpu_memory", current=step["name"],
                    free_gpu_gib=free, gpu_owners=subprocess.check_output(["nvidia-smi",
                        "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True))
                save()
                time.sleep(5)
            state.update(status="running", current=step["name"])
            save()
            env = dict(os.environ)
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                env.pop(key, None)
            env.update(step.get("env", {}))
            with (a.out / (step["name"] + ".log")).open("w") as log:
                child = subprocess.Popen(step["command"], cwd=REPO, env=env, stdout=log, stderr=log)
                state["child_pid"] = child.pid
                save()
                code = child.wait()
            if code:
                raise RuntimeError(f"{step['name']} exited {code}")
            state["completed"].append(step["name"])
            save()
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        state["finished"] = time.time()
        save()


if __name__ == "__main__":
    main()
