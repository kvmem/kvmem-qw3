"""Create/list/restore disk snapshots of the task-owned, powered-off Windows VM."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "restore", "list"])
    parser.add_argument("--name", default="qw3-clean-20261001")
    parser.add_argument("--container", default="qw3-computer-windows")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.name):
        raise SystemExit("Invalid snapshot name")
    info = json.loads(subprocess.check_output(["docker", "inspect", args.container]))[0]
    if info["Config"].get("Labels", {}).get("com.qw3.computer-use.owner") != "codex-goal-20260930":
        raise SystemExit("Refusing to modify a VM not owned by this task")
    if info["State"]["Running"]:
        raise SystemExit("Stop the application and gracefully shut down this VM before snapshot operations")
    storage = Path(next(m["Source"] for m in info["Mounts"] if m["Destination"] == "/storage"))
    root = Path(__file__).resolve().parent
    metadata_dir = storage.parent / "snapshots" / args.name
    command = ["docker", "run", "--rm", "--network", "none", "--entrypoint", "qemu-img",
               "--mount", f"type=bind,src={storage},dst=/storage", info["Image"]]
    disk = "/storage/data.qcow2"
    disk_info = json.loads(subprocess.check_output(command + ["info", "--output=json", disk]))
    snapshots = {s["name"] for s in disk_info.get("snapshots", [])}
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("executor.ps1", "fixture.html", "bootstrap.ps1")}
    if args.action == "list":
        print(json.dumps(disk_info.get("snapshots", []), indent=2)); return
    if args.action == "create":
        if args.name in snapshots or metadata_dir.exists():
            raise SystemExit("Snapshot already exists; choose a new name")
        metadata_dir.mkdir(parents=True)
        shutil.copyfile(storage / "windows.vars", metadata_dir / "windows.vars")
        metadata = {"name": args.name, "created_at": time.time(), "image_id": info["Image"],
                    "source_hashes": hashes, "disk_virtual_bytes": disk_info["virtual-size"],
                    "state": "powered_off_disk_snapshot", "container": args.container}
        subprocess.run(command + ["snapshot", "-c", args.name, disk], check=True)
        (metadata_dir / "manifest.json").write_text(json.dumps(metadata, indent=2))
    else:
        if args.name not in snapshots:
            raise SystemExit("Snapshot not found")
        metadata = json.loads((metadata_dir / "manifest.json").read_text())
        if metadata["source_hashes"] != hashes:
            raise SystemExit("Executor/fixture/bootstrap differs from the snapshot; freeze a new baseline")
        subprocess.run(command + ["snapshot", "-a", args.name, disk], check=True)
        shutil.copyfile(metadata_dir / "windows.vars", storage / "windows.vars")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
