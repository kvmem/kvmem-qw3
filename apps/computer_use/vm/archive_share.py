"""Move completed qualification data out of the powered-off VM's SMB share.

The bind-mounted directory itself stays in place. Every moved file is hashed
before and after the rename; no symlink back into the share is created.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

from archive_runtime import inventory


def archive(name):
    if not re.fullmatch(r"shared-archive-[a-zA-Z0-9_-]+", name):
        raise ValueError("Use a task-specific shared-archive name")
    info = json.loads(subprocess.check_output(["docker", "inspect", "qw3-computer-windows"]))[0]
    if info["Config"].get("Labels", {}).get("com.qw3.computer-use.owner") != "codex-goal-20260930":
        raise ValueError("VM does not belong to this task")
    if info["State"]["Running"]:
        raise ValueError("Gracefully shut down the VM before archiving its share")
    share = Path(next(m["Source"] for m in info["Mounts"] if m["Destination"] == "/shared"))
    expected = Path(__file__).resolve().parents[1] / ".runtime/windows/shared"
    if share != expected or share.is_symlink():
        raise ValueError("Unexpected live share")
    keep = {"executor.ps1", "executor-token.txt", "bootstrap.ps1", "fixture.html"}
    if not keep.issubset(p.name for p in share.iterdir() if p.is_file()):
        raise ValueError("Core guest startup files are missing")
    target = share.parent / name
    target.mkdir(mode=0o700, exist_ok=False)
    manifest = target / "archive-manifest.json"
    if (share / manifest.name).exists():
        raise ValueError("Archive manifest filename collision")
    before = inventory(share)
    inode = share.stat().st_ino
    report = {"kind": "guest_share_qualification_archive", "complete": False,
              "source": str(share), "destination": str(target), "moved": [],
              "retained": sorted(keep), "source_inventory": before}
    try:
        for source in sorted(share.iterdir()):
            if source.name not in keep:
                source.rename(target / source.name)
                report["moved"].append(source.name)
        expected_archive = {k: v for k, v in before.items() if k.split("/")[0] not in keep}
        expected_share = {k: v for k, v in before.items() if k.split("/")[0] in keep}
        if inventory(target) != expected_archive or inventory(share) != expected_share:
            raise ValueError("Archived share content changed; preserve both locations for inspection")
        if share.stat().st_ino != inode:
            raise ValueError("Bind-mounted share identity changed")
        report["complete"] = True
        report["bind_inode_preserved"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "source_inventory"}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    archive(parser.parse_args().name)
