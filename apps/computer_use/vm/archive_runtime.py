"""Relocate completed, task-owned history to a data disk, retaining old paths.

Administrative only. A caller must first verify that no experiment is using
the selected directories. Source content is hashed before/after copying, the
copy is independently hashed, and only then is the old path replaced by a link.
The live Windows VM storage/share and unrelated experiments are never selected.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat

RUNTIME = Path(__file__).resolve().parents[1] / ".runtime"
COMPLETED_NAMES = re.compile(r"(?:windows-native-client-\d+|windows-long-(?:on|off)-\d+|"
                             r"windows-(?:native-)?suite-\d+|windows-osworld-installers-\d+)")


def inventory(root):
    entries = {}
    for directory, subdirs, files in os.walk(root, followlinks=False):
        for name in sorted(subdirs + files):
            path = Path(directory) / name
            before = path.lstat()
            relative = str(path.relative_to(root))
            if path.is_symlink():
                entries[relative] = {"link": os.readlink(path)}
            elif stat.S_ISDIR(before.st_mode):
                entries[relative] = {"directory": True}
            elif stat.S_ISREG(before.st_mode):
                digest = hashlib.sha256()
                with path.open("rb") as reader:
                    while chunk := reader.read(1024 * 1024):
                        digest.update(chunk)
                after = path.stat()
                if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                        after.st_ino, after.st_size, after.st_mtime_ns):
                    raise RuntimeError("An archived file is still changing: " + relative)
                entries[relative] = {"bytes": before.st_size, "mode": stat.S_IMODE(before.st_mode),
                                     "sha256": digest.hexdigest()}
            else:
                raise ValueError("Unsupported active/special file: " + relative)
    return entries


def relocate(destination, names, evidence):
    destination = Path(destination)
    if not destination.is_absolute() or destination.parent != Path("/data") or not re.fullmatch(
            r"qw3-computer-use-[a-zA-Z0-9_-]+", destination.name):
        raise ValueError("Choose a new, task-specific directory immediately under /data")
    if not names or len(set(names)) != len(names) or any(not COMPLETED_NAMES.fullmatch(n) for n in names):
        raise ValueError("Only explicitly selected completed computer-use history is supported")
    evidence = Path(evidence); evidence.mkdir(parents=True, exist_ok=False)
    destination.mkdir(mode=0o700, exist_ok=False)
    report = {"kind": "verified_runtime_relocation", "complete": False, "directories": [],
              "destination": str(destination), "live_vm_storage_or_share_moved": False}
    try:
        for name in names:
            source = RUNTIME / name
            if not source.is_dir() or source.is_symlink():
                raise ValueError("Source is not an ordinary task history directory: " + name)
            target = destination / name
            before = inventory(source)
            shutil.copytree(source, target, symlinks=True)
            copied = inventory(target)
            if before != copied or before != inventory(source):
                raise ValueError("Relocation verification failed; both copies retained: " + name)
            manifest = evidence / (name + ".json")
            manifest.write_text(json.dumps(before, indent=2))
            retained = RUNTIME / (name + ".relocation-source")
            if retained.exists():
                raise ValueError("A preceding relocation source exists")
            source.rename(retained)
            try:
                source.symlink_to(target, target_is_directory=True)
                if source.resolve() != target or inventory(source) != before:
                    raise ValueError("Relocated original path failed verification")
            except BaseException:
                if source.is_symlink():
                    source.unlink()
                retained.rename(source)
                raise
            shutil.rmtree(retained)  # Exact verified copy remains at the linked destination.
            row = {"name": name, "bytes": sum(e.get("bytes", 0) for e in before.values()),
                   "entries": len(before), "content_manifest": str(manifest),
                   "original_path": str(source), "destination": str(target), "verified": True}
            report["directories"].append(row)
            (evidence / "progress.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({"relocated": name, "bytes": row["bytes"]}), flush=True)
        report["complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (evidence / "result.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    relocate(args.destination, args.names, args.evidence)
