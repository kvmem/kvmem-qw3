"""Archive and remove an explicit list of owned Windows qualification data.

Run only after the host has verified a full pre-cleanup disk copy and stopped
the relevant applications. The ZIP is closed and every archived file verified
before any source is removed. This is never exposed as a Computer MCP tool.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as reader:
        while chunk := reader.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(path):
    if path.is_symlink() or path.is_junction():
        raise ValueError("Refusing to traverse a source reparse point: " + str(path))
    if path.is_file():
        return [path]
    files = []
    for directory, subdirs, names in os.walk(path, followlinks=False):
        for name in subdirs + names:
            child = Path(directory) / name
            if child.is_symlink() or child.is_junction():
                raise ValueError("Source contains a reparse point: " + str(child))
        files.extend(Path(directory) / name for name in names)
    return sorted(files)


def remove_readonly(function, path, error):
    # Windows marks some owned recent-file directories read-only. Change only
    # that attribute, and only on an already archived cleanup path. Sharing
    # violations and non-permission failures still propagate.
    if not isinstance(error, PermissionError) or os.stat(path).st_mode & stat.S_IWRITE:
        raise error
    os.chmod(path, stat.S_IWRITE)
    function(path)


def clean(plan_path, output):
    if os.name != "nt" or os.environ.get("USERNAME", "").lower() != "qw3test":
        raise ValueError("This cleanup belongs only to the dedicated Qw3Test Windows guest")
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8-sig"))
    run = plan["run_id"]
    if not isinstance(run, str) or not re.fullmatch(r"baseline-cleanup-[0-9]{2}", run):
        raise ValueError("Invalid cleanup identity")
    backup = plan["verified_disk_backup"]
    if not backup.get("complete") or not backup.get("vm_resumed") or len(backup.get("files", [])) != 3:
        raise ValueError("Host disk preservation has not completed")
    output = Path(output)
    if str(output) != "\\\\host.lan\\Data\\" + run + "-export":
        raise ValueError("Unexpected host export location")
    output.mkdir(exist_ok=False)
    profile = Path(os.environ["USERPROFILE"])
    roots = [profile / name for name in ("Desktop", "Documents", "Downloads", "AppData")]
    roots.append(Path(r"C:\Qw3Computer"))
    protected = [Path(r"C:\Qw3Computer\clients"), Path(r"C:\Qw3Computer\installers"),
                 profile / r"AppData\Local\Kingsoft\WPS Office"]
    protected.extend(Path(r"C:\Qw3Computer") / name for name in
                     ("executor.ps1", "bootstrap.ps1", "executor-token.txt", "fixture.html"))
    paths = [Path(item) for item in plan["remove_paths"]]
    for path in paths:
        if (not path.is_absolute() or not any(path.is_relative_to(root) and path != root for root in roots) or
                any(path == keep or path.is_relative_to(keep) or keep.is_relative_to(path) for keep in protected)):
            raise ValueError("Path is outside reviewed data scope or contains installed software: " + str(path))
    if len(set(paths)) != len(paths) or any(a != b and a.is_relative_to(b) for a in paths for b in paths):
        raise ValueError("Overlapping cleanup paths")
    report = {"kind": "windows_qualification_data_cleanup", "complete": False, "removed": [],
              "plan_sha256": sha256(plan_path), "paths": [], "files": []}
    archive_path = output / "preserved-data.zip"
    try:
        with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for index, path in enumerate(paths):
                existed = path.exists()
                row = {"path": str(path), "existed": existed, "directory": path.is_dir(), "file_count": 0}
                report["paths"].append(row)
                if not existed:
                    continue
                files = files_under(path)
                for source in files:
                    before = source.stat()
                    if not stat.S_ISREG(before.st_mode):
                        raise ValueError("Unexpected special source file")
                    name = f"source-{index:03}/" + (str(source.relative_to(path)).replace("\\", "/") if row["directory"] else source.name)
                    digest = hashlib.sha256()
                    with source.open("rb") as reader, archive.open(name, "w", force_zip64=True) as writer:
                        while chunk := reader.read(1024 * 1024):
                            digest.update(chunk); writer.write(chunk)
                    after = source.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError("An application is still changing a cleanup source")
                    report["files"].append({"path": str(source), "archive_member": name,
                                             "bytes": before.st_size, "sha256": digest.hexdigest()})
                row["file_count"] = len(files)
                print(json.dumps({"archived_path": str(path), "files": len(files)}), flush=True)
        with zipfile.ZipFile(archive_path) as archive:
            for row in report["files"]:
                digest = hashlib.sha256()
                with archive.open(row["archive_member"]) as reader:
                    while chunk := reader.read(1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != row["sha256"] or sha256(row["path"]) != row["sha256"]:
                    raise ValueError("Archive or live source failed independent verification")
        for row in report["paths"]:
            if row["existed"] and len(files_under(Path(row["path"]))) != row["file_count"]:
                raise ValueError("New files appeared after archiving")
        report["archive_sha256"] = sha256(archive_path)
        report["archive_verified_before_removal"] = True
        (output / "verified-manifest.json").write_text(json.dumps(report, indent=2))
        for row in report["paths"]:
            if not row["existed"]:
                continue
            path = Path(row["path"])
            if row["directory"]:
                shutil.rmtree(path, onexc=remove_readonly)
            else:
                try:
                    path.unlink()
                except PermissionError as exc:
                    remove_readonly(os.unlink, path, exc)
            report["removed"].append(str(path))
        report["complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clean(args.plan, args.output)
