"""Administrative binary artifact export after the desktop agent is stopped.

Standard-library only; not exposed through Computer MCP. Sources are explicit
task paths under Desktop/Documents/Downloads or Qw3Computer/outputs. Missing
deliverables are task evidence; unreadable/changing files are collection errors.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

CHUNK = 1024 * 1024
MAX_BYTES = 512 * CHUNK


def identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def default_roots():
    if os.name != "nt":
        raise RuntimeError("The production collector must run in the Windows guest")
    profile = Path(os.environ["USERPROFILE"])
    return [profile / p for p in ("Desktop", "Documents", "Downloads")] + [Path("C:/Qw3Computer/outputs")]


def collect(specification, output, *, allowed_roots=None):
    entries = specification.get("files")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 64:
        raise ValueError("Collection needs 1..64 explicit files")
    roots = [Path(p).resolve() for p in (default_roots() if allowed_roots is None else allowed_roots)]
    names = [e.get("name") for e in entries if isinstance(e, dict)]
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"{p}{n}" for p in ("COM", "LPT") for n in range(1, 10)}
    if (len(names) != len(entries) or any(not isinstance(n, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}", n) for n in names)
            or len(set(n.lower() for n in names)) != len(names)
            or any(n.endswith(".") or n.split(".", 1)[0].upper() in reserved for n in names)
            or any(n.lower() in {"receipt.json", "receipt.tmp"} or n.lower().endswith(".part") for n in names)):
        raise ValueError("Unsafe or duplicate artifact export names")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    results = []
    for entry in entries:
        receipt = {"name": entry["name"], "source": entry.get("source"), "status": "read_error"}
        digest = hashlib.sha256()
        copied = 0
        partial = output / (entry["name"] + ".part")
        try:
            if not isinstance(entry.get("source"), str) or not Path(entry["source"]).is_absolute():
                receipt["status"] = "rejected_path"
                raise ValueError("Source must be an absolute task path")
            source = Path(entry["source"])
            resolved = source.resolve()
            if not any(resolved.is_relative_to(root) and resolved != root for root in roots):
                receipt["status"] = "rejected_path"
                raise ValueError("Source is outside the task artifact roots")
            limit = entry.get("max_bytes", MAX_BYTES)
            if type(limit) is not int or not 1 <= limit <= MAX_BYTES:
                receipt["status"] = "invalid_limit"
                raise ValueError("Artifact byte limit is invalid")
            try:
                before = source.stat()
            except FileNotFoundError:
                receipt["status"] = "missing"
                results.append(receipt)
                continue
            if not stat.S_ISREG(before.st_mode):
                receipt["status"] = "rejected_file_type"
                raise ValueError("Artifact is not a regular file")
            if before.st_size > limit:
                receipt["status"] = "size_limit"
                raise ValueError("Artifact exceeds the collection byte limit")
            with source.open("rb") as reader, partial.open("xb") as writer:
                opened = os.fstat(reader.fileno())
                if identity(opened) != identity(before):
                    receipt["status"] = "changed_during_copy"
                    raise ValueError("Artifact changed before opening")
                while block := reader.read(CHUNK):
                    copied += len(block)
                    if copied > limit:
                        receipt["status"] = "size_limit"
                        raise ValueError("Artifact grew beyond the byte limit")
                    writer.write(block)
                    digest.update(block)
                writer.flush()
                os.fsync(writer.fileno())
                if (identity(os.fstat(reader.fileno())) != identity(before)
                        or identity(source.stat()) != identity(before) or copied != before.st_size):
                    receipt["status"] = "changed_during_copy"
                    raise ValueError("Artifact changed during collection")
            partial.rename(output / entry["name"])
            receipt.update(status="copied", bytes=copied, sha256=digest.hexdigest(), source_mtime_ns=before.st_mtime_ns)
        except Exception as exc:
            receipt.update(error=f"{type(exc).__name__}: {exc}", copied_bytes=copied)
            if partial.exists():
                receipt["partial_file"] = partial.name
        results.append(receipt)
    report = {"kind": "native_binary_artifact_collection", "complete": True,
              "infrastructure_ok": all(r["status"] in {"copied", "missing"} for r in results),
              "read_only_sources": True, "elapsed_seconds": time.monotonic() - started,
              "specification_sha256": hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest(),
              "files": results}
    temporary = output / "receipt.tmp"
    with temporary.open("x", encoding="utf-8") as writer:
        json.dump(report, writer, indent=2)
        writer.flush()
        os.fsync(writer.fileno())
    temporary.rename(output / "receipt.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(json.loads(args.specification.read_text(encoding="utf-8-sig")), args.output)
    print(json.dumps({"complete": True, "infrastructure_ok": result["infrastructure_ok"], "files": len(result["files"])}))
    raise SystemExit(0 if result["infrastructure_ok"] else 2)
