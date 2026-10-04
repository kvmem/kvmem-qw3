"""Stage additive completion feedback against an existing native Windows runtime.

The payload is a synthetic desktop/backend qualification. Installing, changing
or starting a user-facing client and dispatching real model tasks are excluded.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import stat
import zipfile

from osworld_windows.bundle import checked_path, verify_bundle


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(candidate, runtime, output, prefix):
    candidate, runtime, output = (Path(p).resolve() for p in (candidate, runtime, output))
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", prefix):
        raise ValueError("Use a unique bounded qualification prefix")
    candidate_build = candidate / "build.json"
    manifest = json.loads(candidate_build.read_text())
    if (manifest.get("kind") != "isolated_sdk51_literal_completion_feedback_candidate"
            or manifest.get("original_predicates_changed") is not False
            or manifest.get("input_schemas_changed") is not False
            or set(manifest["changed_package_files"]) != {"execution.py", "completion_feedback.py"}):
        raise ValueError("Declare the additive literal feedback candidate")
    files = {"candidate/build.json": candidate_build}
    for name, sha in manifest["candidate_files_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Invalid candidate filename")
        path = candidate / "framework/qw3_computer" / name
        if digest(path) != sha:
            raise ValueError("Candidate source changed")
        files["candidate/framework/qw3_computer/" + name] = path
    verify_bundle(runtime)
    archive = json.loads((runtime / "archive.json").read_text())
    baseline = json.loads((runtime / "manifest.json").read_text())
    runtime_files = {"app/" + name: sha for name, sha in baseline["source_files"].items()}
    for directory in ("python", "codex"):
        for path in sorted((runtime / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                runtime_files[path.relative_to(runtime).as_posix()] = digest(path)
    if not {"python/python.exe", "codex/bin/codex.exe", "app/tests/test_controller.py",
            "app/tests/test_bridge.py", "app/tests/test_protocol.py"} <= runtime_files.keys():
        raise ValueError("Require qualified native CPython/Codex and test fixtures")
    recipe = Path(__file__).with_name("literal_completion_feedback_qualification.py")
    files[recipe.name] = recipe
    output.mkdir(parents=True, exist_ok=False)
    fixture = {"kind": "windows_native_literal_feedback_synthetic_transport_fixture",
               "candidate_build_sha256": digest(candidate_build), "recipe_sha256": digest(recipe),
               "payload_files": {name: digest(path) for name, path in files.items()},
               "qualified_runtime_archive_sha256": archive["sha256"], "runtime_files": runtime_files,
               "qw3_model_calls": 0, "fixture_device_inputs": 0, "business_success_claimed": False,
               "desktop_and_backend_are_synthetic": True}
    payload = output / (prefix + ".zip")
    with zipfile.ZipFile(payload, "x", compression=zipfile.ZIP_DEFLATED) as writer:
        for name, path in files.items():
            checked_path(name); writer.write(path, name)
        writer.writestr("fixture.json", json.dumps(fixture, indent=2))
    payload_sha = digest(payload)
    (output / (prefix + ".sha256")).write_text(payload_sha + "\n")
    template = Path(__file__).parent / "vm/test_literal_feedback.ps1"
    script = template.read_text().replace("__PREFIX__", prefix).replace("__PAYLOAD_SHA__", payload_sha)
    script = script.replace("__RUNTIME_SHA__", archive["sha256"])
    (output / "run.ps1").write_text(script)
    result = {"kind": "prepared_windows_literal_feedback_transport_qualification", "payload_sha256": payload_sha,
              "payload_bytes": payload.stat().st_size, "runtime_files_verified_by_guest": len(runtime_files),
              "fixture": fixture, "packager_sha256": digest(__file__), "guest_template_sha256": digest(template),
              "guest_script_sha256": digest(output / "run.ps1"), "not_yet_executed": True,
              "model_calls": 0, "fixture_device_inputs": 0, "real_device_task_success_claimed": False}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def wire_conditions(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        result = [value] if {"condition", "comparison_evidence", "passed"} <= value.keys() else []
        for part in value.values():
            result.extend(wire_conditions(part))
        return result
    if isinstance(value, list):
        return [evidence for part in value for evidence in wire_conditions(part)]
    return []


def verify_export(output, prefix, host_evidence):
    """Check actual requests, unchanged schemas and durable pending state."""
    output, host_evidence = Path(output).resolve(), Path(host_evidence).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", prefix):
        raise ValueError("Use the declared qualification prefix")
    qualification = json.loads((output / "qualification.json").read_text())
    host = json.loads((host_evidence / "result.json").read_text())
    if (qualification.get("passed") is not True or host.get("passed") is not True
            or qualification["candidate_build_sha256"] != host["candidate_build_sha256"]
            or qualification["portable_recipe_sha256"] != host["recipe_sha256"]):
        raise ValueError("Require successful native and host qualifications of the same source")
    archive_path = output / (prefix + "-evidence.zip")
    if digest(archive_path) != qualification["guest_evidence_sha256"]:
        raise ValueError("Native evidence export changed")
    destination = output / "verified-guest-evidence"
    with zipfile.ZipFile(archive_path) as reader:
        members, paths, total = [], set(), 0
        for member in reader.infolist():
            name = member.filename.replace("\\", "/").rstrip("/")
            relative = checked_path(name)
            if name.casefold() in paths or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Duplicate or symlink native evidence member")
            paths.add(name.casefold()); total += member.file_size
            if total > 256 * 1024**2 or len(paths) > 2000:
                raise ValueError("Native qualification export exceeds declared bounds")
            members.append((member, relative))
        destination.mkdir(mode=0o700, exist_ok=False)
        for member, relative in members:
            path = destination / str(relative)
            if member.is_dir() or member.filename.endswith(("/", "\\")):
                path.mkdir(parents=True, exist_ok=True, mode=0o700)
            else:
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                path.write_bytes(reader.read(member)); path.chmod(0o600)
    schema_requests, wire_requests, pending = 0, 0, 0
    condition = {"kind": "control_text", "text": "Prize Items"}
    for platform in ("windows", "linux", "android"):
        for index in range(1, 4):
            actual = json.loads((destination / platform / f"backend-{index:02}.json").read_text())
            expected = json.loads((host_evidence / platform / f"backend-{index:02}.json").read_text())
            if actual.get("tools") != expected.get("tools"):
                raise ValueError("Native tool schemas differ from the same-source host qualification")
            schema_requests += 1
            if index == 3:
                evidence = [e for message in actual["messages"] if message.get("role") == "tool"
                            for e in wire_conditions(message.get("content"))]
                if not evidence or any(e["condition"] != condition or e["passed"] is not False
                        or e["comparison_evidence"]["input_delivery_inferred"] is not False
                        or e["comparison_evidence"]["independent_business_result_verified"] is not False
                        for e in evidence):
                    raise ValueError("Failed original criterion or evidence scope changed on the native wire")
                wire_requests += 1
        path = destination / platform / "app/execution.sqlite"
        with contextlib.closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            steps = [json.loads(row[0]) for row in db.execute("SELECT data FROM steps")]
            completions = db.execute("SELECT COUNT(*) FROM completions").fetchone()[0]
        if (len(steps) != 1 or steps[0]["status"] != "pending" or steps[0]["conditions"] != [condition]
                or steps[0]["evidence"][0]["passed"] is not False or completions != 0):
            raise ValueError("Native persistent pending step or completion state changed")
        pending += 1
    result = {"kind": "independent_native_literal_feedback_export_verification", "passed": True,
              "source_zip_sha256": digest(archive_path), "native_request_schemas_verified": schema_requests,
              "native_failed_condition_feedback_requests_verified": wire_requests,
              "native_pending_databases_verified": pending, "native_completion_records": 0,
              "qw3_model_calls": 0, "fixture_device_inputs": 0,
              "business_success_improvement_proven": False, "verifier_sha256": digest(__file__)}
    (output / "verified-export.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=name == "output")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--verify-export", action="store_true")
    parser.add_argument("--host-evidence", type=Path)
    args = parser.parse_args()
    if args.verify_export:
        if args.host_evidence is None:
            parser.error("Declare the successful same-source host qualification")
        print(json.dumps(verify_export(args.output, args.prefix, args.host_evidence)))
        return
    if args.candidate is None or args.runtime is None:
        parser.error("Declare the candidate and previously qualified native runtime")
    result = build(args.candidate, args.runtime, args.output, args.prefix)
    print(json.dumps({k: result[k] for k in ("payload_sha256", "payload_bytes", "runtime_files_verified_by_guest")}))


if __name__ == "__main__":
    main()
