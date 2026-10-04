"""Package private image-history qualification for an already qualified Windows runtime.

This does not replace the installed client, contact a model or provision the VM.
The guest helper verifies the complete declared CPython/Codex file inventory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(candidate, runtime, output, prefix):
    candidate, runtime, output = (Path(p).resolve() for p in (candidate, runtime, output))
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", prefix):
        raise ValueError("Use a unique bounded qualification prefix")
    build_path = candidate / "build.json"
    manifest = json.loads(build_path.read_text())
    if manifest.get("kind") != "isolated_visual_working_context_candidate":
        raise ValueError("Declare the image-history candidate")
    source = candidate / "framework/qw3_computer"
    files = {}
    for name, expected in manifest["candidate_files_sha256"].items():
        if Path(name).name != name or digest(source / name) != expected:
            raise ValueError("Candidate inventory changed: " + name)
        files["candidate/framework/qw3_computer/" + name] = source / name
    archive = json.loads((runtime / "archive.json").read_text())
    qualification = json.loads((runtime / "qualification.json").read_text())
    if not qualification.get("passed") or qualification.get("bundle_sha256") != archive["sha256"]:
        raise ValueError("Require the previously qualified native runtime")
    if digest(Path(archive["path"])) != archive["sha256"]:
        raise ValueError("Qualified native runtime archive changed")
    runtime_files = {}
    for directory in ("python", "codex"):
        for path in sorted((runtime / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                runtime_files[path.relative_to(runtime).as_posix()] = digest(path)
    if not {"python/python.exe", "codex/bin/codex.exe"} <= runtime_files.keys():
        raise ValueError("Require native CPython and Codex executables")
    output.mkdir(parents=True, exist_ok=False)
    files["candidate/build.json"] = build_path
    recipe = Path(__file__).with_name("visual_context_qualification.py")
    files[recipe.name] = recipe
    payload = output / (prefix + ".zip")
    payload_manifest = {"kind": "windows_native_image_context_transport_fixture",
        "candidate_build_sha256": digest(build_path),
        "qualification_recipe_sha256": digest(recipe),
        "payload_files": {name: digest(path) for name, path in files.items()},
        "qualified_runtime_archive_sha256": archive["sha256"],
        "runtime_files": runtime_files,
        "real_qw3_model_calls": 0, "native_device_inputs": 0,
        "business_success_claimed": False}
    with zipfile.ZipFile(payload, "x", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for name, path in files.items():
            zip_file.write(path, name)
        zip_file.writestr("fixture.json", json.dumps(payload_manifest, indent=2))
    sha = digest(payload)
    (output / (prefix + ".sha256")).write_text(sha + "\n")
    script = Path(__file__).parent / "vm/test_visual_context.ps1"
    text = script.read_text().replace("__PREFIX__", prefix).replace("__PAYLOAD_SHA__", sha)
    text = text.replace("__RUNTIME_SHA__", archive["sha256"])
    (output / "run.ps1").write_text(text, encoding="utf-8")
    result = {"kind": "prepared_windows_image_context_transport_qualification",
        "payload_sha256": sha, "payload_bytes": payload.stat().st_size,
        "native_runtime_files": len(runtime_files), "manifest": payload_manifest,
        "packager_sha256": digest(Path(__file__)), "guest_template_sha256": digest(script),
        "guest_script_sha256": digest(output / "run.ps1"), "not_yet_executed": True}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_export(output, prefix):
    """Verify Windows ZIP paths and all exported source/presentation hashes."""
    from osworld_linux.archive import decode
    output = Path(output).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", prefix):
        raise ValueError("Use the declared qualification prefix")
    native = json.loads((output / "qualification.json").read_text())
    payload = output / (prefix + "-evidence.zip")
    if not native.get("passed") or digest(payload) != native["guest_evidence_sha256"]:
        raise ValueError("Require the successful native run's exact export")
    destination = output / "verified-guest-evidence"
    with zipfile.ZipFile(payload) as archive:
        members, paths, total = [], set(), 0
        for member in archive.infolist():
            # PowerShell Compress-Archive emits backslashes in member names.
            relative = Path(member.filename.replace("\\", "/"))
            if (relative.is_absolute() or ".." in relative.parts or any(":" in p for p in relative.parts)
                    or stat.S_ISLNK(member.external_attr >> 16) or relative in paths):
                raise ValueError("Unsafe or duplicate native export path")
            paths.add(relative); total += member.file_size
            if total > 1024**3: raise ValueError("Bound the native fixture export size")
            members.append((member, relative))
        destination.mkdir(mode=0o700, exist_ok=False)
        for member, relative in members:
            path = destination / relative
            if member.is_dir() or member.filename.endswith(("/", "\\")):
                path.mkdir(parents=True, exist_ok=True, mode=0o700)
            else:
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                path.write_bytes(archive.read(member)); path.chmod(0o600)
    traces = list(destination.glob("*/app/inference/*/request.full.archive.json"))
    if len(traces) != 33: raise ValueError("Require all 33 native transport requests")
    for path in traces:
        meta = json.loads(path.read_text())
        source = json.dumps(decode(meta["tree"], path.parent.parent / "full-history-blobs"), ensure_ascii=False).encode()
        report = json.loads((path.parent / "image-context.json").read_text())
        presented = (path.parent / "request.json").read_bytes()
        if (hashlib.sha256(source).hexdigest() != meta["sha256"] or len(source) != meta["bytes"]
                or report["source_sha256"] != meta["sha256"] or report["presented_bytes"] != len(presented)
                or report["presented_sha256"] != hashlib.sha256(presented).hexdigest()):
            raise ValueError("Native full-source or presentation hash differs")
    result = {"kind": "independent_exported_windows_image_context_archive_verification",
        "passed": True, "full_source_archives_verified": len(traces), "actual_request_hashes_verified": len(traces),
        "source_zip_sha256": digest(payload), "verifier_sha256": digest(Path(__file__)),
        "native_inputs": 0, "qw3_model_calls": 0, "business_success_improvement_proven": False}
    (output / "verified-export.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=name == "output")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--verify-export", action="store_true")
    args = parser.parse_args()
    if args.verify_export:
        print(json.dumps(verify_export(args.output, args.prefix)))
        return
    if args.candidate is None or args.runtime is None: parser.error("Declare the private candidate and qualified runtime")
    result = build(args.candidate, args.runtime, args.output, args.prefix)
    print(json.dumps({k: result[k] for k in ("payload_sha256", "payload_bytes", "native_runtime_files")}))


if __name__ == "__main__":
    main()
