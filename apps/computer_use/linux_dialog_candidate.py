"""Isolated SDK49 observation fix for GTK file chooser window roles.

The existing exact title and unique-active-window rules still apply. Input,
coordinates, observation expiry and task grading are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path


BASELINE_DRIVER_SHA256 = "d62887b059e2529f3bf2adbad761c383450057e4509f0195cc4efec49e510143"


def patch_driver_bytes(original):
    if hashlib.sha256(original).hexdigest() != BASELINE_DRIVER_SHA256:
        raise ValueError("Use the exact qualified SDK49 Linux driver source")
    text = original.decode("utf-8")
    for before, after in (("in ('frame','dialog')", "in ('frame','dialog','file chooser')"),
                          ("in {'frame','dialog'}", "in {'frame','dialog','file-chooser'}")):
        if text.count(before) != 1:
            raise ValueError("Qualified window-role anchor changed")
        text = text.replace(before, after, 1)
    compile(text, "qualified-sdk49-file-chooser-candidate.py", "exec")
    return text.encode("utf-8")


def build_candidate(source_package, qualification_protocol, output):
    source_package = Path(source_package).resolve()
    qualification_protocol = Path(qualification_protocol)
    files = json.loads(qualification_protocol.read_text()).get("framework_sha256")
    if not isinstance(files, dict) or files.get("linux_driver.py") != BASELINE_DRIVER_SHA256:
        raise ValueError("Declare the exact qualified SDK49 package files")
    contents = {}
    for name, expected in files.items():
        if Path(name).name != name or Path(name).suffix not in {".py", ".html", ".json", ".tex"}:
            raise ValueError("Invalid package filename")
        data = (source_package / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Qualified source package changed: " + name)
        contents[name] = data
    contents["linux_driver.py"] = patch_driver_bytes(contents["linux_driver.py"])
    output = Path(output).resolve()
    if output == source_package or output.is_relative_to(source_package):
        raise ValueError("Keep the experiment outside the immutable source package")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "framework" / "qw3_computer"
    destination.mkdir(parents=True)
    candidate_files = {}
    for name, data in contents.items():
        (destination / name).write_bytes(data)
        candidate_files[name] = hashlib.sha256(data).hexdigest()
    manifest = {"kind": "isolated_sdk49_file_chooser_observation_experiment",
        "qualified_for_production": False, "real_model_retest": False,
        "native_windows_qualified": False, "model_calls": 0, "device_inputs": 0,
        "source_package": str(source_package), "source_files_sha256": files,
        "source_protocol_sha256": hashlib.sha256(qualification_protocol.read_bytes()).hexdigest(),
        "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_files_sha256": candidate_files,
        "changed_package_files": [name for name in files if files[name] != candidate_files[name]],
        "scope": "Include the observed GTK file chooser role in exact-title foreground matching; no fallback or input changes."}
    (output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--qualification-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_candidate(args.source_package, args.qualification_protocol, args.output)
    print(json.dumps({key: result[key] for key in ("kind", "changed_package_files", "real_model_retest")}))


if __name__ == "__main__":
    main()
