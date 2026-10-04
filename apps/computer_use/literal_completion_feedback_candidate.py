"""Add qualified comparison feedback to an isolated SDK51 source copy.

This builder never runs a device, model or grader. It preserves the six SDK51
predicates and all input schemas; only returned condition evidence is enriched.
"""
import argparse
import hashlib
import json
from pathlib import Path

from post_sdk51_candidate import replace_once
from osworld_linux.runner import source_identity


EXECUTION_SHA256 = "de584fd72cd68f651baf607932737cc9048603e9c641bb7ee6cdb9c14377adcf"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def patch_execution(original):
    if digest(original) != EXECUTION_SHA256:
        raise ValueError("Use the declared SDK51 condition execution source")
    text = replace_once(original.decode(),
        "from .desktop import DesktopError, DesktopValidationError\n",
        "from .desktop import DesktopError, DesktopValidationError\n"
        "from .completion_feedback import explain_condition\n")
    text = replace_once(text, "        evidence.append(item)\n",
                        "        evidence.append(explain_condition(item, observation))\n")
    compile(text, "isolated-sdk51-literal-condition-feedback.py", "exec")
    return text.encode()


def build_candidate(source_framework, source_manifest, source_acceptance, output,
                    manifest_sha256, acceptance_sha256):
    source_framework, source_manifest, source_acceptance, output = map(
        lambda p: Path(p).resolve(), (source_framework, source_manifest, source_acceptance, output))
    manifest_bytes = source_manifest.read_bytes()
    if (not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64
            or digest(manifest_bytes) != manifest_sha256):
        raise ValueError("Declared immutable SDK51 source manifest changed")
    manifest = json.loads(manifest_bytes)
    files = manifest.get("files")
    if (manifest.get("kind") != "isolated-qualified-execution-fix-candidate"
            or not isinstance(files, dict) or not files
            or files.get("execution.py") != EXECUTION_SHA256):
        raise ValueError("Require the declared SDK51 package inventory")
    package = source_framework / "qw3_computer"
    if output == source_framework or output.is_relative_to(source_framework):
        raise ValueError("Keep the candidate outside immutable source directories")
    acceptance_bytes = source_acceptance.read_bytes()
    if digest(acceptance_bytes) != acceptance_sha256:
        raise ValueError("Declared SDK51 source admission changed")
    acceptance = json.loads(acceptance_bytes)
    if acceptance.get("framework_files") != source_identity(source_framework):
        raise ValueError("Source framework or frozen OSWorld helper inventory changed")
    auxiliaries = acceptance.get("framework_auxiliary_files")
    if not isinstance(auxiliaries, dict) or "device_gateway_entry.py" in auxiliaries:
        raise ValueError("Require independent canonical entry and auxiliary inventories")
    auxiliary_files = {"device_gateway_entry.py": acceptance.get("canonical_entry_sha256"), **auxiliaries}
    auxiliary_contents = {}
    for name, expected in auxiliary_files.items():
        if not isinstance(name, str) or Path(name).name != name or Path(name).suffix != ".py":
            raise ValueError("Invalid auxiliary filename")
        data = (source_framework / name).read_bytes()
        if digest(data) != expected:
            raise ValueError("Immutable auxiliary changed: " + name)
        auxiliary_contents[name] = data
    contents = {}
    for name, expected in files.items():
        if (not isinstance(name, str) or Path(name).name != name
                or Path(name).suffix not in {".py", ".html", ".json", ".tex"}):
            raise ValueError("Invalid package inventory filename")
        data = (package / name).read_bytes()
        if digest(data) != expected:
            raise ValueError("Immutable SDK51 package changed: " + name)
        contents[name] = data
    helper = Path(__file__).with_name("completion_feedback.py")
    helper_bytes = helper.read_bytes()
    compile(helper_bytes, str(helper), "exec")
    contents["execution.py"] = patch_execution(contents["execution.py"])
    contents["completion_feedback.py"] = helper_bytes
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "framework/qw3_computer"
    destination.mkdir(parents=True)
    for name, data in contents.items():
        (destination / name).write_bytes(data)
    for name, data in auxiliary_contents.items():
        (destination.parent / name).write_bytes(data)
    result = {
        "kind": "isolated_sdk51_literal_completion_feedback_candidate",
        "source_framework": str(source_framework),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": manifest_sha256,
        "source_acceptance": str(source_acceptance),
        "source_acceptance_sha256": acceptance_sha256,
        "source_osworld_helper_files_sha256": {k: v for k, v in acceptance["framework_files"].items()
                                              if not k.startswith("qw3_computer/")},
        "source_files_sha256": files,
        "source_candidate_changes": manifest.get("candidate_changes"),
        "candidate_files_sha256": {n: digest(v) for n, v in contents.items()},
        "unchanged_auxiliary_files_sha256": {n: digest(v) for n, v in auxiliary_contents.items()},
        "recipe_sha256": digest(Path(__file__).read_bytes()),
        "helper_sha256": digest(helper_bytes),
        "changed_package_files": [n for n, v in contents.items() if digest(v) != files.get(n)],
        "input_schemas_changed": False,
        "original_predicates_changed": False,
        "native_field_identity_added": False,
        "recording_policy_changed": False,
        "qualified_for_production": False,
        "native_windows_qualified": False,
        "real_model_retest": False,
        "model_calls": 0,
        "device_inputs": 0,
        "scope": "Only additive literal condition feedback. Preserve exact matching, original hints, "
                 "pending criteria, receipts, repeat guards, task budgets and source archive. "
                 "Missing native text is not inferred input failure; UI evidence is not business proof.",
    }
    (output / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-framework", "source-manifest", "source-acceptance", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--acceptance-sha256", required=True)
    args = parser.parse_args()
    result = build_candidate(args.source_framework, args.source_manifest, args.source_acceptance, args.output,
                             args.manifest_sha256, args.acceptance_sha256)
    print(json.dumps({k: result[k] for k in ("kind", "changed_package_files", "real_model_retest")}))


if __name__ == "__main__":
    main()
