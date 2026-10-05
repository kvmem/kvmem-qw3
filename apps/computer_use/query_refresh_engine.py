"""Qualify OFF/BOTH refresh modes using the unchanged common-quota launcher."""
import argparse
import copy
import inspect
import json
from pathlib import Path
import sys

import shared_quota_engine as common


ROOT = common.ROOT
COMMON_SHA = "25f209b9b6bcba36e099e1e6fc78b47f05eccf989c65022d01da85279601c666"
MODES = ("off", "both")
HELPERS = ("query_refresh_engine.py", "tests/test_query_refresh_engine.py",
           "shared_quota_engine.py", "run_engine.py")


def identity(parent, mode):
    if mode not in MODES:
        raise ValueError("Only frozen off/both refresh modes are admitted")
    result = copy.deepcopy(parent)
    if parent.get("kvmem") != "on" or parent.get("guided_refresh") != "off":
        raise ValueError("Derive both modes from the same original ON profile")
    if "--kvmem-guided-reselect" in result["command"]:
        raise ValueError("Original common profile already overrides query refresh")
    if mode == "both":
        position = result["command"].index("--dflash-model")
        result["command"][position:position] = ["--kvmem-guided-reselect", "both"]
        result["guided_refresh"] = "both"
    return result


def validate(profile):
    if (profile.get("kind") != "declared_same_input_query_refresh_engine_profile"
            or profile.get("modes") != list(MODES) or profile.get("binary_changed") is not False
            or profile.get("windows_changed") is not False):
        raise ValueError("Require the declared independent query refresh profile")
    if common.digest(common.__file__) != COMMON_SHA:
        raise ValueError("Original common-quota adapter changed")
    parent_path = Path(profile["common_profile"])
    if common.digest(parent_path) != profile["common_profile_sha256"]:
        raise ValueError("Frozen common-quota profile changed")
    parent = common.qualify_profile(json.loads(parent_path.read_text()))
    for mode in MODES:
        if profile["engine_identities"][mode] != identity(parent["engine_identities"]["on"], mode):
            raise ValueError("Refresh mode is the only admitted engine difference")
    return parent


def patched_source():
    source = common.patch_source(inspect.getsource(common.original.main))
    before = '            or args.guided_refresh != "off" or args.trace_retrieval or args.trace_memory):\n'
    after = '            or args.guided_refresh != guided_mode or args.kvmem != "on"\n'
    after += '            or args.trace_retrieval or args.trace_memory):\n'
    if source.count(before) != 1:
        raise ValueError("Exact common-quota admission guard changed")
    source = source.replace(before, after, 1)
    before = '    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))\n'
    if source.count(before) != 1:
        raise ValueError("Exact original manifest publication changed")
    return source.replace(before,
        '    manifest.update(query_refresh_profile_sha256=query_profile_sha,\n'
        '                    query_refresh_wrapper_sha256=query_wrapper_sha,\n'
        '                    saved_input_only=True, refresh_mode=guided_mode)\n' + before, 1)


def namespace(profile, mode, profile_sha):
    parent = validate(profile)
    result = dict(vars(common.original))
    result.update(RATIO_FLAG=common.RATIO_FLAG,
                  quota_profile=dict(parent, engine_identities={"on": profile["engine_identities"][mode]}),
                  quota_profile_sha=profile["common_profile_sha256"], quota_wrapper_sha=COMMON_SHA,
                  guided_mode=mode, query_profile_sha=profile_sha,
                  query_wrapper_sha=common.digest(__file__))
    exec(compile(patched_source(), "qualified-saved-input-refresh-launcher", "exec"), result)
    return result


def prepare(common_path, output, qualification):
    common_path, output = Path(common_path).resolve(), Path(output).resolve()
    parent = common.qualify_profile(json.loads(common_path.read_text()))
    profile = dict(kind="declared_same_input_query_refresh_engine_profile", modes=list(MODES),
                   common_profile=str(common_path), common_profile_sha256=common.digest(common_path),
                   qualification=str(Path(qualification).resolve()), binary_changed=False, windows_changed=False,
                   engine_identities={mode: identity(parent["engine_identities"]["on"], mode) for mode in MODES})
    validate(profile)
    with output.open("x") as file:
        json.dump(profile, file, indent=2); file.write("\n")
    return profile


def launch(path, mode, remaining):
    path = Path(path).resolve()
    profile = json.loads(path.read_text())
    validate(profile)
    qualification = json.loads(Path(profile["qualification"]).read_text())
    if (qualification.get("kind") != "query_refresh_engine_cpu_qualification"
            or qualification.get("passed") is not True or qualification.get("skipped") != 0
            or qualification.get("tests_run", 0) < 9
            or set(qualification.get("source_sha256", {})) != set(HELPERS)
            or qualification.get("model_calls") != 0 or qualification.get("device_inputs") != 0):
        raise ValueError("Require the exact completed launcher qualification")
    for name, sha in qualification["source_sha256"].items():
        if common.digest(ROOT / "apps/computer_use" / name) != sha:
            raise ValueError("Qualified launcher source changed")
    runtime = namespace(profile, mode, common.digest(path))
    sys.argv = [str(ROOT / "apps/computer_use/run_engine.py"), *remaining]
    runtime["main"]()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    args, remaining = parser.parse_known_args()
    launch(args.profile, args.mode, remaining)


if __name__ == "__main__":
    main()
