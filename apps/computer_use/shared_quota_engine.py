"""Use an explicit common GPU quota for a fresh paired sharing cohort.

The original launcher, executable, model, windows and generation reservation
remain pinned. This adapter only appends the same quota flag to OFF and ON.
It does not make a claim about peak physical memory or inference quality.
"""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys

import run_engine as original


ROOT = Path(__file__).resolve().parents[2]
RATIO_FLAG = ["--kvmem-gpu-memory-ratio", "1.0"]
ORIGINAL_LAUNCHER_SHA = "a912f87e3586962e984d775c887df2cff150bb244d473874c9cc71f659f73264"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile_identity(identity):
    answer = copy.deepcopy(identity)
    if "--kvmem-gpu-memory-ratio" in answer["command"]:
        raise ValueError("Original profile already specifies a quota")
    answer["command"] += RATIO_FLAG
    return answer


def qualify_profile(profile):
    if (profile.get("kind") != "explicit_common_quota_paired_profile" or type(profile.get("quota_ratio")) is not float
            or profile["quota_ratio"] != 1.0 or profile.get("context_tokens") != 524288
            or profile.get("min_free_mib") != 56000 or profile.get("original_windows_changed") is not False
            or profile.get("binary_changed") is not False or profile.get("common_flag_on_both_sides") is not True):
        raise ValueError("Require the exact fresh common-quota profile")
    if digest(ROOT / "apps/computer_use/run_engine.py") != ORIGINAL_LAUNCHER_SHA:
        raise ValueError("Original launcher changed")
    source = Path(profile["original_protocol"])
    if digest(source) != profile["original_protocol_sha256"]:
        raise ValueError("Original comparison changed")
    parent = json.loads(source.read_text())
    for mode in ("off", "on"):
        before = parent["engine_identities"][mode]
        if profile["engine_identities"][mode] != profile_identity(before):
            raise ValueError("Only the explicit common quota may change the profile")
        if digest(before["command"][0]) != before["binary_sha256"]:
            raise ValueError("Frozen model executable changed")
    return profile


def patch_source(source):
    anchor = '    with socket.socket() as listener:\n'
    if source.count(anchor) != 1:
        raise ValueError("Original admission source anchor changed")
    source = source.replace(anchor,
        '    if (args.gpu_sharing != "spare-capacity" or args.min_free_mib != quota_profile["min_free_mib"]\n'
        '            or args.context_tokens != 524288 or args.kvmem_cpu_gib != 48\n'
        '            or args.prefix_reuse != "off" or args.visual_history != "pin-all"\n'
        '            or args.guided_refresh != "off" or args.trace_retrieval or args.trace_memory):\n'
        '        raise ValueError("Keep the exact shared admission, history and diagnostic policy")\n' + anchor, 1)
    anchor = '    env = {k: v for k, v in os.environ.items() if not k.startswith("QW3_")}\n'
    if source.count(anchor) != 1:
        raise ValueError("Original launcher source anchor changed")
    source = source.replace(anchor,
        '    command += RATIO_FLAG\n'
        '    if command != quota_profile["engine_identities"][args.kvmem]["command"]:\n'
        '        raise ValueError("Launcher arguments differ from the exact paired profile")\n'
        + anchor, 1)
    anchor = '    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))\n'
    if source.count(anchor) != 1:
        raise ValueError("Original manifest source anchor changed")
    source = source.replace(anchor,
        '    manifest.update(shared_quota_profile_sha256=quota_profile_sha, '
        'shared_quota_wrapper_sha256=quota_wrapper_sha, common_quota_ratio=1.0, '
        'original_windows_changed=False, binary_changed=False)\n' + anchor, 1)
    return source


def prepare(original_protocol, output, qualification):
    original_protocol, output = Path(original_protocol).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Use a fresh profile file")
    parent = json.loads(original_protocol.read_text())
    profile = {"kind": "explicit_common_quota_paired_profile", "quota_ratio": 1.0,
               "original_protocol": str(original_protocol), "original_protocol_sha256": digest(original_protocol),
               "engine_identities": {mode: profile_identity(parent["engine_identities"][mode]) for mode in ("off", "on")},
               "context_tokens": 524288, "min_free_mib": 56000, "original_windows_changed": False,
               "binary_changed": False, "common_flag_on_both_sides": True,
               "qualification": str(Path(qualification).resolve()),
               "scope": "Fresh OFF/ON cohort only. Keep window131072/gen65536,48GiB CPU and the same executable/model. "
                        "The default pool stays selection+generation blocks; ratio1 affects admission, not those budgets. "
                        "Actual physical startup/inference capacity still needs qualification; preserve the old1/8 separately."}
    qualify_profile(profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2) + "\n")
    return profile


def launch(profile_path, remaining):
    profile_path = Path(profile_path).resolve()
    profile = qualify_profile(json.loads(profile_path.read_text()))
    q = json.loads(Path(profile["qualification"]).read_text())
    if (q.get("kind") != "shared_quota_engine_cpu_qualification" or q.get("passed") is not True
            or q.get("model_calls") != 0 or q.get("device_inputs") != 0
            or q["source_sha256"]["shared_quota_engine.py"] != digest(__file__)
            or q["source_sha256"]["tests/test_shared_quota_engine.py"] != digest(ROOT / "apps/computer_use/tests/test_shared_quota_engine.py")):
        raise ValueError("Require exact completed CPU profile qualification")
    namespace = dict(vars(original))
    namespace.update(RATIO_FLAG=RATIO_FLAG, quota_profile=profile, quota_profile_sha=digest(profile_path),
                     quota_wrapper_sha=digest(__file__))
    sys.argv = [str(ROOT / "apps/computer_use/run_engine.py"), *remaining]
    exec(compile(patch_source(inspect.getsource(original.main)), "qualified-common-quota-launcher", "exec"), namespace)
    namespace["main"]()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--original-protocol", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qualification", type=Path)
    args, remaining = parser.parse_known_args()
    if args.prepare:
        if remaining or any(v is None for v in (args.original_protocol, args.output, args.qualification)):
            parser.error("Provide the exact original protocol, new profile and CPU qualification path")
        prepare(args.original_protocol, args.output, args.qualification)
        print(json.dumps({"profile": str(args.output), "model_calls": 0, "device_inputs": 0}))
    elif args.profile is None:
        parser.error("Provide a qualified common-quota profile")
    else:
        launch(args.profile, remaining)


if __name__ == "__main__":
    main()
