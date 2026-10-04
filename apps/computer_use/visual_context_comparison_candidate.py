"""Add a matched full-image comparator without editing the frozen image candidate."""
import argparse
import hashlib
import json
from pathlib import Path

from post_sdk51_candidate import replace_once


SOURCES = {
    "app.py": "ad3d49594c83cd69216cea36d1c43395d03e4a82560eaac807edee8c38e0c588",
    "bridge.py": "29f534181d4d336a09229a919bf528b0d52b01a9f4ddb247e2158f8cb624efb3",
    "visual_request_archive.py": "934aeceaccbe0d79264b760443b01a83f04f3a12e69eea2d559e6ed2093b0fdd"}


def patch(name, data):
    if hashlib.sha256(data).hexdigest() != SOURCES[name]:
        raise ValueError("Declare the qualified image candidate: " + name)
    text = data.decode()
    def sub(before, after):
        nonlocal text
        text = replace_once(text, before, after)
    if name == "app.py":
        sub('image_history not in ("full", "recent-observations")', 'image_history not in ("full", "full-retrievable", "recent-observations")')
        sub('image_context=self.history.reference_pins if self.history else None,',
            'image_context=self.history.reference_pins if self.history else None,\n                                              image_policy=self.image_history if self.history else None,')
        sub('choices=("full", "recent-observations")', 'choices=("full", "full-retrievable", "recent-observations")')
    elif name == "bridge.py":
        sub('allowed_tools=None, image_context=None):', 'allowed_tools=None, image_context=None, image_policy=None):')
        sub('    if image_context is not None and',
            '    if (image_policy is not None and image_policy not in ("full-retrievable", "recent-observations")) or (image_policy is not None and image_context is None):\n        raise ValueError("Image policy requires the task-owned retrieval callback")\n    if image_context is not None and')
        sub('present_request(converted.body, image_context(), trace)',
            'present_request(converted.body, image_context(), trace, image_policy or "recent-observations")')
    else:
        sub('from .visual_working_context import working_context', 'from .visual_context_policy import select')
        sub('def present_request(body, pins, trace):', 'def present_request(body, pins, trace, policy="recent-observations"):')
        sub('working_context(body, pins)', 'select(body, pins, policy)')
    compile(text, "isolated-image-comparator-" + name, "exec")
    return text.encode()


def build_candidate(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    build = json.loads((source / "build.json").read_text())
    inventory = build.get("candidate_files_sha256", {})
    if build.get("kind") != "isolated_visual_working_context_candidate" or any(inventory.get(n) != sha for n, sha in SOURCES.items()):
        raise ValueError("Require the declared immutable image-history source")
    contents = {}
    for name, sha in inventory.items():
        if Path(name).name != name:
            raise ValueError("Use flat package filenames")
        data = (source / "framework/qw3_computer" / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError("Candidate source changed: " + name)
        contents[name] = patch(name, data) if name in SOURCES else data
    helper = Path(__file__).with_name("visual_context_policy.py")
    contents[helper.name] = helper.read_bytes()
    compile(contents[helper.name], helper.name, "exec")
    if output == source or output.is_relative_to(source):
        raise ValueError("Keep the comparison outside the immutable source")
    output.mkdir(parents=True, exist_ok=False)
    package = output / "framework/qw3_computer"
    package.mkdir(parents=True)
    for name, data in contents.items():
        (package / name).write_bytes(data)
    result = {"kind": "isolated_matched_visual_context_comparison_candidate",
        "source": str(source), "source_build_sha256": hashlib.sha256((source / "build.json").read_bytes()).hexdigest(),
        "source_files_sha256": inventory, "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(contents[helper.name]).hexdigest(),
        "candidate_files_sha256": {n: hashlib.sha256(data).hexdigest() for n, data in contents.items()},
        "changed_package_files": sorted([*SOURCES, helper.name]),
        "default_policy": "full", "comparison_policies": ["full-retrievable", "recent-observations"],
        "matched": ["history tools", "history guidance", "source validation", "full-source archive", "checkpoints", "input guards", "task budget", "execution predicates"],
        "real_model_calls": 0, "native_device_inputs": 0, "deployed": False,
        "business_success_improvement_proven": False,
        "scope": "Full retrievable and recent observations have the same tools and instructions; only old eligible observation image parts differ. Original default full mode is unchanged."}
    (output / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = build_candidate(args.source, args.output)
    print(json.dumps({k: result[k] for k in ("kind", "comparison_policies", "changed_package_files")}))


if __name__ == "__main__":
    main()
