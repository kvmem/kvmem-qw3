"""Build an isolated feedback experiment without altering result predicates."""
import argparse
import hashlib
import json
from pathlib import Path

from post_sdk51_candidate import replace_once


EXECUTION_SHA256 = "768dc126331d4c527501846910da2ee26c4840aebabc0539afa14ad75f2c36d9"


def patch_execution(original):
    if hashlib.sha256(original).hexdigest() != EXECUTION_SHA256:
        raise ValueError("Use the declared native-field execution source")
    text = replace_once(original.decode(),
        "from .field_identity import field_value_evidence, FIELD_PATTERN\n",
        "from .field_identity import field_value_evidence, FIELD_PATTERN\nfrom .completion_feedback import explain_condition\n")
    text = replace_once(text, "evidence.append(field_value_evidence(check, observation))",
                        "evidence.append(explain_condition(field_value_evidence(check, observation), observation))")
    text = replace_once(text, "        evidence.append(item)\n", "        evidence.append(explain_condition(item, observation))\n")
    compile(text, "isolated-completion-feedback.py", "exec")
    return text.encode()


def build_candidate(source_package, source_build, output):
    source_package, source_build, output = map(lambda p: Path(p).resolve(), (source_package, source_build, output))
    manifest = json.loads(source_build.read_text())
    inventory = manifest.get("candidate_files_sha256", {})
    if (manifest.get("kind") != "isolated_post51_native_field_identity_candidate"
            or inventory.get("execution.py") != EXECUTION_SHA256):
        raise ValueError("Declare the isolated native-field source inventory")
    contents = {}
    for name, digest in inventory.items():
        if Path(name).name != name or Path(name).suffix not in {".py", ".html", ".json", ".tex"}:
            raise ValueError("Invalid package filename")
        data = (source_package / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Private source changed: " + name)
        contents[name] = data
    contents["execution.py"] = patch_execution(contents["execution.py"])
    helper = Path(__file__).with_name("completion_feedback.py")
    contents[helper.name] = helper.read_bytes()
    if output == source_package or output.is_relative_to(source_package):
        raise ValueError("Keep the experiment outside its immutable source")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "framework/qw3_computer"
    destination.mkdir(parents=True)
    for name, data in contents.items():
        (destination / name).write_bytes(data)
    result = {"kind": "isolated_observation_comparison_feedback_candidate", "qualified_for_production": False,
        "native_windows_qualified": False, "real_model_retest": False, "model_calls": 0, "device_inputs": 0,
        "source_package": str(source_package), "source_files_sha256": inventory,
        "source_build_sha256": hashlib.sha256(source_build.read_bytes()).hexdigest(),
        "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "candidate_files_sha256": {n: hashlib.sha256(v).hexdigest() for n, v in contents.items()},
        "changed_package_files": [n for n,v in contents.items() if hashlib.sha256(v).hexdigest()!=inventory.get(n)],
        "scope": "Add literal comparison and evidence-scope feedback. Original condition booleans, exact matching, delivery receipts, budgets, pause and pending-step rules unchanged. No inferred app or field identity."}
    (output / "build.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-package",type=Path,required=True)
    p.add_argument("--source-build",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=build_candidate(args.source_package,args.source_build,args.output)
    print(json.dumps({k:result[k] for k in ("kind","changed_package_files","real_model_retest")}))


if __name__=="__main__":main()
