"""Build control/candidate copies of the qualified SDK51 controller.

Running or queued packages are read only. Only the candidate's turn lifecycle
changes; model/device tools, conditions, budgets and receipts stay unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path


SDK51_APP_SHA256 = "0df0b0d94c66d30695d7c226450ffc1b205e580aa5ff056da2a94d038b9063a1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Qualified controller source anchor changed")
    return text.replace(before, after, 1)


def patch_app(data):
    if digest(data) != SDK51_APP_SHA256:
        raise ValueError("Use the exact qualified SDK51 controller")
    text = data.decode()
    text = replace_once(text, "class Controller:\n", "from .bounded_turn_continuation import BoundedTurnContinuation, checkpoint\n\n\nclass Controller(BoundedTurnContinuation):\n")
    text = replace_once(text, "        self.ready = False\n        self.client = None\n",
                        "        self.bounded_turn_owner = uuid.uuid4().hex\n        self.ready = False\n        self.client = None\n")
    text = replace_once(text, '                    status = params["turn"]["status"]\n',
                        '                    bounded_turn_origin = self.state.get("status")\n                    status = params["turn"]["status"]\n')
    text = replace_once(text, '                    self.state["finished_at"] = time.time()\n',
                        '                    continued = await self.continue_unchecked_zero_tool_turn(bounded_turn_origin)\n'
                        '                    if not continued:\n                        self.state["finished_at"] = time.time()\n')
    text = replace_once(text, '            self.state["execution_mode"] = "verified" if self.workflow else "legacy"\n',
                        '            self.state["bounded_turn_continuation"] = checkpoint(self.bounded_turn_owner, 0)\n'
                        '            self.state["execution_mode"] = "verified" if self.workflow else "legacy"\n')
    text = replace_once(text, '            else:\n                if command == "pause" and self.state["status"] not in ("starting", "running", "paused"):\n',
                        '            else:\n                self.state["bounded_turn_user_control"] = command\n'
                        '                if command == "pause" and self.state["status"] not in ("starting", "running", "paused"):\n')
    compile(text, "isolated-bounded-turn-app.py", "exec")
    return text.encode()


def build(package, acceptance, output):
    package, acceptance, output = map(Path, (package, acceptance, output))
    if output.exists():
        raise ValueError("Use a fresh isolated candidate directory")
    declared = json.loads(acceptance.read_text())["framework_files"]
    files = {}
    for name, expected in declared.items():
        if not name.startswith("qw3_computer/"):
            continue
        parts = Path(name).parts
        if len(parts) != 2 or Path(name).suffix not in {".py", ".html", ".json", ".tex"}:
            raise ValueError("Unexpected qualified package inventory path")
        source = package / parts[1]
        if source.is_symlink() or not source.is_file():
            raise ValueError("Qualified package file is not regular")
        data = source.read_bytes()
        if digest(data) != expected:
            raise ValueError("Qualified package source changed: " + name)
        files[parts[1]] = data
    if digest(files["app.py"]) != SDK51_APP_SHA256:
        raise ValueError("Acceptance is not the qualified SDK51 source")
    candidate = dict(files)
    candidate["app.py"] = patch_app(files["app.py"])
    candidate["bounded_turn_continuation.py"] = Path(__file__).with_name("bounded_turn_continuation.py").read_bytes()
    manifests = {}
    for arm, contents in (("control", files), ("candidate", candidate)):
        target = output / arm / "qw3_computer"
        target.mkdir(parents=True, exist_ok=False)
        for name, data in contents.items():
            (target / name).write_bytes(data)
        manifests[arm] = {name: digest(data) for name, data in contents.items()}
    report = {"kind": "isolated_bounded_zero_tool_turn_candidate", "source_package": str(package.resolve()),
              "source_acceptance_sha256": digest(acceptance.read_bytes()), "source_app_sha256": SDK51_APP_SHA256,
              "framework_files": manifests, "changed_files": ["app.py", "bounded_turn_continuation.py"],
              "source_pins_unchanged": True, "production_deployed": False, "real_model_retest": False,
              "source_sha256": {p.name: digest(p.read_bytes()) for p in
                                 (Path(__file__), Path(__file__).with_name("bounded_turn_continuation.py"))}}
    (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "acceptance", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    report = build(args.package, args.acceptance, args.output)
    print(json.dumps({k: report[k] for k in ("kind", "changed_files", "production_deployed", "real_model_retest")}))


if __name__ == "__main__":
    main()
