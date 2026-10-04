"""Isolated follow-up to SDK51: window roles and durable dispatch phase evidence.

No active package, input reservation, retry, receipt status or grading changes.
The phase journal is diagnostic only: it never resolves an uncertain action.
"""
import argparse
import hashlib
import json
from pathlib import Path


SDK51_DESKTOP_SHA256 = "b504edbd0f178febcae1f95cc113d14ff4c4febba1a8b99cf1b02401c71f75f0"
SDK51_GATEWAY_SHA256 = "f24597a9a038c32c38bdf1659eb952b78f2f5d53542d879d9d7c3c82841ea6ec"


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Qualified SDK51 source anchor changed")
    return text.replace(before, after, 1)


def patch_desktop_bytes(original):
    if hashlib.sha256(original).hexdigest() != SDK51_DESKTOP_SHA256:
        raise ValueError("Use the exact qualified SDK51 Desktop source")
    text = replace_once(original.decode(), "c['kind'].casefold() in {'frame', 'window'}",
                        "c['kind'].casefold() in {'frame', 'window', 'dialog', 'file-chooser'}")
    compile(text, "isolated-sdk51-window-roles.py", "exec")
    return text.encode()


PHASE_METHODS = '''    def record_phase(self, action_id, phase, detail=None):
        """Commit adapter milestones without resolving the action status."""
        row = self.db.execute("SELECT request FROM actions WHERE id=?", (action_id,)).fetchone()
        if not row:
            raise RuntimeError("Phase evidence requires a reserved action")
        request = json.loads(row[0])
        evidence = {"action_id": action_id, "phase": phase,
                    "request_sha256": hashlib.sha256(row[0].encode()).hexdigest(),
                    "desktop_id": request["desktop_id"], "session_id": self.session_id,
                    "recorded_at": time.time(), "detail": detail,
                    "business_verified": False}
        self.db.execute("INSERT INTO action_phases VALUES (?,?,?)",
                        (action_id, phase, json.dumps(evidence)))
        self.db.commit()

    def phase_evidence(self, action_id):
        row = self.db.execute("SELECT request FROM actions WHERE id=?", (action_id,)).fetchone()
        if not row:
            return []
        request = json.loads(row[0])
        expected = hashlib.sha256(row[0].encode()).hexdigest()
        records = []
        for phase, serialized in self.db.execute(
                "SELECT phase,evidence FROM action_phases WHERE id=? ORDER BY rowid", (action_id,)):
            evidence = json.loads(serialized)
            if (evidence.get("action_id") != action_id or evidence.get("phase") != phase
                    or evidence.get("request_sha256") != expected
                    or evidence.get("desktop_id") != request["desktop_id"]
                    or evidence.get("business_verified") is not False):
                raise RuntimeError("Phase evidence does not match the reserved request")
            records.append(evidence)
        return records

'''


def patch_gateway_bytes(original):
    if hashlib.sha256(original).hexdigest() != SDK51_GATEWAY_SHA256:
        raise ValueError("Use the exact qualified SDK51 Gateway source")
    text = original.decode()
    text = replace_once(text, '        self.db.commit()\n        self.paused = True\n',
        '        self.db.execute("CREATE TABLE IF NOT EXISTS action_phases(id TEXT,phase TEXT,evidence TEXT,PRIMARY KEY(id,phase))")\n'
        '        self.db.commit()\n        self.paused = True\n')
    text = replace_once(text, '            # Exactly one driver call. Network timeouts are not automatically retried.\n',
        '            self.record_phase(action_id, "reserved")\n'
        '            # Exactly one driver call. Network timeouts are not automatically retried.\n')
    text = replace_once(text, '            delivery = await self.driver.dispatch(action)\n',
        '            delivery = await self.driver.dispatch(action)\n'
        '            self.record_phase(action_id, "driver_returned", delivery)\n')
    text = replace_once(text, '            result = {"action_id": action_id, "status": "completed",',
        '            self.record_phase(action_id, "post_input_observed",\n'
        '                              {"observation_id": observation["observation_id"], "captures": captures})\n'
        '            result = {"action_id": action_id, "status": "completed",')
    text = replace_once(text, '    def receipt(self, action_id):\n', PHASE_METHODS + '    def receipt(self, action_id):\n')
    text = replace_once(text, 'return {"status": row[0], "result": json.loads(row[1]) if row[1] else None} if row else {"status": "unknown"}',
        'return {"status": row[0], "result": json.loads(row[1]) if row[1] else None,\n'
        '                "phase_evidence": self.phase_evidence(action_id),\n'
        '                "phase_scope": "Adapter milestones only; no uncertain-action reconciliation, retry or business verification."} if row else {"status": "unknown"}')
    compile(text, "isolated-sdk51-dispatch-phases.py", "exec")
    return text.encode()


def build_candidate(source_package, native_manifest, output):
    source_package = Path(source_package).resolve()
    native_manifest = Path(native_manifest)
    files = json.loads(native_manifest.read_text()).get("source_files", {})
    files = {name.removeprefix("qw3_computer/"): digest for name, digest in files.items()
             if name.startswith("qw3_computer/")}
    if (files.get("desktop.py") != SDK51_DESKTOP_SHA256
            or files.get("device_gateway.py") != SDK51_GATEWAY_SHA256):
        raise ValueError("Declare the exact qualified SDK51 package inventory")
    contents = {}
    for name, digest in files.items():
        if Path(name).name != name or Path(name).suffix not in {".py", ".html", ".json", ".tex"}:
            raise ValueError("Invalid package filename")
        contents[name] = (source_package / name).read_bytes()
        if hashlib.sha256(contents[name]).hexdigest() != digest:
            raise ValueError("Qualified SDK51 package changed: " + name)
    contents["desktop.py"] = patch_desktop_bytes(contents["desktop.py"])
    contents["device_gateway.py"] = patch_gateway_bytes(contents["device_gateway.py"])
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
    manifest = {"kind": "isolated_post_sdk51_window_roles_and_phase_journal",
        "qualified_for_production": False, "real_model_retest": False,
        "native_windows_qualified": False, "model_calls": 0, "device_inputs": 0,
        "source_package": str(source_package), "source_files_sha256": files,
        "native_manifest_sha256": hashlib.sha256(native_manifest.read_bytes()).hexdigest(),
        "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_files_sha256": candidate_files,
        "changed_package_files": [name for name in files if files[name] != candidate_files[name]],
        "scope": "Include positively observed dialog/file-chooser window roles; persist driver-return evidence before capture. All original reservation/status/pause/retry/completion rules retained."}
    (output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_candidate(args.source_package, args.native_manifest, args.output)
    print(json.dumps({key: manifest[key] for key in ("kind", "changed_package_files", "real_model_retest")}))


if __name__ == "__main__":
    main()
