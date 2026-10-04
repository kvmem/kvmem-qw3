"""Administrative native field probe; never a QW3 task or business acceptance.

Reads the private candidate's AT-SPI provider. All input goes through the
existing, exclusively borrowed gateway so its physical-device lease and durable
receipts remain authoritative. Restore the filename without saving any file.
"""
import argparse
import asyncio
import hashlib
import io
import json
from pathlib import Path
import re
import time
import uuid

from PIL import Image

import execution_retest_queue as base
import execution_retest_queue_v2 as successor
from platform_pilots import linux_execute, owned_container
from linux_recording import LinuxRecording
from bounded_linux_observation import BoundedOfficeLinuxDriver
from qw3_computer.device_lease import DeviceLease
from qw3_computer.field_identity import identify_fields
from qw3_computer.execution import check_observation
from qw3_computer.targeting import identify_controls


def guest_json(source):
    return json.loads(linux_execute(source)["output"])


def windows():
    return guest_json("import subprocess,json;lines=subprocess.check_output(['wmctrl','-lx'],text=True).splitlines();"
        "print(json.dumps([dict(zip(['handle','desktop','class','host','title'],line.split(None,4))) for line in lines if len(line.split(None,4))==5]))")


def profile_identity():
    return guest_json("from pathlib import Path;import hashlib,json;p=Path('/home/user/.profile');"
        "print(json.dumps({'exists':p.is_file(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}))")


class Probe:
    def __init__(self, args):
        self.args = args
        p, _ = successor.validate(args.protocol)
        successor.PROTOCOL = p
        successor.DECLARATION = base.read(p["quarantine_declaration"])
        self.protocol = p
        self.item = p["previous_gateways"]["linux-vm"]["item"]
        base.owned.assert_apps_idle(p["previous_app"])
        self.before_ledger = successor.scoped_ledger(Path(self.item["token_file"]).parent)
        self.vm = owned_container("qw3-computer-linux", "qw3.goal", "cross-device-20261003")
        queue = base.read(base.ROOT / p["parent_osworld_queue"])
        if queue.get("status") == "running":
            active = base.read((base.ROOT / p["parent_osworld_queue"]).parent / "attempts" / queue["current_trial"] / "vm.json")
            if active["container_id"] == self.vm["Id"]:
                raise base.SafetyHalt("Administrative pilot is the active model VM")
        build = base.read(args.candidate / "build.json")
        if build["kind"] != "isolated_post51_native_field_identity_candidate":
            raise base.SafetyHalt("Declare the isolated native-field candidate")
        import qw3_computer.linux_driver as driver_module
        package = Path(driver_module.__file__).resolve().parent
        if package != (args.candidate / "framework/qw3_computer").resolve():
            raise base.SafetyHalt("PYTHONPATH does not select the declared private observer")
        for name, digest in build["candidate_files_sha256"].items():
            if base.owned.digest(package / name) != digest:
                raise base.SafetyHalt("Private candidate source changed: " + name)
        args.output.mkdir(parents=True, exist_ok=False)
        self.record = {"kind": "administrative_actual_native_field_identity_qualification",
            "started_at": time.time(), "passed": False, "status": "prepared", "model_calls": 0,
            "real_model_retest": False, "business_task_passed": False,
            "administrative_input_requests": 0, "confirmed_delivered_inputs": 0,
            "window_activations": 0, "input_route": "existing gateway with its unchanged physical driver lease",
            "existing_gateway_core_hotpatched": False, "new_physical_gateway_started": False,
            "field_evidence_scope": "Actual foreground native observations paired with existing-gateway administrative input. Model/Codex acceptance is separate.",
            "container_id": self.vm["Id"], "build_sha256": base.owned.digest(args.candidate / "build.json"),
            "recipe_sha256": base.owned.digest(Path(__file__)), "actions": [], "checks": {}}
        self.task = "admin-native-field-" + uuid.uuid4().hex
        self.original = self.target = self.field = self.before = None
        self.lease = self.driver = None
        self.recorder = LinuxRecording(args.output, self.vm["Id"])
        self.changed = False
        self.sentinel = "qw3-native-field-qualification.txt"
        self.save()

    def save(self):
        base.save(self.args.output / "result.json", self.record)

    def activate(self, handle):
        if not re.fullmatch(r"0x[0-9a-fA-F]+", handle):
            raise base.SafetyHalt("Invalid owned window handle")
        linux_execute("import subprocess;subprocess.run(['wmctrl','-ia'," + repr(handle) + "],check=True,timeout=5)")
        self.record["window_activations"] += 1
        self.save()

    async def capture(self, name):
        raw = await self.driver.observe()
        if raw["foreground"] != self.target or raw.get("accessibility", {}).get("status") != "ok":
            raise base.SafetyHalt("Native foreground identity is not established")
        png = raw.pop("png")
        with Image.open(io.BytesIO(png)) as image:
            width, height = image.size
        context = self.gateway_context
        observed = raw | {"desktop_id": context["desktop_id"], "device_id": context["device_id"],
            "session_id": context["session_id"], "display_id": context["display_id"],
            "observation_id": "admin-native-" + uuid.uuid4().hex, "captured_at": int(time.time()*1000),
            "width": width, "height": height, "sha256": hashlib.sha256(png).hexdigest(),
            "capabilities": self.driver.capabilities, "administrative_capture": True}
        observed = identify_fields(identify_controls(observed), self.task)
        path = self.args.output / (name + ".png")
        path.write_bytes(png); path.chmod(0o600)
        base.save(self.args.output / (name + ".json"), observed)
        return observed

    def send(self, action):
        # Fresh source IDs are obtained from the existing physical input owner.
        observed = base.gateway_api(self.item, "/observe")
        if (observed["foreground"] != self.target
                or observed["desktop_id"] != self.gateway_context["desktop_id"]):
            raise base.SafetyHalt("Physical gateway focus/session changed; no input sent")
        identity = self.task + "-" + str(len(self.record["actions"]))
        saved = {"action_id": identity, "status": "request_pending", "kind": action["kind"],
                 "source_observation_id": observed["observation_id"]}
        self.record["actions"].append(saved)
        self.record["administrative_input_requests"] += 1
        self.save()
        result = base.gateway_api(self.item, "/act", {"action_id": identity,
            "desktop_id": observed["desktop_id"], "observation_id": observed["observation_id"], "action": action})
        if (result.get("status") != "completed" or result.get("action_id") != identity
                or result.get("observation", {}).get("desktop_id") != observed["desktop_id"]):
            raise base.SafetyHalt("Original delivery receipt does not match; no automatic retry")
        saved.update(status="completed", receipt_sha256=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest())
        self.record["confirmed_delivered_inputs"] += 1
        self.save()

    async def restore_field(self):
        if not self.changed:
            return
        if any(a["status"] != "completed" for a in self.record["actions"]):
            raise base.SafetyHalt("Administrative delivery is unconfirmed; pause before further input")
        current = await self.capture("before-restore")
        matches = [c for c in current["accessibility"]["controls"]
                   if c.get("provider_identity") == self.field["provider_identity"] and c.get("editable") is True]
        if len(matches) != 1:
            raise base.SafetyHalt("Original editable native object is no longer visible; no guessed restore")
        # The known old field is still visible. Focus it from its new observed
        # bounds, never from an old coordinate or a changed name.
        self.send({"kind":"click","x":matches[0]["center_x"],"y":matches[0]["center_y"],"button":"left","clicks":1})
        self.send({"kind":"key","key":"CTRL+A"})
        self.send({"kind":"type","text":self.field["value"]})
        restored = await self.capture("restored")
        check = {"kind":"field_value","field_ref":self.field["field_ref"],"value":self.field["value"]}
        self.record["checks"]["filename_restored_and_reference_still_matches"] = check_observation([check],restored,self.before)[0]["passed"]
        if not self.record["checks"]["filename_restored_and_reference_still_matches"]:
            raise base.SafetyHalt("Filename restoration did not verify")
        self.changed = False

    async def run(self):
        initial = base.request(self.protocol["previous_app"], "/api/state")["devices"]["selected"]
        self.record["original_selected_device"] = initial
        profile_before = profile_identity()
        try:
            self.recorder.start()
            if initial == "linux-vm":
                changed = base.request(self.protocol["previous_app"], "/api/device", {"device_id":"windows-vm"})
                if changed["status"] != "idle" or changed.get("uncertain_actions"):
                    raise base.SafetyHalt("Paused input ownership transfer did not verify")
            self.lease = DeviceLease(self.item["endpoint"]).acquire()
            self.gateway_context = base.gateway_api(self.item, "/observe")
            if self.gateway_context.get("paused") is not True:
                raise base.SafetyHalt("Existing device is not paused")
            self.original = self.gateway_context["foreground"]
            self.record["original_foreground"] = self.original
            self.record["original_gateway_session_id"] = self.gateway_context["session_id"]
            self.save()
            self.driver = BoundedOfficeLinuxDriver("http://127.0.0.1:18126", "linux-vm")
            await self.driver.start()
            matches = [w for w in windows() if w["title"] == "Save As" and "gedit" in w["class"].lower()]
            if len(matches) != 1:
                raise base.SafetyHalt("Expected one retained Gedit file chooser")
            self.activate(matches[0]["handle"])
            self.target = await self.driver.foreground()
            if int(self.target["handle"],16) != int(matches[0]["handle"],16) or self.target["title"] != "Save As":
                raise base.SafetyHalt("File chooser activation did not verify")
            self.before = await self.capture("before")
            fields = [c for c in self.before["accessibility"]["controls"]
                      if c.get("editable") is True and c.get("kind") == "text" and c.get("value") == ".profile" and c.get("field_ref")]
            if len(fields) != 1:
                raise base.SafetyHalt("Expected one positively observed editable filename reference")
            self.field = fields[0]
            self.record["before_field_ref"] = self.field["field_ref"]
            self.record["before_field_native_identity"] = self.field["provider_identity"]
            self.save()
            base.gateway_api(self.item, "/control", {"command":"resume"})
            self.send({"kind":"click","x":self.field["center_x"],"y":self.field["center_y"],"button":"left","clicks":1})
            self.send({"kind":"key","key":"CTRL+A"})
            self.changed = True  # Restore even if later capture or assertions fail.
            self.send({"kind":"type","text":self.sentinel})
            after = await self.capture("after")
            new = {"kind":"field_value","field_ref":self.field["field_ref"],"value":self.sentinel}
            old = {"kind":"control_value","text":self.field["name"],"value":self.sentinel,"control_kind":"text"}
            native = [c for c in after["accessibility"]["controls"] if c.get("field_ref") == self.field["field_ref"]]
            self.record["checks"].update(
                actual_native_object_reference_survives_name_and_value_change=len(native)==1 and native[0]["name"]==self.sentinel,
                explicit_native_field_value_condition_verifies=check_observation([new],after,self.before)[0]["passed"],
                old_exact_name_condition_remains_false=not check_observation([old],after,self.before)[0]["passed"],
                passive_labels_do_not_mint_editable_references=all(c.get("editable") is True for c in after["accessibility"]["controls"] if c.get("field_ref")))
            self.record["before_field_ref"] = self.field["field_ref"]
            self.record["before_field_native_identity"] = self.field["provider_identity"]
            self.record["native_references_in_capture"] = sum(bool(c.get("field_ref")) for c in after["accessibility"]["controls"])
            await self.restore_field()
        except BaseException as exc:
            self.record["error"] = type(exc).__name__ + ": " + str(exc)
        finally:
            if self.changed:
                try: await self.restore_field()
                except BaseException as exc: self.record["restore_error"] = type(exc).__name__ + ": " + str(exc)
            try:
                paused = base.gateway_api(self.item,"/control",{"command":"pause"})
                self.record["checks"]["existing_gateway_paused_again"] = paused.get("paused") is True
                if self.original is not None and not self.changed:
                    known = [w for w in windows() if int(w["handle"],16)==int(self.original["handle"],16) and w["title"]==self.original["title"]]
                    if len(known)!=1: raise base.SafetyHalt("Original window identity changed")
                    self.activate(known[0]["handle"])
                    self.record["checks"]["original_foreground_restored"] = await self.driver.foreground() == self.original
            except BaseException as exc:self.record["pause_or_window_restore_error"]=type(exc).__name__+": "+str(exc)
            if self.driver is not None: await self.driver.close()
            if self.lease is not None:self.lease.close()
            try:
                restored = base.request(self.protocol["previous_app"],"/api/device",{"device_id":initial})
                self.record["checks"]["paused_device_selection_restored"] = (restored["status"]=="idle" and restored.get("desktop_paused") is True and not restored.get("uncertain_actions") and restored["devices"]["selected"]==initial)
                ledger = successor.scoped_ledger(Path(self.item["token_file"]).parent)
                self.record["checks"]["current_unknown_zero_historical_unknown_unchanged"] = ledger["current_unknowns"]==0 and ledger["quarantined_unknowns_preserved"]==self.before_ledger["quarantined_unknowns_preserved"]
                self.record["checks"]["profile_file_unchanged"] = profile_identity()==profile_before
                self.record["checks"]["all_administrative_inputs_have_original_completed_receipts"] = all(a["status"]=="completed" for a in self.record["actions"])
            except BaseException as exc:self.record["final_gate_error"]=type(exc).__name__+": "+str(exc)
            try:
                receipt = self.recorder.finish()
                self.record["checks"]["continuous_vnc_recording_complete"] = receipt.get("complete") is True
                self.record["video"] = str(self.args.output/"continuous.mp4")
            except BaseException as exc:self.record["recording_error"]=type(exc).__name__+": "+str(exc)
            self.record["passed"] = (len(self.record["checks"])==12 and all(self.record["checks"].values())
                and not any("error" in k for k in self.record) and not self.changed)
            self.record.update(status="completed" if self.record["passed"] else "failed-preserved",finished_at=time.time())
            self.save()
        return self.record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(Probe(args).run())
    print(json.dumps({k:result.get(k) for k in ("passed","checks","confirmed_delivered_inputs","window_activations","model_calls","error","restore_error","final_gate_error","recording_error")}))
    if not result["passed"]:raise SystemExit(1)


if __name__ == "__main__":main()
