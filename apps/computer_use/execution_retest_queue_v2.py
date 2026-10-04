"""Frozen successor wrapper with explicit historical-quarantine admission.

The first waiting controller and protocol remain unchanged. All model trials,
source packages, initializers, evaluators, budgets and owned-process gates are
reused. Only the known historical Linux receipt receives scoped admission.
"""
import argparse
import json
from pathlib import Path
import signal

import execution_retest_queue as base
import quarantined_receipt_admission as quarantine
from osworld_windows.source import ROOT


BASE_VALIDATE = base.validate
BASE_LEDGER = base.ledger_terminal
PROTOCOL = None
DECLARATION = None


def validate(path):
    p, kp = BASE_VALIDATE(path)
    if (base.owned.digest(Path(__file__)) != p["successor_wrapper_sha256"]
            or base.owned.digest(Path(quarantine.__file__)) != p["quarantine_admission_sha256"]
            or base.owned.digest(p["quarantine_declaration"]) != p["quarantine_declaration_sha256"]):
        raise base.SafetyHalt("Frozen successor admission source changed")
    declaration = base.read(p["quarantine_declaration"])
    quarantine.validate_evidence(declaration)
    previous = base.read(p["previous_waiting_controller_retirement"])
    if (base.owned.digest(p["previous_waiting_controller_retirement"]) != p["previous_waiting_controller_retirement_sha256"]
            or previous.get("passed") is not True or previous.get("model_trials_dispatched") != 0
            or previous.get("live_child_or_engine_owned") is not False
            or previous.get("original_protocol_changed") is not False):
        raise base.SafetyHalt("Previous zero-trial waiting controller was not safely retired")
    if not base.process_retired(previous["controller_identity"]):
        raise base.SafetyHalt("Previous waiting controller remains alive")
    if base.owned.digest(previous["original_protocol"]) != previous["original_protocol_sha256"]:
        raise base.SafetyHalt("Original zero-trial protocol changed")
    return p, kp


def scoped_ledger(runtime):
    if Path(runtime).resolve() != Path(DECLARATION["gateway_runtime"]).resolve():
        return BASE_LEDGER(runtime)
    registration = PROTOCOL["previous_gateways"]["linux-vm"]
    base.require_identity(registration["identity"])
    observed = base.gateway_api(registration["item"], "/observe")
    from platform_pilots import owned_container, linux_execute
    container = owned_container("qw3-computer-linux", "qw3.goal", "cross-device-20261003")
    boot = json.loads(linux_execute("from pathlib import Path;import json;print(json.dumps({'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))")["output"])["boot_id"]
    context = {"paused": observed.get("paused"), "container_running": container["State"]["Running"],
               "container_owner": container["Config"].get("Labels", {}).get("qw3.goal"),
               "container_id": container["Id"], "guest_boot_id": boot,
               "gateway_pid": registration["identity"]["pid"], "session_id": observed.get("session_id"),
               "desktop_id": observed.get("desktop_id")}
    return quarantine.admit_rows(quarantine.ledger_rows(runtime), DECLARATION, context)


class Controller(base.Controller):
    def __init__(self, args, protocol, state):
        super().__init__(args, protocol, state)
        self.state.update(successor_wrapper_sha256=protocol["successor_wrapper_sha256"],
                          quarantine_declaration_sha256=protocol["quarantine_declaration_sha256"],
                          historical_unknown_not_reclassified_or_replayed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=172800)
    parser.add_argument("--resource-wait-seconds", type=int, default=21600)
    parser.add_argument("--min-free-mib", type=int, default=56000)
    args = parser.parse_args()
    if args.min_free_mib < 56000 or not (60 <= args.wait_seconds <= 172800
                                       and 60 <= args.resource_wait_seconds <= 86400):
        parser.error("Use the original bounded shared-resource profile")
    args.protocol, args.state_dir = args.protocol.resolve(), args.state_dir.resolve()
    global PROTOCOL, DECLARATION
    PROTOCOL, _ = validate(args.protocol)
    DECLARATION = base.read(PROTOCOL["quarantine_declaration"])
    # In this new process only. The original live controller/module is never
    # patched on disk, and every unchanged base gate is still applied.
    base.validate = validate
    base.ledger_terminal = scoped_ledger
    base.Controller = Controller
    def interrupted(*_):
        raise base.SafetyHalt("Scoped successor interrupted; live work preserved")
    signal.signal(signal.SIGTERM, interrupted)
    outcome = base.run(args)
    print(json.dumps({"status": outcome["status"], "trials": len(outcome["trials"])}))


if __name__ == "__main__":
    main()
