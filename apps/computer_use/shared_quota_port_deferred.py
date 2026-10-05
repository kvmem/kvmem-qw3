"""Reconnect the untouched bounded/query cohorts after port recovery.

The frozen controllers, task packages and saved requests are reused unchanged.
New protocols change only predecessor routing and pins for those predecessors.
All original terminal receipt, VM, service and final engine checks still run.
"""
import argparse
import copy
import os
from pathlib import Path
import signal
import sys
import time
import types

import bounded_continuation_retest_queue as original_bounded
import query_refresh_replay_queue as original_query
import shared_quota_port_successors as successors

q = successors.RUNTIME
ROOT, RUNTIME, read, save, owned, SafetyHalt = q.ROOT, q.image.RUNTIME, q.read, q.save, q.owned, q.SafetyHalt
MODULE = "shared_quota_port_deferred"
STAGES = ("bounded", "query")
COUNTS = dict(bounded=8, query=4)
HELPERS = (MODULE + ".py", "tests/test_" + MODULE + ".py",
           "tests/test_bounded_continuation_retest_queue.py", "tests/test_query_refresh_replay_queue.py",
           "shared_quota_port_successors.py")
SOURCE_SHA = dict(bounded="d6c83a2abae42355c77ac667f9a7ea7877fe4b0b7a2b69de9afbc1a540c2eb02",
                  query="dfc098cb20b2a28f794a31daf5337b2d34f24ba5647a78c6ddb6ca270af313d2")
PROTOCOL_SHA = dict(bounded="c2cb89afc5cb7fef6b9383630675431e6f4c53cbd8248e3163c08f513b001bcd",
                    query="7e22b2e94b1c0f43d7bc4e974e7d79c0d3568262344aa48562070ff2dceead27")
SOURCE_PROTOCOLS = dict(bounded=RUNTIME / "bounded-continuation-retest-20261005/protocol.json",
                        query=RUNTIME / "query-refresh-readonly-20261005/protocol.json")
SOURCE_REGISTRATIONS = dict(bounded=RUNTIME / "bounded-continuation-retest-launch-01/registration.json",
                            query=RUNTIME / "query-refresh-replay-launch-01/registration.json")
FOUR_DECLARATION = RUNTIME / "shared-quota-port-successors-20261005-01/successors.json"
ROUTING = ("parent_registration", "parent_protocol", "parent_queue", "parent_controller")


def qualification(path):
    result = read(path)
    if (result.get("kind") != "shared_quota_port_deferred_cpu_qualification"
            or result.get("passed") is not True or result.get("tests_run", 0) < 38
            or result.get("skipped") != 0 or result.get("model_calls") != 0 or result.get("device_inputs") != 0):
        raise SafetyHalt("Require completed deferred port-routing qualification")
    for name in HELPERS:
        q.image.pinned(ROOT / "apps/computer_use" / name, result["source_sha256"][name])


def original_scope(stage):
    path = SOURCE_PROTOCOLS[stage]
    q.image.pinned(path, PROTOCOL_SHA[stage])
    reg = read(SOURCE_REGISTRATIONS[stage])
    state = read(reg["queue"])
    if (reg.get("protocol") != str(path) or reg.get("protocol_sha256") != PROTOCOL_SHA[stage]
            or reg.get("controller_sha256") != SOURCE_SHA[stage]
            or reg.get("planned_trials") != COUNTS[stage]
            or state.get("status") != "halted-no-further-dispatch" or state.get("trials") != []
            or state.get("controller_sha256") != SOURCE_SHA[stage]
            or state.get("protocol_sha256") != PROTOCOL_SHA[stage]
            or reg["identity"].get("uid") != os.getuid() or reg["identity"].get("cwd") != str(ROOT)
            or owned.process_identity(reg["identity"]["pid"]) is not None):
        raise SafetyHalt("Only the original retired zero-trial deferred cohorts may be rebound")
    return read(path), reg


def routed_protocol(stage, original, parent, declaration=None):
    result = copy.deepcopy(original)
    result.update(parent_registration=parent["registration_path"], parent_protocol=parent["protocol"],
                  parent_queue=parent["queue"], parent_controller=copy.deepcopy(parent["identity"]))
    if stage == "bounded":
        result["parent_declaration"] = str(declaration)
    old_paths = {original["parent_registration"], original["parent_protocol"]}
    if stage == "bounded": old_paths.add(original["parent_declaration"])
    replacements = [Path(result["parent_registration"]), Path(result["parent_protocol"])]
    if stage == "bounded": replacements.append(Path(result["parent_declaration"]))
    result["files"] = [copy.deepcopy(item) for item in original["files"] if item["path"] not in old_paths]
    result["files"] += [dict(path=str(path), sha256=owned.digest(path)) for path in dict.fromkeys(replacements)]
    return result


def route_proof(stage, original, protocol):
    parent = dict(registration_path=protocol["parent_registration"], protocol=protocol["parent_protocol"],
                  queue=protocol["parent_queue"], identity=protocol["parent_controller"])
    expected = routed_protocol(stage, original, parent, protocol.get("parent_declaration"))
    if expected != protocol:
        raise SafetyHalt("Deferred copy changed more than the exact predecessor routing")


def validate_declaration(path, stage):
    d = read(path)
    if (d.get("kind") != "qualified_frozen_port_deferred_routing" or d.get("stage") != stage
            or d.get("planned_trials") != COUNTS[stage]
            or d.get("controller_python") != str(q.CONTROLLER_PYTHON)
            or d.get("original_protocol") != str(SOURCE_PROTOCOLS[stage])
            or d.get("original_registration") != str(SOURCE_REGISTRATIONS[stage])
            or d.get("only_predecessor_routing_changed") is not True
            or d.get("old_failures_overwritten") is not False
            or d.get("model_tasks_budgets_or_saved_inputs_changed") is not False
            or d.get("protocol") != str(Path(path).parent / "protocol.json")
            or d.get("state_dir") != str(Path(path).parent / "queue-runtime-01")):
        raise SafetyHalt("Deferred routing declaration scope changed")
    for item in d["files"]: q.image.pinned(item["path"], item["sha256"])
    qualification(d["qualification"])
    original, _ = original_scope(stage)
    protocol = read(d["protocol"])
    route_proof(stage, original, protocol)
    if stage == "bounded" and protocol["parent_declaration"] != str(FOUR_DECLARATION):
        raise SafetyHalt("Use the actual restored four-stage parent declaration")
    return d


def deferred_registration(path, stage):
    record = read(path)
    d = validate_declaration(Path(record["declaration"]), stage)
    command = [str(q.CONTROLLER_PYTHON), "-m", MODULE, "--declaration", record["declaration"], "--stage", stage]
    if (record.get("route_kind") != "actual_frozen_port_deferred_identity"
            or record.get("stage") != stage or record.get("planned_trials") != COUNTS[stage]
            or record.get("declaration_sha256") != owned.digest(record["declaration"])
            or record.get("controller_sha256") != owned.digest(__file__)
            or record.get("protocol") != d["protocol"]
            or record.get("protocol_sha256") != owned.digest(d["protocol"])
            or record.get("queue") != str(Path(d["state_dir"]) / "queue.json")
            or record["identity"].get("command") != command
            or record["identity"].get("cwd") != str(ROOT) or record["identity"].get("uid") != os.getuid()):
        raise SafetyHalt("Actual deferred producer registration changed")
    state = q.old.guard_queue(read(record["queue"]), record["identity"], record["protocol_sha256"], "protocol_sha256")
    kind = "serial_matched_bounded_continuation_retest" if stage == "bounded" else "serial_saved_desktop_query_refresh_queue"
    if state.get("kind") != kind or state.get("controller_sha256") != owned.digest(__file__):
        raise SafetyHalt("Actual deferred queue source or kind changed")
    return record, state


def bounded_validate(path):
    if BOUNDED.DECLARATION is None:
        raise SafetyHalt("Configure the actual bounded declaration before validating")
    d = validate_declaration(BOUNDED.DECLARATION, "bounded")
    if str(path) != d["protocol"]: raise SafetyHalt("Bounded protocol path changed")
    return BOUNDED.FROZEN_VALIDATE(path)


def query_registration(path):
    record, state = deferred_registration(path, "bounded")
    BOUNDED.DECLARATION = Path(record["declaration"])
    return record, state


def query_validate(path):
    if QUERY.DECLARATION is None:
        raise SafetyHalt("Configure the actual query declaration before validating")
    d = validate_declaration(QUERY.DECLARATION, "query")
    if str(path) != d["protocol"]: raise SafetyHalt("Query protocol path changed")
    query_registration(read(path)["parent_registration"])
    return QUERY.FROZEN_VALIDATE(path)


def namespace(stage, source):
    q.image.pinned(source.__file__, SOURCE_SHA[stage])
    module = types.ModuleType("qualified_port_deferred_" + stage)
    module.__file__ = __file__
    # Keep the exact original source filename for inspect.getsource used by
    # the query handoff; its function globals still belong to this namespace.
    exec(compile(Path(source.__file__).read_text(), source.__file__, "exec"), module.__dict__)
    module.FROZEN_VALIDATE = module.validate
    module.__dict__.update(q=q, DECLARATION=None)
    return module


BOUNDED = namespace("bounded", original_bounded)
BOUNDED.validate = bounded_validate
QUERY = namespace("query", original_query)
QUERY.__dict__.update(bounded=BOUNDED, q=q, BOUNDED_SHA=owned.digest(__file__),
                      registration=query_registration, validate=query_validate)


def prepare(args):
    qualification(args.qualification)
    original, retired = original_scope(args.stage)
    if args.stage == "bounded":
        if args.parent != FOUR_DECLARATION: raise SafetyHalt("Use the restored four-stage declaration")
        d, _, _ = q.validate(args.parent)
        path = Path(d["feedback_registration"])
        parent = q.successor_registration(path, "feedback", args.parent, d)
        q.successor_queue(parent, args.parent)
        parent = dict(parent, registration_path=str(path))
        actual = q.frozen_protocol("feedback", Path(parent["protocol"]))
    else:
        record, state = query_registration(args.parent)
        parent = dict(record, registration_path=str(args.parent))
        actual = BOUNDED.validate(Path(record["protocol"]))
    if original["parent_app"] != actual["parent_app"] or original["engine_off_identity"] != actual["engine_off_identity"]:
        raise SafetyHalt("Original user service or model engine profile changed")
    protocol = routed_protocol(args.stage, original, parent, args.parent if args.stage == "bounded" else None)
    route_proof(args.stage, original, protocol)
    for spec in original["trials"]:
        if (SOURCE_PROTOCOLS[args.stage].parent / "attempts" / spec["id"]).exists():
            raise SafetyHalt("Original deferred attempt was already dispatched")
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    ppath = args.output / "protocol.json"
    save(ppath, protocol)
    paths = [SOURCE_PROTOCOLS[args.stage], SOURCE_REGISTRATIONS[args.stage], Path(retired["queue"]),
             ppath, args.qualification, args.parent, Path(parent["registration_path"])]
    paths += [ROOT / "apps/computer_use" / name for name in HELPERS]
    paths += [Path(original_bounded.__file__), Path(original_query.__file__)]
    d = dict(kind="qualified_frozen_port_deferred_routing", stage=args.stage, planned_trials=COUNTS[args.stage],
             declared_at=time.time(), controller_python=str(q.CONTROLLER_PYTHON), protocol=str(ppath),
             state_dir=str(args.output / "queue-runtime-01"), original_protocol=str(SOURCE_PROTOCOLS[args.stage]),
             original_registration=str(SOURCE_REGISTRATIONS[args.stage]), qualification=str(args.qualification),
             only_predecessor_routing_changed=True, old_failures_overwritten=False,
             model_tasks_budgets_or_saved_inputs_changed=False,
             files=[dict(path=str(path),sha256=owned.digest(path)) for path in dict.fromkeys(paths)])
    path = args.output / "declaration.json"
    save(path, d)
    module = BOUNDED if args.stage == "bounded" else QUERY
    module.DECLARATION = path
    module.validate(ppath)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--prepare", action="store_true")
    for name in ("qualification", "parent", "output", "declaration"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--wait-seconds", type=int, default=172800)
    parser.add_argument("--resource-wait-seconds", type=int, default=21600)
    parser.add_argument("--min-free-mib", type=int, default=56000)
    args = parser.parse_args()
    if not (60 <= args.wait_seconds <= 172800 and 60 <= args.resource_wait_seconds <= 86400 and args.min_free_mib == 56000):
        parser.error("Keep the exact shared resource and cumulative wait bounds")
    if args.prepare:
        for key in ("qualification", "parent", "output"):
            if getattr(args, key) is None: parser.error("Missing " + key)
            setattr(args, key, getattr(args, key).resolve())
        print(dict(declaration=str(prepare(args)), model_calls=0, device_inputs=0))
        return
    if args.declaration is None: parser.error("Provide the actual deferred declaration")
    args.declaration = args.declaration.resolve()
    d = validate_declaration(args.declaration, args.stage)
    if sys.executable != str(q.CONTROLLER_PYTHON): raise SafetyHalt("Keep the qualified lexical controller environment")
    args.protocol, args.state_dir = Path(d["protocol"]), Path(d["state_dir"])
    module = BOUNDED if args.stage == "bounded" else QUERY
    module.DECLARATION = args.declaration
    def interrupted(*_): raise SafetyHalt("Deferred waiter interrupted; same live work preserved")
    signal.signal(signal.SIGTERM, interrupted)
    print(dict(status=module.run(args)["status"]))


if __name__ == "__main__": main()
