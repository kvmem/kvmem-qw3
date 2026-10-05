"""Reconnect frozen successor trials to the qualified memory continuation.

Only predecessor routing, fresh controller state and the SDK trial interpreter
change. The four original protocols, trial bodies, packages and grades remain
unchanged. A pending producer must have its registered process live now.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import runner_runtime_continuation as memory
import kvmem_capacity_continuation as capacity
import retired_051_successors as old
import thinking_budget_replay_queue as thinking
import feedback_retest_queue as feedback

owned, base, image, sdk = old.owned, old.base, old.image, old.sdk
read, save, SafetyHalt, ROOT = old.read, old.save, old.SafetyHalt, memory.ROOT
MODULE = "runner_runtime_successors"
HELPERS = ("runner_runtime_successors.py", "tests/test_runner_runtime_successors.py")
STAGES = ("sdk", "image", "thinking", "feedback")
COUNTS = dict(sdk=10, image=12, thinking=4, feedback=8)
CONTROLLER_PYTHON = ROOT / ".venv/bin/python"
FROZEN_VALIDATORS = dict(sdk=sdk.validate, image=old.IMAGE_VALIDATE,
                         thinking=thinking.validate, feedback=feedback.validate)


def frozen_protocol(stage, path):
    # Nested validators continue to read their historical static registrations.
    with patch.object(image, "validate", FROZEN_VALIDATORS["image"]), \
            patch.object(thinking, "validate", FROZEN_VALIDATORS["thinking"]), \
            patch.object(feedback, "validate", FROZEN_VALIDATORS["feedback"]):
        return FROZEN_VALIDATORS[stage](Path(path))


def qualification(path):
    q = read(path)
    if (q.get("kind") != "runner_runtime_successors_cpu_qualification" or q.get("passed") is not True
            or q.get("tests_run", 0) < 20 or q.get("model_calls") != 0 or q.get("device_inputs") != 0):
        raise SafetyHalt("Require completed runtime successor admission qualification")
    for name in HELPERS:
        image.pinned(ROOT / "apps/computer_use" / name, q["source_sha256"][name])


def memory_registration(path, continuation, runtime):
    r = read(path)
    is_capacity = runtime.get("kind") == "exact_kvmem_capacity_failure_continuation"
    producer = capacity if is_capacity else memory
    kind = "actual_kvmem_capacity_continuation_identity" if is_capacity else "actual_runner_runtime_memory_continuation_identity"
    command = [runtime["python"], "-m", Path(producer.__file__).stem, "--declaration", str(continuation)]
    if (r.get("kind") != kind
            or r.get("declaration") != str(continuation)
            or r.get("declaration_sha256") != owned.digest(continuation)
            or r.get("protocol") != runtime["continued_protocol"]
            or r.get("protocol_sha256") != owned.digest(runtime["continued_protocol"])
            or r.get("queue") != str(Path(runtime["state_dir"]) / "queue.json")
            or r["identity"].get("command") != command
            or r["identity"].get("cwd") != str(ROOT) or r["identity"].get("uid") != os.getuid()):
        raise SafetyHalt("Actual memory runtime producer registration changed")
    for name in producer.HELPERS:
        image.pinned(ROOT / "apps/computer_use" / name, r["source_sha256"][name])
    return r


def memory_declaration(path):
    kind = read(path).get("kind")
    if kind == "pre_vm_cssselect_runner_runtime_continuation": return memory.validate(path)
    if kind == "exact_kvmem_capacity_failure_continuation": return capacity.validate(path)
    raise SafetyHalt("Unknown frozen memory continuation scope")


def memory_queue(r, continuation):
    is_capacity = r.get("kind") == "actual_kvmem_capacity_continuation_identity"
    key = "capacity_continuation_sha256" if is_capacity else "runner_runtime_continuation_sha256"
    wrapper_key = "capacity_continuation_wrapper_sha256" if is_capacity else "continued_successor_wrapper_sha256"
    producer = capacity if is_capacity else memory
    q = old.guard_queue(read(r["queue"]), r["identity"], owned.digest(continuation),
                        key)
    if (q.get("protocol_sha256") != r["protocol_sha256"]
            or q.get(wrapper_key) != owned.digest(Path(producer.__file__))
            or q.get("qualified_runner_python") != str(memory.PYTHON)):
        raise SafetyHalt("Memory queue is not the qualified frozen runtime continuation")
    return q


def successor_registration(path, stage, declaration, d):
    r = read(path)
    command = [d["controller_python"], "-m", MODULE, "--declaration", str(declaration), "--stage", stage]
    if (r.get("kind") != "actual_runner_runtime_successor_identity" or r.get("stage") != stage
            or r.get("declaration") != str(declaration) or r.get("declaration_sha256") != owned.digest(declaration)
            or r.get("wrapper_sha256") != owned.digest(Path(__file__))
            or r.get("protocol") != d[stage + "_protocol"]
            or r.get("protocol_sha256") != owned.digest(d[stage + "_protocol"])
            or r.get("queue") != str(Path(d[stage + "_state_dir"]) / "queue.json")
            or r["identity"].get("command") != command
            or r["identity"].get("cwd") != str(ROOT) or r["identity"].get("uid") != os.getuid()):
        raise SafetyHalt("Actual successor registration is not this declared stage")
    return r


def successor_queue(r, declaration):
    q = old.guard_queue(read(r["queue"]), r["identity"], owned.digest(declaration),
                        "runtime_successor_declaration_sha256")
    if (q.get("protocol_sha256") != r["protocol_sha256"]
            or q.get("runtime_successor_wrapper_sha256") != owned.digest(Path(__file__))):
        raise SafetyHalt("Successor producer source or frozen protocol changed")
    return q


def validate(path):
    d = read(path)
    if (d.get("kind") != "qualified_runner_runtime_frozen_successors"
            or d.get("original_protocols_changed") is not False
            or d.get("original_failures_overwritten") is not False
            or d.get("original_unknowns_replayed") is not False
            or d.get("controller_python") != str(CONTROLLER_PYTHON)
            or d.get("runner_python") != str(memory.PYTHON) or d.get("counts") != COUNTS):
        raise SafetyHalt("Use the exact declared runtime successor scope")
    for item in d["files"]:
        image.pinned(item["path"], item["sha256"])
    qualification(d["qualification"])
    image.pinned(CONTROLLER_PYTHON, d["controller_binary_sha256"])
    image.pinned(memory.PYTHON, d["runner_binary_sha256"])
    runtime, canonical = memory_declaration(Path(d["memory_continuation"]))
    mr = memory_registration(Path(d["memory_registration"]), Path(d["memory_continuation"]), runtime)
    memory_queue(mr, Path(d["memory_continuation"]))
    if (d["original_followup"] != runtime["original_followup"]
            or d["memory_protocol"] != runtime["continued_protocol"]
            or d["memory_queue"] != mr["queue"]
            or d["sdk_protocol"] != canonical["sdk_protocol"]
            or d["image_protocol"] != canonical["image_protocol"]):
        raise SafetyHalt("Frozen predecessor protocols or fresh evidence routing changed")
    for stage in STAGES:
        frozen_protocol(stage, d[stage + "_protocol"])
    return d, canonical, mr


def prepare(args):
    continuation, registration = args.memory_continuation.resolve(), args.memory_registration.resolve()
    runtime, canonical = memory_declaration(continuation)
    mr = memory_registration(registration, continuation, runtime)
    memory_queue(mr, continuation)
    qualification(args.qualification)
    protocols = dict(sdk=Path(canonical["sdk_protocol"]), image=Path(canonical["image_protocol"]),
                     thinking=ROOT / "apps/computer_use/.runtime/thinking-budget-replay-20261005/protocol.json",
                     feedback=ROOT / "apps/computer_use/.runtime/literal-feedback-matched-retest-20261005/protocol.json")
    state_dirs = {}
    for stage, protocol in protocols.items():
        index = dict(sdk=4, image=3, thinking=2, feedback=2)[stage]
        while (protocol.parent / f"queue-runtime-{index:02d}").exists(): index += 1
        state_dirs[stage] = f"queue-runtime-{index:02d}"
        p = frozen_protocol(stage, protocol)
        if stage == "sdk": p = p[0]
        if len(p["trials"]) != COUNTS[stage]: raise SafetyHalt("Frozen trial count changed")
        if (protocol.parent / state_dirs[stage]).exists(): raise SafetyHalt("New controller state already exists")
        for row in p["trials"]:
            if (protocol.parent / "attempts" / row["id"]).exists():
                raise SafetyHalt("A frozen successor attempt has already been dispatched")
    # Keep the controller's jsonschema dependencies and the OSWorld runner's
    # cssselect/PyAV dependencies in their already installed environments.
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "apps/computer_use") + ":" + str(image.RUNTIME / "media-qualification"))
    checks = []
    for python, imports in ((str(CONTROLLER_PYTHON), "jsonschema,thinking_budget_replay_queue,feedback_retest_queue"),
                            (str(memory.PYTHON), "cssselect,lxml.cssselect,numpy,av,aiohttp,osworld_office_candidate_run,platform_pilots,receipt_model_pilot")):
        command = [python, "-c", "import json,sys;import " + imports + ";print(json.dumps(dict(executable=sys.executable,imports=" + repr(imports.split(",")) + ",model_calls=0,device_inputs=0)))"]
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=20, check=True)
        checks.append(json.loads(result.stdout))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / "runtime-qualification.json", dict(passed=True, checks=checks, model_calls=0, device_inputs=0))
    files = [continuation, registration, Path(runtime["original_followup"]), args.qualification.resolve(),
             out / "runtime-qualification.json", *protocols.values()]
    d = dict(kind="qualified_runner_runtime_frozen_successors", declared_at=time.time(), counts=COUNTS,
             memory_continuation=str(continuation), memory_registration=str(registration), memory_protocol=runtime["continued_protocol"],
             memory_queue=mr["queue"], original_followup=runtime["original_followup"],
             controller_python=str(CONTROLLER_PYTHON), runner_python=str(memory.PYTHON),
             controller_binary_sha256=owned.digest(CONTROLLER_PYTHON), runner_binary_sha256=owned.digest(memory.PYTHON),
             qualification=str(args.qualification.resolve()), original_protocols_changed=False,
             original_failures_overwritten=False, original_unknowns_replayed=False,
             files=[dict(path=str(p), sha256=owned.digest(p)) for p in files],
             scope="Original 10/12/4/8 frozen trials; actual resumed memory receipts; new controller states. Only SDK trial child Python changes to the original qualified OSWorld venv. No model, package, task, budget, grade or input gate changes.")
    for stage, protocol in protocols.items():
        d[stage + "_protocol"] = str(protocol)
        d[stage + "_state_dir"] = str(protocol.parent / state_dirs[stage])
        d[stage + "_registration"] = str(out / (stage + "-registration.json"))
    path = out / "successors.json"
    save(path, d); validate(path)
    return path


def metadata(state, d, path):
    state.update(runtime_successor_declaration_sha256=owned.digest(path),
                 runtime_successor_wrapper_sha256=owned.digest(Path(__file__)), qualified_runner_python=d["runner_python"],
                 continued_successor_declaration_sha256=owned.digest(d["original_followup"]))


def sdk_protocol(p, d, canonical, mr):
    effective = copy.deepcopy(p)
    effective.update(parent_osworld_queue=canonical["continued_osworld_queue"], parent_kvmem_queue=d["memory_queue"],
                     parent_kvmem_protocol=d["memory_protocol"],
                     predecessor_processes=dict(osworld=canonical["continuation_controller_identity"], memory=mr["identity"]))
    return effective


def run_sdk(args, d, canonical, mr):
    original_ready = base.predecessors_ready
    def checked(path):
        validate(args.declaration)
        p, kp = frozen_protocol("sdk", path)
        return sdk_protocol(p, d, canonical, mr), kp
    def ready(p, kp):
        memory_queue(mr, Path(d["memory_continuation"]))
        return original_ready(p, kp)
    class Controller(sdk.Controller):
        def __init__(self, *values):
            super().__init__(*values); metadata(self.state, d, args.declaration)
        def trial_command(self, row, output):
            command = super().trial_command(row, output)
            if command[:2] != [sys.executable, "-m"]: raise SafetyHalt("Frozen SDK runner command shape changed")
            return [d["runner_python"], *command[1:]]
    sdk.PROTOCOL, _ = checked(args.protocol); sdk.DECLARATION = read(sdk.PROTOCOL["quarantine_declaration"])
    with patch.object(base, "validate", checked), patch.object(base, "predecessors_ready", ready), \
            patch.object(base, "Controller", Controller), patch.object(base, "ledger_terminal", sdk.scoped_ledger), \
            patch.object(owned, "terminal_parent", lambda p, q: old.terminal_parent(p, q, Path(d["original_followup"]))):
        return base.run(args)


def parent_registration(stage, d, args):
    path = Path(d[stage + "_registration"])
    r = successor_registration(path, stage, args.declaration, d)
    return r, owned.digest(path)


def run_image(args, d, canonical, mr):
    sr, sha = parent_registration("sdk", d, args)
    def checked(path):
        validate(args.declaration); image.pinned(d["sdk_registration"], sha)
        p = frozen_protocol("image", path)
        p.update(osworld_queue=canonical["continued_osworld_queue"], memory_queue=d["memory_queue"],
                 sdk_queue=sr["queue"], memory_protocol=d["memory_protocol"])
        return p
    def ready(p):
        memory_queue(mr, Path(d["memory_continuation"]))
        image.pinned(d["sdk_registration"], sha); successor_queue(sr, args.declaration)
        original_guard = old.guard_queue
        def guard(state, identity, declaration_sha, key):
            if mr.get("kind") == "actual_kvmem_capacity_continuation_identity" and identity == mr["identity"]:
                return original_guard(state, identity, owned.digest(d["memory_continuation"]), "capacity_continuation_sha256")
            return original_guard(state, identity, declaration_sha, key)
        with patch.object(old, "guard_queue", guard):
            return old.image_ready(p, canonical, Path(d["original_followup"]), mr, sr)
    class Controller(image.Controller):
        def __init__(self, *values):
            super().__init__(*values); metadata(self.state, d, args.declaration)
            self.state["sdk_registration_sha256"] = sha
    with patch.object(image, "validate", checked), patch.object(image, "Controller", Controller), \
            patch.object(image, "predecessors_ready", ready):
        return image.run(args)


def run_thinking(args, d, canonical, mr):
    ir, sha = parent_registration("image", d, args)
    original_ready = thinking.predecessor_ready
    def checked(path):
        validate(args.declaration); image.pinned(d["image_registration"], sha)
        p = frozen_protocol("thinking", path)
        p.update(parent_queue=ir["queue"], parent_controller=ir["identity"], parent_continuation=None)
        return p
    def ready(p):
        image.pinned(d["image_registration"], sha); successor_queue(ir, args.declaration)
        return original_ready(p)
    class Controller(image.Controller):
        def __init__(self, *values):
            super().__init__(*values); metadata(self.state, d, args.declaration)
            self.state["image_registration_sha256"] = sha
    with patch.object(thinking, "validate", checked), patch.object(thinking, "predecessor_ready", ready), \
            patch.object(image, "Controller", Controller):
        return thinking.run(args)


def run_feedback(args, d, canonical, mr):
    tr, sha = parent_registration("thinking", d, args)
    original_ready = feedback.terminal_predecessor
    def checked(path):
        validate(args.declaration); image.pinned(d["thinking_registration"], sha)
        p = frozen_protocol("feedback", path)
        p.update(parent_queue=tr["queue"], parent_controller=tr["identity"])
        return p
    def ready(p, q):
        image.pinned(d["thinking_registration"], sha); successor_queue(tr, args.declaration)
        return original_ready(p, q)
    class Controller(feedback.Controller):
        def __init__(self, *values):
            super().__init__(*values); metadata(self.state, d, args.declaration)
            self.state["thinking_registration_sha256"] = sha
    with patch.object(feedback, "validate", checked), patch.object(feedback, "terminal_predecessor", ready), \
            patch.object(feedback, "Controller", Controller):
        return feedback.run(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true"); p.add_argument("--output", type=Path)
    p.add_argument("--memory-continuation", type=Path); p.add_argument("--memory-registration", type=Path)
    p.add_argument("--qualification", type=Path); p.add_argument("--declaration", type=Path)
    p.add_argument("--stage", choices=STAGES)
    p.add_argument("--wait-seconds", type=int, default=172800)
    p.add_argument("--resource-wait-seconds", type=int, default=21600); p.add_argument("--min-free-mib", type=int, default=56000)
    args = p.parse_args()
    if not (60 <= args.wait_seconds <= 172800 and 60 <= args.resource_wait_seconds <= 86400 and args.min_free_mib >= 56000):
        p.error("Keep the original bounded shared-resource profile")
    if args.prepare:
        if any(getattr(args, name) is None for name in ("output", "memory_continuation", "memory_registration", "qualification")):
            p.error("Provide actual runtime producer and CPU qualification")
        print(json.dumps(dict(declaration=str(prepare(args)), model_calls=0, device_inputs=0))); return
    if args.declaration is None or args.stage is None: p.error("Provide the immutable declaration and stage")
    args.declaration = args.declaration.resolve()
    d, canonical, mr = validate(args.declaration)
    if sys.executable != d["controller_python"]: raise SafetyHalt("Keep the qualified controller environment")
    args.protocol, args.state_dir = Path(d[args.stage + "_protocol"]), Path(d[args.stage + "_state_dir"])
    def interrupted(*_): raise SafetyHalt("Runtime successor interrupted; live work and evidence preserved")
    signal.signal(signal.SIGTERM, interrupted)
    result = {"sdk": run_sdk, "image": run_image, "thinking": run_thinking, "feedback": run_feedback}[args.stage](args, d, canonical, mr)
    print(json.dumps(dict(status=result["status"], trials=len(result["trials"]))))


if __name__ == "__main__": main()
