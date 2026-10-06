"""Device-independent steps and evidence; never sends input or reads graders."""
import asyncio
import hashlib
import json
import re
import sqlite3
import time

from .desktop import DesktopError, DesktopValidationError


CHECK_KINDS = {"control_text", "control_absent", "control_value", "foreground_title", "foreground_app", "screen_changed"}


def validate_checks(checks):
    if not isinstance(checks, list) or not 1 <= len(checks) <= 12:
        raise DesktopError("conditions must contain 1..12 explicit observable conditions")
    for check in checks:
        if not isinstance(check, dict) or set(check) - {"kind", "text", "value", "control_kind"} or check.get("kind") not in CHECK_KINDS:
            raise DesktopError("Unsupported condition; use a documented observable condition")
        required = {"kind"} if check["kind"] == "screen_changed" else {"kind", "text"}
        if check["kind"] == "control_value":
            required.add("value")
            if "control_kind" in check:
                required.add("control_kind")
        if set(check) != required or any(not isinstance(check[k], str) or not 1 <= len(check[k]) <= 512 for k in required - {"kind"}):
            raise DesktopError("Condition fields are missing, empty or unsupported")
    return checks


def check_observation(checks, observation, before):
    """Absence requires a complete accessibility read, not an API failure."""
    accessibility = observation.get("accessibility", {})
    controls = accessibility.get("controls", [])
    available = accessibility.get("status") == "ok"
    complete = available and accessibility.get("truncated") is False
    front = observation.get("foreground", {})
    evidence = []
    for check in checks:
        kind, text = check["kind"], check.get("text", "")
        matched = [c for c in controls if text in str(c.get("name", ""))]
        if kind == "control_text":
            passed = available and bool(matched)
        elif kind == "control_absent":
            passed = complete and not matched
        elif kind == "control_value":
            # Exact value, exact unique control label: never choose a fuzzy match.
            exact = [c for c in controls if c.get("name") == text
                     and ("control_kind" not in check or c.get("kind") == check["control_kind"])]
            passed = available and len(exact) == 1 and exact[0].get("value") == check["value"]
        elif kind == "foreground_title":
            passed = bool(front.get("title")) and text in front["title"]
        elif kind == "foreground_app":
            allowed = observation.get("capabilities", {}).get("app_processes", {}).get(text, [text])
            passed = front.get("process") in allowed
        else:
            original = before.get("progress_sha256", before.get("sha256"))
            current = observation.get("progress_sha256", observation.get("sha256"))
            passed = bool(original and current and current != original)
        item = {"condition": check, "passed": bool(passed), "observation_id": observation.get("observation_id"),
                "basis": "visible_observation", "matching_controls": len(exact) if kind == "control_value"
                else len(matched) if kind.startswith("control_") else None}
        if kind == "control_value" and len(exact) > 1:
            item.update(matching_control_kinds=sorted({str(c.get("kind", "")) for c in exact}),
                hint="The exact label is ambiguous. Set control_kind to the intended field's observed kind; multiple fields of that kind still cannot be verified.")
        evidence.append(item)
    return evidence


def state_fingerprint(observation):
    # Observation UUIDs and the desktop clock must not reset a repeated intent.
    state = [observation.get("desktop_id"), observation.get("foreground"),
             observation.get("progress_sha256", observation.get("sha256"))]
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def intent_fingerprint(action, observation):
    action = dict(action)
    if action.get("kind") == "click_control":
        controls = observation.get("accessibility", {}).get("controls", [])
        control = next((c for c in controls if c.get("control_id") == action.get("control_id")), None)
        if control:
            action = {"kind": "click", "x": control["center_x"], "y": control["center_y"], "button": "left", "clicks": 1}
    elif action.get('kind') in {'activate_window','focus_control'}:
        window=action['kind']=='activate_window'
        candidates=observation.get('windows',[]) if window else observation.get('accessibility',{}).get('controls',[])
        control=next((c for c in candidates if c.get('window_ref' if window else 'native_ref')==action.get('window_ref' if window else 'control_ref')),None)
        if control and control.get('identity' if window else 'native_identity'):
            action={'kind':action['kind'],'target_identity':control['identity' if window else 'native_identity']}
    return hashlib.sha256(json.dumps(action, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Execution:
    def __init__(self, desktop, runtime):
        self.desktop = desktop
        self.db = sqlite3.connect(runtime / "execution.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS steps(task TEXT,id TEXT,data TEXT,PRIMARY KEY(task,id))")
        self.db.execute("CREATE TABLE IF NOT EXISTS intents(task TEXT,receipt TEXT PRIMARY KEY,state TEXT,intent TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS completions(task TEXT PRIMARY KEY,data TEXT)")
        self.db.commit()
        self.lock = asyncio.Lock()

    @property
    def task(self):
        return self.desktop.task_id

    def steps(self):
        return [json.loads(row[0]) for row in self.db.execute("SELECT data FROM steps WHERE task=? ORDER BY rowid", (self.task,))]

    def save_step(self, step):
        self.db.execute("INSERT OR REPLACE INTO steps VALUES (?,?,?)", (self.task, step["step_id"], json.dumps(step, ensure_ascii=False)))
        self.db.commit()

    def completion(self, observation=None):
        row = self.db.execute("SELECT data FROM completions WHERE task=?", (self.task,)).fetchone()
        result = json.loads(row[0]) if row else None
        if result and observation is not None:
            checks = [e["condition"] for e in result["conditions"]]
            evidence = check_observation(checks, observation, {})
            result = result | {"verified": all(e["passed"] for e in evidence), "final_recheck": evidence}
        return result

    def status(self):
        return {"task_id": self.task, "steps": self.steps(), "completion": self.completion(),
                "unresolved_actions": self.desktop.unresolved(), "recovery": "Reconcile durable receipts before resuming; never replay an uncertain input."}

    def before_dispatch(self, arguments, dispatched, receipt_id):
        pending = [s for s in self.steps() if s["status"] == "pending"]
        definition = arguments.get("step")
        if definition is not None:
            if not isinstance(definition, dict) or set(definition) - {"goal", "conditions", "recovery_reason"} or not {"goal", "conditions"} <= set(definition):
                raise DesktopValidationError({"code":"invalid_step","path":"$.step","message":"step requires goal and conditions, with optional recovery_reason","hint":"Declare the observable result for this action."})
            try:
                validate_checks(definition["conditions"])
                if not isinstance(definition["goal"], str) or not 1 <= len(definition["goal"].strip()) <= 1024:
                    raise DesktopError("Step goal must contain 1..1024 characters")
            except DesktopError as exc:
                raise DesktopValidationError({"code":"invalid_step","path":"$.step","message":str(exc),"hint":"Use a nonempty set of documented observable conditions."}) from exc
            # The saved result criteria define continuity. Rewording an action
            # label must not abandon the plan or reset intent reservations.
            if pending and pending[0]["conditions"] != definition["conditions"]:
                reason = definition.get("recovery_reason")
                if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 1024 or self.desktop.unresolved():
                    raise DesktopValidationError({"code":"pending_step","path":"$.step.recovery_reason",
                        "message":"The previous step is still pending; it cannot be silently replaced.",
                        "hint":"Use workflow wait for this pending step. To continue toward the same result, omit step in the next act call. To replace it, declare the new step with a concrete recovery_reason. Never supersede an uncertain input.",
                        "checkpoint":{"pending_step":{k:pending[0].get(k) for k in
                            ("step_id","goal","conditions","status","last_observation_id")},
                            "uncertain_action_ids":[a["action_id"] for a in self.desktop.unresolved()],
                            "input_dispatched":False}})
                pending[0].update(status="abandoned",reason=reason,abandoned_at=time.time())
                self.save_step(pending[0]);pending=[]
            if not pending:
                # Bind to the exact model observation. No hidden capture or ID
                # rewriting between target selection and input dispatch.
                step = {"step_id":"step_"+receipt_id[3:27],"goal":definition["goal"],"conditions":definition["conditions"],
                        "status":"pending","started_at":time.time(),"origin":"inline_action",
                        "before":dict(self.desktop.latest)}
                self.save_step(step);pending=[step]
        if len(pending) != 1:
            raise DesktopValidationError({"code": "step_required", "path": "$.action",
                "message": "Begin one step with observable expected conditions before sending input.",
                "hint": "The previous step may already be verified. Supply step={goal:<expected result>,conditions:[...]} in this act call to begin the next step; a verified step does not need abandonment."})
        state = state_fingerprint(self.desktop.latest)
        intent = intent_fingerprint(dispatched["action"], self.desktop.latest)
        count = self.db.execute("SELECT count(*) FROM intents WHERE task=? AND state=? AND intent=?", (self.task, state, intent)).fetchone()[0]
        if count >= 2:
            raise DesktopValidationError({"code": "repeated_intent", "path": "$.action",
                "message": "The same input intent has already been reserved twice on this unchanged state; no input was sent.",
                "hint": "Wait for the step conditions, inspect focus or an obstructing dialog, or abandon this step with a reason and choose a different recovery. Changing action_id does not reset this guard."})
        self.db.execute("INSERT INTO intents VALUES (?,?,?,?)", (self.task, receipt_id, state, intent))
        self.db.execute("DELETE FROM completions WHERE task=?", (self.task,))
        self.db.commit()

    def after_delivery(self, observation):
        pending = next((s for s in self.steps() if s["status"] == "pending"), None)
        if not pending:
            return None
        evidence = check_observation(pending["conditions"], observation, pending["before"])
        passed = all(c["passed"] for c in evidence)
        pending.update(status="verified" if passed else "pending", evidence=evidence,
                       checked_at=time.time(), last_observation_id=observation["observation_id"])
        self.save_step(pending)
        return {"step_id":pending["step_id"],"verified":passed,"conditions":evidence}

    async def wait(self, step, seconds):
        deadline = time.monotonic() + seconds
        started = time.monotonic()
        count = 0
        while True:
            observation = await self.desktop.observe()
            count += 1
            evidence = check_observation(step["conditions"], observation, step["before"])
            passed = all(c["passed"] for c in evidence)
            step.update(status="verified" if passed else "pending", evidence=evidence,
                        checked_at=time.time(), last_observation_id=observation["observation_id"])
            self.save_step(step)
            if passed or time.monotonic() >= deadline or self.desktop.paused:
                return {"status": "verified" if passed else "waiting", "step_id": step["step_id"],
                        "verified": passed, "conditions": evidence, "polls": count,
                        "elapsed_seconds": time.monotonic() - started, "input_dispatched": False,
                        "observation": observation, "hint": None if passed else "Expected conditions are not established. Do not repeat a possibly delivered input; inspect or wait within the remaining task budget."}
            await asyncio.sleep(min(0.5, max(0, deadline - time.monotonic())))

    async def invoke(self, arguments):
        async with self.lock:
            if not isinstance(arguments, dict):
                raise DesktopError("workflow arguments must be an object")
            operation = arguments.get("operation")
            allowed = {"status": {"operation"}, "begin_step": {"operation", "step_id", "goal", "conditions"},
                       "wait": {"operation", "step_id", "timeout_seconds"},
                       "abandon_step": {"operation", "step_id", "reason"},
                       "finish": {"operation", "conditions", "timeout_seconds", "goal"},
                       "request_takeover": {"operation", "reason"}}
            required = allowed.get(operation, set()) - {"timeout_seconds"}
            if operation == "wait":
                required -= {"step_id"}
            if operation == "finish":
                required -= {"goal"}
            if operation not in allowed or set(arguments) - allowed[operation] or not required <= set(arguments):
                raise DesktopError("Unsupported workflow operation or fields")
            if operation == "status":
                return self.status()
            if operation == "request_takeover":
                reason = arguments["reason"]
                if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 1024:
                    raise DesktopError("A concrete takeover reason is required")
                return {"status": "takeover_requested", "reason": reason,
                        "checkpoint": self.status(), "input_dispatched": False}
            if self.desktop.paused:
                raise DesktopError("Device is paused; reconcile and resume before workflow changes")
            if "step_id" in arguments and (not isinstance(arguments["step_id"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", arguments["step_id"])):
                raise DesktopError("step_id must use 1..80 letters, digits, underscores or hyphens")
            if arguments.get("step_id") == "__final__":
                raise DesktopError("This step identity is reserved for final verification")
            seconds = arguments.get("timeout_seconds", 0)
            if type(seconds) is not int or not 0 <= seconds <= 20:
                raise DesktopError("timeout_seconds must be an integer from 0 to 20")
            steps = self.steps()
            if operation == "begin_step":
                validate_checks(arguments["conditions"])
                goal = arguments["goal"]
                if not isinstance(goal, str) or not 1 <= len(goal.strip()) <= 1024:
                    raise DesktopError("goal must describe the expected result in 1..1024 characters")
                prior = next((s for s in steps if s["step_id"] == arguments["step_id"]), None)
                if prior:
                    if prior["goal"] != goal or prior["conditions"] != arguments["conditions"]:
                        raise DesktopError("step_id reused with different conditions")
                    return prior
                if any(s["status"] == "pending" for s in steps):
                    raise DesktopError("Verify or explicitly abandon the pending step first")
                observation = await self.desktop.observe()
                step = {"step_id": arguments["step_id"], "goal": goal, "conditions": arguments["conditions"],
                        "status": "pending", "started_at": time.time(), "before": {k: v for k, v in observation.items() if k != "image"}}
                self.save_step(step)
                self.db.execute("DELETE FROM completions WHERE task=?", (self.task,))
                self.db.commit()
                return step | {"observation": observation}
            if operation == "finish":
                validate_checks(arguments["conditions"])
                if 'goal' in arguments and (not isinstance(arguments['goal'],str) or not 1<=len(arguments['goal'].strip())<=1024):
                    raise DesktopError('Optional final goal must contain 1..1024 characters')
                if not steps or any(s["status"] == "pending" for s in steps) or self.desktop.unresolved():
                    raise DesktopError("Finish requires checked steps, no pending step and no uncertain input")
                if not any(s["status"] == "verified" for s in steps):
                    raise DesktopError("At least one verified step is required")
                # A screen change is useful for a step, but is not a final artifact condition.
                if any(c["kind"] == "screen_changed" for c in arguments["conditions"]):
                    raise DesktopError("Final conditions must identify the requested result; screen_changed is insufficient")
                observation = await self.desktop.observe()
                final = {"step_id": "__final__", "goal": "Check declared final visible conditions",
                         "conditions": arguments["conditions"], "before": {k: v for k, v in observation.items() if k != "image"}, "status": "pending"}
                result = await self.wait(final, seconds)
                result["verification_scope"] = "Declared visible conditions only; independent business acceptance remains separate."
                stored = {k: v for k, v in result.items() if k != "observation"}
                self.db.execute("INSERT OR REPLACE INTO completions VALUES (?,?)", (self.task, json.dumps(stored)))
                # Final checks are not an input step and must not strand subsequent recovery.
                self.db.execute("DELETE FROM steps WHERE task=? AND id='__final__'", (self.task,))
                self.db.commit()
                return result
            if operation == "wait" and "step_id" not in arguments:
                pending = [s for s in steps if s["status"] == "pending"]
                if len(pending) != 1:
                    raise DesktopError("No unique pending step to wait for; inspect workflow status or supply step_id")
                step = pending[0]
            else:
                step = next((s for s in steps if s["step_id"] == arguments["step_id"]), None)
            if step is None:
                raise DesktopError("Unknown step_id; inspect workflow status")
            if operation == "abandon_step":
                reason = arguments["reason"]
                if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 1024:
                    raise DesktopError("A concrete abandonment/recovery reason is required")
                if step["status"] != "pending" or self.desktop.unresolved():
                    raise DesktopError("Only a pending step with no uncertain input can be abandoned")
                step.update(status="abandoned", reason=reason, abandoned_at=time.time())
                self.save_step(step)
                return step
            return await self.wait(step, seconds)

    def close(self):
        self.db.close()
