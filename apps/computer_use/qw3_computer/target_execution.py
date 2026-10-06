"""Bounded application transition using the existing durable action boundary.

Application foreground verification is intentionally narrower than window,
control focus or business completion. Unsupported operations remain explicit.
"""
import asyncio
import json
import re
import time

from .desktop import DesktopError, DesktopRejected, DesktopValidationError
from .desktop_contract import capability_manifest, target_facts


class TargetExecution:
    def __init__(self, desktop):
        self.desktop = desktop
        desktop.db.execute("CREATE TABLE IF NOT EXISTS target_transitions (task TEXT,id TEXT,request TEXT,record TEXT,PRIMARY KEY(task,id))")
        desktop.db.commit()

    def save(self, record):
        self.desktop.db.execute("INSERT OR REPLACE INTO target_transitions VALUES (?,?,?,?)",
            (record["task_id"], record["transition_id"], json.dumps(record["request"], sort_keys=True), json.dumps(record)))
        self.desktop.db.commit()

    def result(self, record, observation, duplicate=False):
        return {k: v for k, v in record.items() if k not in ("request", "task_id")} | {
            "duplicate": duplicate, "observation": observation,
            "target_facts": target_facts(observation), "observe_before_input": observation.get("state_only") is True,
            "verification_scope": "Foreground application and current driver task scope only; window selection, control focus and business completion are not verified",
        }

    async def invoke(self, arguments):
        allowed = {"operation", "app", "transition_id", "timeout_seconds"}
        if (not isinstance(arguments, dict) or set(arguments) - allowed
                or not {"operation", "app", "transition_id"} <= set(arguments)
                or arguments["operation"] != "ensure_app"):
            raise DesktopError("target supports only operation=ensure_app, app, transition_id and optional timeout_seconds")
        identifier = arguments["transition_id"]
        seconds = arguments.get("timeout_seconds", 8)
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identifier):
            raise DesktopError("Invalid transition_id")
        if type(seconds) is not int or not 1 <= seconds <= 12:
            raise DesktopError("Target wait must be an integer from 1 to 12 seconds")
        if self.desktop.paused:
            raise DesktopError("Desktop is paused")
        if self.desktop.unresolved():
            raise DesktopError("Uncertain input requires paused reconciliation before a target transition")
        request = arguments | {"timeout_seconds": seconds}
        row = self.desktop.db.execute("SELECT request,record FROM target_transitions WHERE task=? AND id=?",
                                      (self.desktop.task_id, identifier)).fetchone()
        if row and json.loads(row[0]) != request:
            raise DesktopError("transition_id reused with different intent")
        record = json.loads(row[1]) if row else None
        started = time.time()
        try:
            observation = await asyncio.wait_for(self.desktop.observe(), seconds)
        except TimeoutError as exc:
            raise DesktopError("Target observation_timeout before input; no launch sent") from exc
        manifest = capability_manifest(observation.get("capabilities", {}))
        alias = arguments["app"]
        processes = manifest.get("app_processes", {}).get(alias, []) if isinstance(alias, str) else []
        feature = manifest["features"]["ensure_app"]
        if (alias not in manifest.get("open_apps", []) or not processes or not feature["enabled_for_task"]
                or feature["available_now"] is False):
            raise DesktopError("Application transition is unavailable on this device")
        duplicate = record is not None
        if record:
            # A saved transition never sends a second launch, even after a crash.
            if record["status"] in {"dispatching", "delivery_uncertain"}:
                record.update(status="needs_reconciliation", verified=False, failure_category="delivery_uncertain")
                self.save(record)
                return self.result(record, observation, True)
            if record["status"] in {"timed_out", "rejected", "session_changed", "permission_required", "interrupted", "verified", "state_changed"}:
                record["currently_verified"] = self.matches(observation, processes)
                record["historical_verified"] = record["verified"]
                record["verified"] = record["verified"] and record["currently_verified"]
                if record["status"] == "verified" and not record["verified"]:
                    record.update(status="state_changed", failure_category="focus_lost")
                return self.result(record, observation, True)
        else:
            record = {"task_id": self.desktop.task_id, "transition_id": identifier, "request": request,
                      "status": "reserved", "created_at": started, "deadline": started + seconds,
                      "desktop_id": observation["desktop_id"], "capability_version": manifest["capability_version"],
                      "input_sent": False, "verified": False, "polls": 0}
            self.save(record)
        try:
            if self.matches(observation, processes):
                record.update(status="verified", verified=True, completed_at=time.time())
                self.save(record)
                return self.result(record, observation, duplicate)
            if not duplicate:
                remaining = record["deadline"] - time.time()
                if remaining <= 0:
                    record.update(status="timed_out", failure_category="pre_dispatch_timeout")
                    self.save(record)
                    return self.result(record, observation)
                record.update(status="dispatching")
                self.save(record)
                action = {"desktop_id": observation["desktop_id"], "observation_id": observation["observation_id"],
                          "action_id": "target_" + identifier, "action": {"kind": "open_app", "app": alias}}
                execution = self.desktop.execution
                if execution and not any(s["status"] == "pending" for s in execution.steps()):
                    action["step"] = {"goal": "Bring the explicitly selected application forward",
                                      "conditions": [{"kind": "foreground_app", "text": alias}]}
                receipt_only = manifest.get("receipt_only_dispatch") is True
                delivered = await asyncio.wait_for(self.desktop.act(action, capture_after_input=not receipt_only), remaining)
                if not receipt_only:
                    observation = delivered["observation"]
                record.update(status="waiting", input_sent=True, receipt_id=delivered["receipt_id"])
                record["observation_is_post_input"] = not receipt_only
                self.save(record)
            while True:
                if self.desktop.paused:
                    record.update(status="interrupted", failure_category="paused")
                    break
                if observation["desktop_id"] != record["desktop_id"]:
                    record.update(status="session_changed", failure_category="session_changed")
                    break
                current = capability_manifest(observation.get("capabilities", {}))
                if current["capability_version"] != record["capability_version"]:
                    record.update(status="permission_required", failure_category="capability_changed")
                    break
                if record.get("observation_is_post_input", True) and self.matches(observation, processes):
                    record.update(status="verified", verified=True, completed_at=time.time())
                    break
                remaining = record["deadline"] - time.time()
                if remaining <= 0:
                    record.update(status="timed_out", failure_category="focus_not_confirmed")
                    break
                await asyncio.sleep(min(.2, remaining))
                # Each observation is bounded by the original deadline too.
                try:
                    reader = self.desktop.state if manifest.get("lightweight_state") is True else self.desktop.observe
                    observation = await asyncio.wait_for(reader(), max(.001, record["deadline"] - time.time()))
                    record["observation_is_post_input"] = True
                except TimeoutError:
                    record.update(status="timed_out", failure_category="observation_timeout")
                    break
                record["polls"] += 1
            self.save(record)
            return self.result(record, observation, duplicate)
        except (DesktopRejected, DesktopValidationError) as exc:
            record.update(status="rejected", failure_category="target_rejected", error=str(exc))
            self.save(record)
            raise
        except BaseException:
            if record["status"] == "dispatching":
                record["input_sent"] = None
            record.update(status="delivery_uncertain" if record["status"] == "dispatching" else "interrupted",
                          failure_category="delivery_uncertain" if record["status"] == "dispatching" else "transition_interrupted")
            self.save(record)
            raise

    @staticmethod
    def matches(observation, processes):
        return (observation.get("foreground", {}).get("process") in processes
                and observation.get("foreground_in_scope") is not False
                and observation.get("accessibility", {}).get("status") != "outside_task_scope")
