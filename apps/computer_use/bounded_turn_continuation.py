"""Bounded recovery for an unchecked turn that ends without using a tool.

Installed only in an isolated candidate package. This requests another model
turn in the same task; it never replays an input or verifies business success.
"""
import asyncio
import contextlib
import json
import math
import time


POLICY = {"version": 1, "max_additional_turns": 2, "max_task_seconds": 1800, "max_delivery_seconds": 30,
          "only_zero_tool_turns": True, "reset_budget_or_recovery": False}


def checkpoint(owner, used, *, now=None):
    return {"policy": dict(POLICY), "owner": owner,
            "deadline_monotonic": (time.monotonic() if now is None else now) + POLICY["max_task_seconds"],
            "turn_start_tools": used, "attempts": []}


def eligibility(state, *, owner, origin, ready, paused, unresolved, workflow, software, now=None):
    """Explain admission using explicit task state, never assistant prose."""
    if origin not in {"starting", "running"} or state.get("status") != "paused":
        return "not_an_active_turn_ending_unchecked"
    if not ready or not paused or unresolved:
        return "device_or_runtime_not_reconciled"
    if not workflow or software:
        return "not_verified_gui_mode"
    if state.get("turn_status") != "completed" or state.get("turn_error") or state.get("evidence_error"):
        return "turn_or_evidence_error"
    if state.get("takeover") or state.get("bounded_turn_user_control"):
        return "explicit_control_or_takeover"
    if state.get("gui_recovery", {}).get("paused_reason"):
        return "bounded_recovery_already_paused"
    if state.get("completion_origin") or state.get("current_action"):
        return "completion_or_input_already_pending"
    if any(not isinstance(state.get(k), str) or not state[k] for k in ("id", "thread_id", "turn_id")):
        return "missing_task_or_turn_identity"
    used, limit = state.get("tool_calls_used"), state.get("tool_call_limit")
    if type(used) is not int or type(limit) is not int or not 0 <= used < limit <= 10000:
        return "invalid_or_exhausted_cumulative_budget"
    saved = state.get("bounded_turn_continuation")
    if not isinstance(saved, dict) or saved.get("policy") != POLICY or saved.get("owner") != owner:
        return "checkpoint_missing_changed_or_restarted"
    deadline = saved.get("deadline_monotonic")
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        return "invalid_task_deadline"
    if (time.monotonic() if now is None else now) >= deadline:
        return "task_deadline_exhausted"
    start = saved.get("turn_start_tools")
    if type(start) is not int or start != used:
        return "turn_already_used_tools"
    attempts = saved.get("attempts")
    if not isinstance(attempts, list) or len(attempts) >= POLICY["max_additional_turns"]:
        return "additional_turn_limit"
    if any(not isinstance(a, dict) or a.get("delivery") != "accepted" for a in attempts):
        return "previous_turn_delivery_not_confirmed"
    if any(a.get("completed_turn_id") == state["turn_id"] for a in attempts):
        return "completed_turn_already_reserved"
    if any(not isinstance(f, dict) or f.get("delivery") != "accepted" for f in state.get("followups", [])):
        return "external_followup_delivery_not_confirmed"
    return "eligible"


class BoundedTurnContinuation:
    async def continue_unchecked_zero_tool_turn(self, origin):
        from .desktop import DesktopError, model_observation

        async with self.control_lock:
            reason = eligibility(self.state, owner=self.bounded_turn_owner, origin=origin, ready=self.ready,
                                 paused=self.desktop.paused, unresolved=self.desktop.unresolved(),
                                 workflow=self.workflow is not None, software=self.software is not None)
            self.state["bounded_turn_last_decision"] = {"reason": reason, "turn_id": self.state.get("turn_id")}
            if reason != "eligible":
                self.save()
                return False
            try:
                self.require_saved_inference_mode()
            except DesktopError:
                self.state["bounded_turn_last_decision"]["reason"] = "inference_or_device_checkpoint_changed"
                self.save()
                return False

            saved = self.state["bounded_turn_continuation"]
            receipt = {"completed_turn_id": self.state["turn_id"], "thread_id": self.state["thread_id"],
                       "tool_calls_used": self.state["tool_calls_used"], "delivery": "reserved",
                       "reserved_at": time.time()}
            saved["attempts"].append(receipt)
            self.state.update(status="starting", draft="", notice="任务尚未完成，正在继续核对并操作。")
            for key in ("finished_at", "turn_id", "turn_status", "turn_error"):
                self.state.pop(key, None)
            self.save()  # Reserve before any await; lost replies cannot grant a retry.
            try:
                seconds = min(POLICY["max_delivery_seconds"], saved["deadline_monotonic"] - time.monotonic())
                async with asyncio.timeout(seconds):
                    await self.desktop.control("resume")
                    observation = await self.desktop.observe()
                    if self.desktop.unresolved():
                        raise DesktopError("Unresolved action appeared during continuation admission")
                    if time.monotonic() >= saved["deadline_monotonic"]:
                        raise DesktopError("Existing task deadline expired")
                    self.desktop.task_id = self.state["id"]
                    text = ("The same task is still unchecked. Your preceding turn ended without using a tool. "
                            "Continue the original user task using its existing history and remaining budget. "
                            "Do not repeat any uncertain input or reset checked/pending steps. If necessary user "
                            "information or authorization is missing, request workflow takeover. Completion "
                            "requires the existing visible conditions; narration alone does not complete it.\n"
                            "Current desktop observation:\n" + json.dumps(model_observation(observation), ensure_ascii=False)
                            + "\nDurable checked/pending steps:\n" + json.dumps(self.workflow.status(), ensure_ascii=False))
                    receipt["delivery"] = "requesting"
                    self.save()
                    turn = await self.client.turn(self.state["thread_id"], text,
                                                  ["data:image/png;base64," + observation["image"]])
                if not isinstance(turn, str) or not turn or turn == receipt["completed_turn_id"]:
                    raise DesktopError("Continuation did not acknowledge a new turn")
                self.state["turn_id"] = turn
                receipt.update(delivery="accepted", turn_id=turn)
                saved["turn_start_tools"] = self.state["tool_calls_used"]
                self.state["status"] = "running"
                self.save()
                return True
            except BaseException as exc:
                receipt["delivery"] = "uncertain" if receipt["delivery"] == "requesting" else "not_requested"
                self.state.update(status="paused", notice="续轮未能确认，已暂停并保留进度。")
                self.save()
                with contextlib.suppress(DesktopError):
                    await self.desktop.control("pause")
                if receipt["delivery"] == "uncertain":
                    # The app-server may have started an unidentified turn.
                    # Its owned client must stop; a paused device alone does
                    # not establish that the model request has retired.
                    self.ready = False
                    receipt["runtime_close_attempted"] = True
                    self.save()
                    with contextlib.suppress(Exception):
                        await self.client.close()
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    raise
                return False
