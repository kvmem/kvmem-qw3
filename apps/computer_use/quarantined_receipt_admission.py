"""Distinguish one frozen, reboot-isolated receipt from current unknown input.

This never updates a receipt or authorizes replay. The exception requires the
exact original row, failed task, recorded process retirement, changed guest
boot and gateway session, and the same currently paused owned device instance.
Every other pending/unknown row still blocks service handoff.
"""
from collections import Counter
import contextlib
import hashlib
import json
from pathlib import Path
import sqlite3

from kvmem_osworld_queue import SafetyHalt, digest


def checked_file(reference):
    path = Path(reference["path"])
    if digest(path) != reference["sha256"]:
        raise SafetyHalt("Historical quarantine evidence changed")
    return json.loads(path.read_text())


def validate_evidence(declaration):
    reset = checked_file(declaration["reset_evidence"])
    result = checked_file(declaration["original_result"])
    expected_true = ("passed", "old_app_stop_confirmed", "old_gateway_stop_confirmed", "old_unknown_receipt_preserved")
    expected_false = ("original_failure_regraded", "old_uncertain_action_replayed", "old_unknown_reclassified_as_completed")
    if (reset.get("kind") != "administrative_quarantine_and_owned_test_vm_reset"
            or reset.get("status") != "quarantined_old_task_and_new_session_ready"
            or any(reset.get(key) is not True for key in expected_true)
            or any(reset.get(key) is not False for key in expected_false)
            or not reset.get("finished_at")
            or not reset.get("old_guest_boot_id") or not reset.get("new_guest_boot_id")
            or reset["old_guest_boot_id"] == reset["new_guest_boot_id"]
            or not reset.get("old_gateway_session") or not reset.get("new_gateway_session")
            or reset["old_gateway_session"] == reset["new_gateway_session"]
            or type(result.get("uncertain_actions")) is not int or result["uncertain_actions"] != 1
            or result.get("accepted") is not False or result.get("artifact_passed") is not False):
        raise SafetyHalt("Historical task was not independently isolated and retained as failed")
    row = declaration["row"]
    remaining = reset.get("old_task", {}).get("receipt_reconciliation", {}).get("remaining", [])
    if (row["id"] != reset["reserved_receipt_id"] or row["status"] != "uncertain"
            or row["result_is_null"] is not True
            or row["desktop_id"] != reset["old_gateway"]["device_id"] + ":" + reset["old_gateway_session"]
            or not any(item.get("receipt_id") == row["id"] for item in remaining)
            or declaration["gateway_runtime"] != str(Path(reset["old_gateway"]["command"][
                reset["old_gateway"]["command"].index("--runtime") + 1]).resolve())):
        raise SafetyHalt("Receipt does not belong to the frozen isolated task")
    return reset


def admit_rows(rows, declaration, context):
    reset = validate_evidence(declaration)
    if (context.get("paused") is not True or context.get("container_running") is not True
            or context.get("container_owner") != "cross-device-20261003"
            or context.get("container_id") != reset["container_id"]
            or context.get("guest_boot_id") != reset["new_guest_boot_id"]
            or context.get("gateway_pid") != reset["new_gateway_pid"]
            or context.get("session_id") != reset["new_gateway_session"]
            or context.get("desktop_id") != reset["old_gateway"]["device_id"] + ":" + reset["new_gateway_session"]):
        raise SafetyHalt("Current paused device does not match the proven post-quarantine instance")
    counts = Counter()
    isolated = 0
    expected = declaration["row"]
    for action_id, request, status, result in rows:
        counts[status] += 1
        if status in {"completed", "rejected"}:
            if action_id == expected["id"]:
                raise SafetyHalt("Historical unknown receipt was reclassified")
            continue
        if (action_id != expected["id"] or status != "uncertain" or result is not None
                or hashlib.sha256(request.encode()).hexdigest() != expected["request_sha256"]
                or json.loads(request).get("desktop_id") != expected["desktop_id"]):
            raise SafetyHalt("Current or undeclared pending input blocks handoff")
        isolated += 1
    if isolated != 1:
        raise SafetyHalt("The frozen historical unknown was removed or duplicated")
    return dict(counts, current_unknowns=0, quarantined_unknowns_preserved=isolated)


def ledger_rows(runtime):
    path = (Path(runtime) / "receipts.sqlite").resolve()
    with contextlib.closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        return db.execute("SELECT id,request,status,result FROM actions").fetchall()
