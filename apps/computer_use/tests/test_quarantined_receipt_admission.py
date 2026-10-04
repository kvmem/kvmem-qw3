import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import quarantined_receipt_admission as admission


class HistoricalReceiptTests(unittest.TestCase):
    def fixture(self, root):
        runtime = root / "gateway"
        reset = {"kind": "administrative_quarantine_and_owned_test_vm_reset",
                 "status": "quarantined_old_task_and_new_session_ready", "finished_at": 123,
                 "passed": True, "old_app_stop_confirmed": True, "old_gateway_stop_confirmed": True,
                 "old_unknown_receipt_preserved": True, "original_failure_regraded": False,
                 "old_uncertain_action_replayed": False, "old_unknown_reclassified_as_completed": False,
                 "old_guest_boot_id": "old-boot", "new_guest_boot_id": "new-boot",
                 "old_gateway_session": "old-session", "new_gateway_session": "new-session",
                 "reserved_receipt_id": "old-receipt", "container_id": "owned-container", "new_gateway_pid": 12,
                 "old_gateway": {"device_id": "linux-vm", "command": ["python", "--runtime", str(runtime)]},
                 "old_task": {"receipt_reconciliation": {"remaining": [{"receipt_id": "old-receipt"}]}}}
        request = json.dumps({"desktop_id": "linux-vm:old-session", "action": {"kind": "key", "key": "CTRL+W"}})
        reset_path = root / "reset.json"
        reset_path.write_text(json.dumps(reset))
        result_path = root / "original-result.json"
        result_path.write_text(json.dumps({"uncertain_actions": 1, "accepted": False, "artifact_passed": False}))
        declaration = {"gateway_runtime": str(runtime),
                       "reset_evidence": {"path": str(reset_path), "sha256": admission.digest(reset_path)},
                       "original_result": {"path": str(result_path), "sha256": admission.digest(result_path)},
                       "row": {"id": "old-receipt", "status": "uncertain", "result_is_null": True,
                               "request_sha256": hashlib.sha256(request.encode()).hexdigest(),
                               "desktop_id": "linux-vm:old-session"}}
        context = {"paused": True, "container_running": True, "container_owner": "cross-device-20261003",
                   "container_id": "owned-container", "guest_boot_id": "new-boot", "gateway_pid": 12,
                   "session_id": "new-session", "desktop_id": "linux-vm:new-session"}
        rows = [("complete", "{}", "completed", "{}"), ("old-receipt", request, "uncertain", None)]
        return declaration, context, rows

    def test_exact_retained_row_after_proven_reset_is_historical_only(self):
        with tempfile.TemporaryDirectory() as temp:
            declaration, context, rows = self.fixture(Path(temp))
            before = copy.deepcopy(rows)
            result = admission.admit_rows(rows, declaration, context)
            self.assertEqual(result["current_unknowns"], 0)
            self.assertEqual(result["uncertain"], 1)
            self.assertEqual(result["quarantined_unknowns_preserved"], 1)
            self.assertEqual(rows, before)

    def test_current_session_unknown_blocks_even_with_historical_exception(self):
        with tempfile.TemporaryDirectory() as temp:
            declaration, context, rows = self.fixture(Path(temp))
            rows.append(("current-lost", json.dumps({"desktop_id": context["desktop_id"]}), "uncertain", None))
            with self.assertRaises(admission.SafetyHalt):
                admission.admit_rows(rows, declaration, context)

    def test_undeclared_old_session_unknown_also_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            declaration, context, rows = self.fixture(Path(temp))
            rows.append(("another-old", rows[-1][1], "uncertain", None))
            with self.assertRaises(admission.SafetyHalt):
                admission.admit_rows(rows, declaration, context)

    def test_original_row_replay_mutation_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            declaration, context, rows = self.fixture(Path(temp))
            rows[-1] = ("old-receipt", rows[-1][1].replace("CTRL+W", "CTRL+S"), "uncertain", None)
            with self.assertRaises(admission.SafetyHalt):
                admission.admit_rows(rows, declaration, context)

    def test_original_unknown_cannot_be_removed_or_regraded(self):
        with tempfile.TemporaryDirectory() as temp:
            declaration, context, rows = self.fixture(Path(temp))
            for changed in (rows[:-1], rows[:-1] + [("old-receipt", rows[-1][1], "completed", "{}")]):
                with self.assertRaises(admission.SafetyHalt):
                    admission.admit_rows(changed, declaration, context)

    def test_dispatching_is_not_an_acceptable_historical_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            declaration, context, rows = self.fixture(Path(temp))
            rows[-1] = ("old-receipt", rows[-1][1], "dispatching", None)
            with self.assertRaises(admission.SafetyHalt):
                admission.admit_rows(rows, declaration, context)

    def test_different_boot_session_pid_container_owner_or_pause_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            declaration, context, rows = self.fixture(Path(temp))
            for key, value in (("guest_boot_id", "old-boot"), ("session_id", "old-session"),
                               ("gateway_pid", 99), ("container_id", "other"), ("container_owner", "other"),
                               ("paused", False), ("container_running", False), ("desktop_id", "linux-vm:old-session")):
                with self.subTest(key=key), self.assertRaises(admission.SafetyHalt):
                    admission.admit_rows(rows, declaration, dict(context, **{key: value}))

    def test_modified_original_failure_evidence_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            declaration, context, rows = self.fixture(Path(temp))
            Path(declaration["original_result"]["path"]).write_text(json.dumps({"accepted": True}))
            with self.assertRaises(admission.SafetyHalt):
                admission.admit_rows(rows, declaration, context)

    def test_reset_without_retirement_or_reboot_is_not_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            declaration, context, rows = self.fixture(Path(temp))
            path = Path(declaration["reset_evidence"]["path"])
            original = json.loads(path.read_text())
            for change in ({"old_gateway_stop_confirmed": False}, {"old_app_stop_confirmed": False},
                           {"new_guest_boot_id": "old-boot"}, {"new_gateway_session": "old-session"},
                           {"old_uncertain_action_replayed": True}, {"old_unknown_reclassified_as_completed": True}):
                path.write_text(json.dumps(dict(original, **change)))
                declaration["reset_evidence"]["sha256"] = admission.digest(path)
                with self.subTest(change=change), self.assertRaises(admission.SafetyHalt):
                    admission.admit_rows(rows, declaration, context)

    def test_forged_failed_grade_is_not_an_isolated_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            declaration, context, rows = self.fixture(Path(temp))
            path = Path(declaration["original_result"]["path"])
            path.write_text(json.dumps({"uncertain_actions": 0, "accepted": False, "artifact_passed": False}))
            declaration["original_result"]["sha256"] = admission.digest(path)
            with self.assertRaises(admission.SafetyHalt):
                admission.admit_rows(rows, declaration, context)


if __name__ == "__main__":
    unittest.main()
