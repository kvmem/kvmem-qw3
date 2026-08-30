#!/usr/bin/env python3

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from regrade_deepswe_report import regrade
from run_deepswe_guided_smoke import server_evidence, verifier_grade


def metrics(reward=1, f2p=(3, 3), p2p=(5, 5)):
    return {
        "verifier": {
            "reward": reward,
            "f2p_passed": f2p[0],
            "f2p_total": f2p[1],
            "p2p_passed": p2p[0],
            "p2p_total": p2p[1],
        }
    }


class DeepSweGradingTest(unittest.TestCase):
    def test_full_reward_and_counts_pass(self):
        self.assertTrue(verifier_grade(metrics())["passed"])

    def test_zero_reward_fails_even_when_verifier_process_succeeded(self):
        self.assertFalse(verifier_grade(metrics(reward=0))["passed"])

    def test_partial_f2p_or_p2p_fails(self):
        self.assertFalse(verifier_grade(metrics(f2p=(2, 3)))["passed"])
        self.assertFalse(verifier_grade(metrics(p2p=(4, 5)))["passed"])

    def test_missing_metrics_fail_closed(self):
        self.assertFalse(verifier_grade(None)["passed"])
        self.assertFalse(verifier_grade({})["passed"])

    def test_report_regrade_uses_metrics_not_container_exit(self):
        report = {
            "schema": "old",
            "tasks": [
                {
                    "task_id": "pass",
                    "arm_status": 0,
                    "verifier_status": 0,
                    "metrics": metrics(),
                },
                {
                    "task_id": "fail",
                    "arm_status": 0,
                    "verifier_status": 0,
                    "metrics": metrics(reward=0, f2p=(2, 3)),
                },
            ],
        }
        regrade(report)
        self.assertTrue(report["tasks"][0]["passed"])
        self.assertFalse(report["tasks"][1]["passed"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["schema"],
                         "qw3.deepswe_guided_kvmem_smoke.v2")

    def test_server_evidence_requires_root_task_on_every_request(self):
        with tempfile.TemporaryDirectory() as directory:
            arm = pathlib.Path(directory)
            (arm / "qw3_server.timed.log").write_text(
                "KVMem harness pins spans_root=1 blocks_root=4 "
                "mandatory_tokens=6272\n"
                "KVMem harness pins spans_root=1 blocks_root=4 "
                "mandatory_tokens=6400\n",
                encoding="utf-8",
            )
            evidence = server_evidence(arm)
        self.assertEqual(evidence["harness_pin_request_count"], 2)
        self.assertEqual(evidence["root_task_pin_request_count"], 2)
        self.assertTrue(evidence["all_harness_requests_pin_root_task"])
        self.assertEqual(evidence["root_task_blocks_min"], 4)
        self.assertEqual(evidence["root_task_blocks_max"], 4)


if __name__ == "__main__":
    unittest.main()
