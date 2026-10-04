"""A crashed formal worker is not authorization for another model workload."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from completed_campaign_pilot import completed_campaign
from osworld_linux.campaign import jobs
from windows_fault_pilot import admitted_child


class CompletedAdmissionTests(unittest.TestCase):
    def fixture(self,root):
        rows=jobs()
        for row in rows:
            row['status']='finished'
            directory=root/row['id'];directory.mkdir()
            (directory/'result.json').write_text(json.dumps({'status':'scored','score':0,'finished_at':123}))
        plan={'status':'finished','jobs':rows}
        (root/'plan.json').write_text(json.dumps(plan))
        return plan

    def test_finished_plan_without_last_terminal_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);plan=self.fixture(root)
            (root/plan['jobs'][-1]['id']/'result.json').write_text(json.dumps({'status':'running'}))
            with self.assertRaises(ValueError):completed_campaign(root)

    def test_pending_original_attempt_rejects_standalone_handoff(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);plan=self.fixture(root);plan['jobs'][-1]['status']='pending'
            (root/'plan.json').write_text(json.dumps(plan))
            with self.assertRaises(ValueError):completed_campaign(root)

    def test_live_formal_app_rejects_handoff_even_with_all_result_files(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.fixture(root)
            with patch('completed_campaign_pilot.verify_frozen'),patch('completed_campaign_pilot.command',
                return_value=['python','-m','qw3_computer.app','--runtime',str(root/'attempt/app')]):
                with self.assertRaises(RuntimeError):completed_campaign(root)

    def test_standalone_child_checks_completed_proof_not_just_status_and_pid(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.fixture(root)
            with patch('completed_campaign_pilot.verify_frozen'),patch('completed_campaign_pilot.command',return_value=[]):
                proof=completed_campaign(root)
                path=root/'admission.json'
                path.write_text(json.dumps({'status':'running-standalone-pilot','pilot_pid':123,'formal_completion':proof}))
                self.assertEqual(admitted_child(path,pid=123)['formal_completion'],proof)
                value=json.loads(path.read_text());value['formal_completion']['plan_sha256']='wrong'
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):admitted_child(path,pid=123)


if __name__=='__main__':unittest.main()
