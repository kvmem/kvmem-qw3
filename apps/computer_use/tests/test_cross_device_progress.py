"""Missing graders, interrupted attempts and absent media must stay distinct."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cross_device_progress as progress


class EvidenceSummaryTests(unittest.TestCase):
    def test_zero_is_valid_but_missing_and_interrupted_scores_are_not(self):
        rows=[dict(status='scored',model_score_valid=True,score=0),
              dict(status='scored',model_score_valid=True,score=1),
              dict(status='infrastructure-failed',model_score_valid=False,score=0),
              dict(status='evaluator-unavailable',model_score_valid=False,score=None)]
        result=progress.formal_summary(rows)
        self.assertEqual(result['ended_attempts'],4)
        self.assertEqual(result['valid_original_scores'],2)
        self.assertEqual(result['successes'],1)
        self.assertEqual(result['infrastructure_failures'],1)
        self.assertEqual(result['unavailable_original_evaluators'],1)
        self.assertFalse(result['improvement_demonstrated'])

    def test_running_result_file_is_not_a_finished_attempt(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);campaign=root/'campaign'
            for identity,result in [('013-legacy-r1',{'status':'running'}),
                       ('013-verified-r1',{'status':'scored','score':0,'finished_at':1})]:
                path=campaign/identity/'result.json';path.parent.mkdir(parents=True)
                path.write_text(json.dumps(result))
            with patch.object(progress,'ROOT',root):rows=progress.formal_attempts(campaign)
            self.assertEqual(len(rows),1)
            self.assertTrue(rows[0]['model_score_valid'])

    def test_business_pass_without_video_or_with_unstopped_device_is_incomplete(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);path=root/'result.json'
            base=dict(task='product-python-new',status='complete',finished_at=1,passed=True,
                      model_used=True,guest_result={'desktop_stopped':True})
            with patch.object(progress,'ROOT',root):
                path.write_text(json.dumps(base))
                self.assertFalse(progress.product_row(path)['accepted'])
                path.write_text(json.dumps(base|{'recording':{'complete':True}}))
                self.assertTrue(progress.product_row(path)['accepted'])
                path.write_text(json.dumps(base|{'recording':{'complete':True},'guest_result':{'desktop_stopped':False}}))
                self.assertFalse(progress.product_row(path)['accepted'])

    def test_batch_recovery_is_not_counted_as_formal_or_business_success(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);runtime=root/'runtime';campaign=runtime/'campaign';campaign.mkdir(parents=True)
            (campaign/'plan.json').write_text(json.dumps({'jobs':[]}))
            path=root/'progress.json';path.write_text(json.dumps({'formal_attempts':[],'integration_pilots':[],
                                                               'frozen_campaign':{}}))
            attempt=runtime/'xdev-recovery/batch/xdev-probe-01';attempt.mkdir(parents=True)
            (attempt/'integration-result.json').write_text(json.dumps({'accepted':False,'artifact_passed':False,
                'app_status':'paused','fault_condition':'no_progress','fault_accepted':True,
                'fault_response_passed':True,'owned_task_retired':True}))
            with patch.object(progress,'ROOT',root),patch.object(progress,'RUNTIME',runtime):
                value=progress.refresh(path,campaign)
            self.assertEqual(value['formal_summary']['ended_attempts'],0)
            row=value['integration_pilots'][0]
            self.assertEqual(row['id'],'xdev-probe-01')
            self.assertTrue(row['fault_accepted'])
            self.assertFalse(row['accepted'])
            self.assertFalse(row['formal'])


if __name__=='__main__':unittest.main()
