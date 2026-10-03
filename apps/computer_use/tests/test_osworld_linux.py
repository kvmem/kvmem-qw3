import math
import unittest
import signal
import time
import copy
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import re
import zipfile
import importlib.util

from osworld_linux.runner import evaluation_score, stage_deadline
from osworld_linux.campaign import jobs
from osworld_windows.sites import Site


class SiteInitializerTests(unittest.TestCase):
    def test_only_declared_scheduler_fields_can_be_consumed_without_hiding_lost_business_data(self):
        class Backend(Site):
            lose_business=False
            def request(self,method,path,body=None):
                if method=='PUT':
                    self.saved=copy.deepcopy(body)
                    for field in ['time_data','action_data','enable_notifications']:
                        self.saved['data'].pop(field,None)
                    if self.lose_business:
                        self.saved['data'].pop('teamchat')
                if not hasattr(self,'saved'):
                    self.saved={'data':{},'meta':{'created_at':datetime.now(timezone.utc).isoformat(),'version':1}}
                return {'user_id':self.user,'state':self.saved}
        initial={'data':{'teamchat':{'messages':{'channel':[]}},'time_data':[{'messageId':'later','arrive_after_s':600}]}}
        with tempfile.TemporaryDirectory() as temporary:
            for explicit,lose_business,passed in [(False,False,False),(True,False,True),(True,True,False)]:
                site=Backend('teamchat','qualification'+str(explicit)+str(lose_business),Path(temporary))
                site.lose_business=lose_business
                fields=('time_data','action_data','enable_notifications') if explicit else ()
                if passed:
                    site.initialize(initial,consumed_fields=fields)
                else:
                    with self.assertRaisesRegex(ValueError,'preserve initialized task data'):
                        site.initialize(initial,consumed_fields=fields)
        self.assertIn('time_data',initial['data'])


class OriginalScoreTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('openpyxl'), 'Optional original evaluator dependency')
    def test_published_worksheet_without_dimension_still_yields_rows_for_local_metric(self):
        import openpyxl
        from osworld_windows.task035 import rows
        with tempfile.TemporaryDirectory() as temporary:
            original=Path(temporary)/'original.xlsx'
            missing=Path(temporary)/'missing-dimension.xlsx'
            book=openpyxl.Workbook()
            book.active.append(['Request','Owner','Vendor','Category','Amount','Date','Approved'])
            book.active.append(['test','alice','vendor','tools',150,'2026-10-04',True])
            book.save(original)
            with zipfile.ZipFile(original) as source,zipfile.ZipFile(missing,'w') as target:
                for member in source.infolist():
                    data=source.read(member.filename)
                    if member.filename=='xl/worksheets/sheet1.xml':
                        data=re.sub(rb'<dimension[^>]*/>',b'',data)
                    target.writestr(member,data)
            self.assertEqual(rows(original),rows(missing))

    def test_stalled_official_stage_is_bounded_and_restores_signal_handler(self):
        previous=signal.getsignal(signal.SIGALRM)
        with self.assertRaisesRegex(TimeoutError,'qualification stage'):
            with stage_deadline(.02,'qualification stage'):
                time.sleep(.2)
        self.assertEqual(signal.getsignal(signal.SIGALRM),previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.0,0.0))

    def test_each_task_has_two_interleaved_attempts_per_arm_without_id_collisions(self):
        plan = jobs()
        self.assertEqual(len({j["id"] for j in plan}), 40)
        from collections import Counter
        self.assertTrue(all(n == 2 for n in Counter((j["task"], j["arm"]) for j in plan).values()))
        for left, right in zip(plan[::2], plan[1::2]):
            self.assertEqual((left["task"], left["repetition"]), (right["task"], right["repetition"]))
            self.assertNotEqual(left["arm"], right["arm"])

    def test_original_safety_report_is_not_confused_with_numeric_score(self):
        original = {"score": 1.0, "safety": {"checks": [{"name": "reference", "result": "PASS"}]}}
        self.assertEqual(evaluation_score(original), 1.0)
        self.assertEqual(original["safety"]["checks"][0]["result"], "PASS")
        self.assertEqual(evaluation_score(0.25), 0.25)

    def test_invalid_evaluator_return_cannot_be_counted_as_success(self):
        for value in [True, None, {"safety": {}}, {"score": "1"}, math.nan, math.inf, -1, 2]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluation_score(value)
