"""Real local HTTP fixture checks and conservative recovery acceptance."""
import contextlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

import fault_browser_pilot as probe


@contextlib.contextmanager
def local_fixture(condition):
    node=shutil.which('node')
    if not node:raise unittest.SkipTest('Node is required for real fixture HTTP checks')
    with tempfile.TemporaryDirectory() as name:
        root=Path(name)
        (root/'server.js').write_text(probe.SERVER.read_text().replace('__QW3_FAULT_CONDITION__',json.dumps(condition)))
        with socket.socket() as endpoint:
            endpoint.bind(('127.0.0.1',0));port=endpoint.getsockname()[1]
        process=subprocess.Popen([node,str(root/'server.js'),str(root),str(port)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        try:
            deadline=time.monotonic()+5
            while not (root/'server-ready.json').exists():
                if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Owned fixture did not start')
                time.sleep(.01)
            yield root,'http://127.0.0.1:'+str(port)
        finally:
            if process.poll() is None:process.terminate()
            process.wait(timeout=5)
            process.stderr.close()


def request(origin,path,body=None):
    req=urllib.request.Request(origin+path,data=None if body is None else json.dumps(body).encode(),
                               headers={'Content-Type':'application/json'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req,timeout=3) as response:return response.status,response.read()
    except urllib.error.HTTPError as error:return error.code,error.read()


class ActualFixtureTests(unittest.TestCase):
    def test_unarmed_and_unready_form_cannot_save(self):
        with local_fixture('slow_loading') as (root,origin):
            self.assertEqual(request(origin,'/save',probe.FIELDS)[0],409)
            (root/'arm.json').write_text(json.dumps({'armed_at_ms':time.time()*1000}))
            self.assertFalse(json.loads(request(origin,'/phase')[1])['ready'])
            self.assertEqual(request(origin,'/save',probe.FIELDS)[0],409)
            self.assertFalse((root/'submissions.jsonl').exists())

    def test_normal_save_is_durable_and_private_state_is_not_served(self):
        with local_fixture('normal') as (root,origin):
            (root/'arm.json').write_text(json.dumps({'armed_at_ms':time.time()*1000}))
            self.assertEqual(request(origin,'/save',probe.FIELDS)[0],200)
            rows=(root/'submissions.jsonl').read_text()
            self.assertTrue(probe.ordinary.grade_records(rows,probe.FIELDS)['passed'])
            self.assertEqual(json.loads((root/'fault-state.json').read_text())['saved_count'],1)
            self.assertEqual(request(origin,'/fault-state.json')[0],404)
            self.assertEqual(request(origin,'/arm.json')[0],404)

    def test_no_progress_preserves_pending_without_a_saved_record(self):
        with local_fixture('no_progress') as (root,origin):
            (root/'arm.json').write_text(json.dumps({'armed_at_ms':time.time()*1000}))
            self.assertEqual(request(origin,'/save',probe.FIELDS)[0],202)
            state=json.loads((root/'fault-state.json').read_text())
            self.assertEqual(state['saved_count'],0)
            self.assertEqual(len(state['save_attempts']),1)
            self.assertFalse((root/'submissions.jsonl').exists())

    def test_malformed_or_extra_fields_never_create_a_submission(self):
        with local_fixture('normal') as (root,origin):
            (root/'arm.json').write_text(json.dumps({'armed_at_ms':time.time()*1000}))
            for body in [[],{},probe.FIELDS|{'extra':'bad'},probe.FIELDS|{'name':''}]:
                self.assertEqual(request(origin,'/save',body)[0],400)
            self.assertFalse((root/'submissions.jsonl').exists())


class RecoveryGradingTests(unittest.TestCase):
    def sample(self,condition):
        return {'condition':condition,'fault_exercised':True,'passed':True,'saved_visible':True,
                'saved_count':1,'notice_closed':True}

    def test_normal_artifact_without_a_triggered_dialog_does_not_pass_fault_test(self):
        value=self.sample('focus_dialog')|{'fault_exercised':False}
        state={'status':'completed','uncertain_actions':[]}
        self.assertFalse(probe.response_grade(value,state)['passed'])

    def test_deadline_or_model_error_is_not_successful_no_progress_handling(self):
        value=self.sample('no_progress')|{'passed':False,'saved_count':0,'saved_visible':False}
        state={'status':'paused','uncertain_actions':[],'gui_recovery':{'paused_reason':'unchanged_observation_loop'}}
        grade=probe.response_grade(value,state)
        self.assertTrue(grade['safe_no_progress_guard_passed'])
        self.assertFalse(grade['unassisted_business_passed'])
        self.assertFalse(probe.response_grade(value,state,deadline_reached=True)['passed'])
        self.assertFalse(probe.response_grade(value,state|{'turn_error':'model error'})['passed'])
        self.assertFalse(probe.response_grade(value,state|{'status':'completed'})['passed'])

    def test_unreconciled_receipt_cannot_pass_even_with_correct_saved_data(self):
        value=self.sample('slow_loading')
        self.assertFalse(probe.response_grade(value,{'status':'completed','uncertain_actions':[{'status':'uncertain'}]})['passed'])

    def test_initializer_reuses_the_owned_transport_without_leaking_patch_to_plain_tasks(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);ordinary=probe.ordinary
            original=ordinary.SERVER
            fixture={'platform':'linux','identity':'qualification','container_id':'owned'}
            def initialize(*args,**kwargs):
                self.assertNotEqual(ordinary.SERVER,original)
                self.assertEqual(ordinary.expected('qualification'),probe.FIELDS)
                self.assertIn('Reference R-204',ordinary.instruction('fixture-url',probe.FIELDS))
                return dict(fixture),'prompt'
            with patch.object(probe,'ROOT',root),patch.object(ordinary,'initialize',side_effect=initialize):
                output,_=probe.initialize('linux','qualification',root,condition='focus_dialog')
            self.assertEqual(output['condition'],'focus_dialog')
            self.assertIs(ordinary.SERVER,original)


if __name__=='__main__':unittest.main()
