"""Actual stdlib HTTP behavior for Windows recovery fixtures, without a model."""
import contextlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

from fault_browser_windows_guest import Fixture,page
from fault_browser_pilot import SERVER,FIELDS
from browser_pilot import grade_records


@contextlib.contextmanager
def server(condition):
    with tempfile.TemporaryDirectory() as name:
        root=Path(name);html=page(condition,SERVER.read_text())
        fixture=Fixture(root,condition,html)
        http=ThreadingHTTPServer(('127.0.0.1',0),fixture.handler())
        thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
        try:yield root,'http://127.0.0.1:'+str(http.server_port)
        finally:http.shutdown();http.server_close();thread.join(timeout=2)


def request(origin,path,body=None):
    req=urllib.request.Request(origin+path,data=None if body is None else json.dumps(body).encode(),
                               headers={'Content-Type':'application/json'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req,timeout=3) as response:return response.status,response.read()
    except urllib.error.HTTPError as exc:return exc.code,exc.read()


class WindowsFixtureTests(unittest.TestCase):
    def arm(self,root,seconds_ago=0):
        (root/'arm.json').write_text(json.dumps({'armed_at_ms':(time.time()-seconds_ago)*1000}))

    def test_page_uses_same_qualified_template_with_one_literal_condition(self):
        for condition in ['normal','slow_loading','focus_dialog','no_progress']:
            html=page(condition,SERVER.read_text())
            self.assertIn('const mode='+json.dumps(condition),html)
            self.assertNotIn('${',html)
            self.assertIn('Session notice',html)
            self.assertNotIn(FIELDS['reference'],html)

    def test_unready_slow_form_never_saves_and_private_files_are_not_served(self):
        with server('slow_loading') as (root,origin):
            self.assertEqual(request(origin,'/save',FIELDS)[0],409)
            self.arm(root)
            self.assertFalse(json.loads(request(origin,'/phase')[1])['ready'])
            self.assertEqual(request(origin,'/save',FIELDS)[0],409)
            self.assertEqual(request(origin,'/fault-state.json')[0],404)
            self.assertFalse((root/'submissions.jsonl').exists())

    def test_ready_save_is_durable_and_malformed_events_are_rejected(self):
        with server('normal') as (root,origin):
            self.arm(root)
            self.assertEqual(request(origin,'/event',{'name':[]})[0],400)
            self.assertEqual(request(origin,'/save',FIELDS|{'extra':'bad'})[0],400)
            self.assertEqual(request(origin,'/save',FIELDS)[0],200)
            self.assertTrue(grade_records((root/'submissions.jsonl').read_text(),FIELDS)['passed'])
            self.assertEqual(json.loads((root/'fault-state.json').read_text())['saved_count'],1)

    def test_permanent_saving_returns_pending_without_business_artifact(self):
        with server('no_progress') as (root,origin):
            self.arm(root)
            self.assertEqual(request(origin,'/save',FIELDS)[0],202)
            value=json.loads((root/'fault-state.json').read_text())
            self.assertEqual(value['saved_count'],0)
            self.assertEqual(len(value['save_attempts']),1)
            self.assertFalse((root/'submissions.jsonl').exists())


if __name__=='__main__':unittest.main()
