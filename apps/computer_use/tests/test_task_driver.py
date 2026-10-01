import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from aiohttp import web

from task_driver import run_task


class TaskDriverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.requests=[]
        self.before={"ready":True,"status":"idle","desktop_paused":True}
        self.states=[];self.index=0;self.started=False;self.stop_fails=False
        async def state(request):
            self.requests.append(request.path)
            if not self.started:return web.json_response(self.before)
            item=self.states[min(self.index,len(self.states)-1)];self.index+=1
            return web.json_response(item)
        async def task(request):
            self.requests.append(request.path);self.started=True
            return web.json_response(self.states[0])
        async def control(request):
            self.requests.append(request.path)
            self.assertEqual(await request.json(),{"command":"stop"})
            if self.stop_fails:return web.json_response({"error":"lost executor"},status=503)
            return web.json_response({"id":"owned","ready":True,"status":"stopped","desktop_paused":True})
        app=web.Application();app.router.add_get('/api/state',state);app.router.add_post('/api/task',task);app.router.add_post('/api/control',control)
        self.runner=web.AppRunner(app);await self.runner.setup();site=web.TCPSite(self.runner,'127.0.0.1',0);await site.start()
        self.url='http://127.0.0.1:'+str(site._server.sockets[0].getsockname()[1])

    async def asyncTearDown(self):
        await self.runner.cleanup();self.temp.cleanup()

    def fixture(self,status,**extra):
        return dict(id='owned',ready=True,status=status,desktop_paused=status not in ('running','starting'),**extra)

    async def run_case(self,timeout=1):
        output=Path(self.temp.name)/'run'
        result=await run_task(self.url,'Public task only',output,timeout,poll_interval=.01)
        self.assertEqual(json.loads((output/'result.json').read_text()),result)
        self.assertNotIn('/api/observe',self.requests)
        return result

    async def test_completed_is_not_a_grade(self):
        self.states=[self.fixture('completed')];r=await self.run_case()
        self.assertEqual(r['outcome'],'agent_completed');self.assertFalse(r['graded']);self.assertTrue(r['queue_may_continue'])

    async def test_agent_failure_does_not_stop_campaign(self):
        self.states=[self.fixture('failed',turn_error={'code':'model_failure'})];r=await self.run_case()
        self.assertEqual(r['outcome'],'agent_failed');self.assertTrue(r['queue_may_continue'])

    async def test_budget_and_uncertainty_are_terminal_without_resume(self):
        self.states=[self.fixture('paused',tool_calls_used=500,tool_call_limit=500)];r=await self.run_case()
        self.assertEqual(r['outcome'],'tool_budget_exhausted');self.assertTrue(r['queue_may_continue'])
        self.assertNotIn('/api/control',self.requests)

    async def test_uncertain_pause_is_preserved(self):
        self.states=[self.fixture('paused',uncertain_actions=[{'action_id':'unknown-delivery'}])];r=await self.run_case()
        self.assertEqual(r['outcome'],'paused_uncertain');self.assertTrue(r['queue_may_continue'])

    async def test_timeout_stops_owned_task_and_continues(self):
        self.states=[self.fixture('running')];r=await self.run_case(.05)
        self.assertEqual(r['outcome'],'time_limit');self.assertTrue(r['queue_may_continue']);self.assertIn('/api/control',self.requests)

    async def test_failed_stop_is_infrastructure_error(self):
        self.states=[self.fixture('running')];self.stop_fails=True;r=await self.run_case(.05)
        self.assertEqual(r['outcome'],'infrastructure_error');self.assertFalse(r['queue_may_continue']);self.assertEqual(r['prior_outcome'],'time_limit')

    async def test_existing_task_is_not_stopped(self):
        self.before=self.fixture('running');r=await self.run_case()
        self.assertEqual(r['outcome'],'infrastructure_error');self.assertNotIn('/api/task',self.requests);self.assertNotIn('/api/control',self.requests)
