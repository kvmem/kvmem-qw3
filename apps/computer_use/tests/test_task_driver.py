import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from aiohttp import web

from task_driver import ReviewMailbox, run_task


class TaskDriverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.requests=[]
        self.before={"ready":True,"status":"idle","desktop_paused":True}
        self.states=[];self.index=0;self.started=False;self.stop_fails=False
        self.followup_states=[];self.followups=[];self.followup_fails=False
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
        async def followup(request):
            self.requests.append(request.path);self.followups.append(await request.json())
            if self.followup_fails:return web.json_response({'error':'uncertain delivery'},status=503)
            self.states=[self.followup_states.pop(0)];self.index=0
            return web.json_response(self.states[0])
        app=web.Application();app.router.add_get('/api/state',state);app.router.add_post('/api/task',task);app.router.add_post('/api/control',control);app.router.add_post('/api/followup',followup)
        self.runner=web.AppRunner(app);await self.runner.setup();site=web.TCPSite(self.runner,'127.0.0.1',0);await site.start()
        self.url='http://127.0.0.1:'+str(site._server.sockets[0].getsockname()[1])

    async def asyncTearDown(self):
        await self.runner.cleanup();self.temp.cleanup()

    def fixture(self,status,**extra):
        return dict(id='owned',ready=True,status=status,desktop_paused=status not in ('running','starting'),**extra)

    async def run_case(self,timeout=1,**kwargs):
        output=Path(self.temp.name)/'run'
        result=await run_task(self.url,'Public task only',output,timeout,poll_interval=.01,**kwargs)
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

    def review(self,number,**changes):
        return self.fixture('completed',turn_status='completed',thread_id='same-thread',turn_id=f'turn-{number}',
                            tool_calls_used=number*3,tool_call_limit=100) | changes

    async def updates(self,state,batch):
        if batch>3:return None
        return dict(task_id=state['id'],completed_turn_id=state['turn_id'],event_id=f'update-{batch}',prompt=f'Public update {batch}')

    async def test_three_updates_preserve_task_budget_and_record_four_reviews(self):
        self.states=[self.review(1)];self.followup_states=[self.review(i) for i in (2,3,4)]
        r=await self.run_case(on_completed=self.updates)
        self.assertEqual(r['outcome'],'agent_completed');self.assertTrue(r['queue_may_continue'])
        self.assertEqual(len(r['completed_reviews']),4);self.assertEqual(len(self.followups),3)
        self.assertEqual(self.requests.count('/api/task'),1)
        self.assertEqual([p['completed_turn_id'] for p in self.followups],['turn-1','turn-2','turn-3'])

    async def test_lost_followup_reply_is_not_retried(self):
        self.states=[self.review(1)];self.followup_fails=True
        r=await self.run_case(on_completed=self.updates)
        self.assertEqual(r['outcome'],'infrastructure_error');self.assertFalse(r['queue_may_continue'])
        self.assertEqual(len(self.followups),1);self.assertIn('/api/control',self.requests)

    async def test_budget_exhaustion_does_not_deliver_messages(self):
        self.states=[self.review(1,tool_calls_used=100)]
        async def forbidden(*_):raise AssertionError('Delivered with no budget')
        r=await self.run_case(on_completed=forbidden)
        self.assertEqual(r['outcome'],'tool_budget_exhausted');self.assertTrue(r['queue_may_continue'])

    async def test_review_cannot_reset_budget(self):
        self.states=[self.review(1)];self.followup_states=[self.review(2,tool_calls_used=0)]
        r=await self.run_case(on_completed=self.updates)
        self.assertEqual(r['outcome'],'infrastructure_error');self.assertEqual(len(self.followups),1)

    async def test_review_cannot_increase_limit(self):
        self.states=[self.review(1)];self.followup_states=[self.review(2,tool_call_limit=200)]
        r=await self.run_case(on_completed=self.updates)
        self.assertEqual(r['outcome'],'infrastructure_error');self.assertEqual(len(self.followups),1)

    async def test_replayed_completed_turn_is_not_new_checkpoint(self):
        self.states=[self.review(1)];self.followup_states=[self.review(1)]
        r=await self.run_case(on_completed=self.updates)
        self.assertEqual(r['outcome'],'infrastructure_error');self.assertEqual(len(self.followups),1)

    async def test_new_thread_is_not_continued_as_same_history(self):
        self.states=[self.review(1)];self.followup_states=[self.review(2,thread_id='new-thread')]
        r=await self.run_case(on_completed=self.updates)
        self.assertEqual(r['outcome'],'infrastructure_error');self.assertEqual(len(self.followups),1)

    async def test_mailbox_wait_is_inside_whole_task_deadline(self):
        self.states=[self.review(1)]
        mailbox=ReviewMailbox(Path(self.temp.name)/'mailbox',3)
        r=await self.run_case(timeout=.05,on_completed=mailbox)
        self.assertEqual(r['outcome'],'time_limit');self.assertTrue(r['queue_may_continue'])
        self.assertTrue((mailbox.path/'1/checkpoint.json').is_file());self.assertEqual(self.followups,[])

    async def test_mailbox_rejects_other_checkpoint_response(self):
        self.states=[self.review(1)]
        mailbox=ReviewMailbox(Path(self.temp.name)/'mailbox',3)
        async def host():
            while not (mailbox.path/'1/checkpoint.json').is_file():await asyncio.sleep(.001)
            (mailbox.path/'1/followup.json').write_text(json.dumps({'batch':2,'request':await self.updates(self.states[0],1)}))
        background=asyncio.create_task(host())
        r=await self.run_case(on_completed=mailbox);await background
        self.assertEqual(r['outcome'],'infrastructure_error');self.assertEqual(self.followups,[])

    async def test_wrong_task_followup_is_never_sent(self):
        self.states=[self.review(1)]
        async def other(state,batch):return (await self.updates(state,batch)) | {'task_id':'other'}
        r=await self.run_case(on_completed=other)
        self.assertEqual(r['outcome'],'infrastructure_error');self.assertEqual(self.followups,[])

    async def test_host_delivery_failure_stops_wait_without_resuming(self):
        self.states=[self.review(1)]
        mailbox=ReviewMailbox(Path(self.temp.name)/'mailbox',3)
        async def host():
            while not (mailbox.path/'1/checkpoint.json').is_file():await asyncio.sleep(.001)
            (mailbox.path/'1/abort.json').write_text(json.dumps({'batch':1,'infrastructure_error':True}))
        background=asyncio.create_task(host())
        r=await self.run_case(on_completed=mailbox);await background
        self.assertEqual(r['outcome'],'infrastructure_error');self.assertFalse(r['queue_may_continue'])
        self.assertEqual(self.followups,[]);self.assertIn('/api/control',self.requests)
