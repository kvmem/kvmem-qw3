"""Opt-in: real pinned Codex and MCP; deterministic backend, no GPU claims."""

import asyncio
import base64
import io
import json
import os
import random
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock

from aiohttp import web
from PIL import Image

from qw3_computer.bridge import create_app
from qw3_computer.app import Controller, web_app
from qw3_computer.codex import CodexClient, write_config, RPC_MESSAGE_LIMIT
from test_bridge import chunk, serve
from test_controller import FixtureDesktop
from qw3_computer.desktop import Desktop
from qw3_computer.device_gateway import Gateway, create_app as gateway_app
import test_device_gateway as gateway_fixture


@unittest.skipUnless(os.environ.get("QW3_TEST_CODEX") == "1", "set QW3_TEST_CODEX=1 for real Codex")
class CodexIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_observed_target_tool_executes_and_returns_its_scope_through_real_codex(self):
        driver = gateway_fixture.FakeDriver()
        driver.capabilities.update(actions=['open_app'], open_apps=['notepad'],
                                   app_processes={'notepad':['Gedit']}, available_now=True)
        original_observe = driver.observe
        async def observe():
            result = await original_observe()
            result['foreground']['process'] = 'Gedit' if driver.effects else 'other'
            return result
        driver.observe = observe
        calls = []
        async def backend(request):
            body = await request.json(); calls.append(body)
            response = web.StreamResponse(headers={'Content-Type':'text/event-stream'})
            await response.prepare(request)
            if len(calls) == 1:
                tool = next(t['function'] for t in body['tools'] if t['function']['name'].endswith('__target'))
                self.assertIn('operation', tool['parameters']['properties'])
                self.assertEqual(tool['parameters']['properties']['app']['enum'], ['notepad'])
                await chunk(response, {'tool_calls':[{'index':0,'id':'wire-target-call','type':'function',
                    'function':{'name':tool['name'],'arguments':json.dumps({'operation':'ensure_app','app':'notepad',
                                                        'transition_id':'wire-target-one','timeout_seconds':2})}}]})
                await chunk(response, finish='tool_calls')
            else:
                await chunk(response, {'content':'Application foreground confirmed; business completion is untested.'})
                await chunk(response, finish='stop')
            await response.write(b'data: [DONE]\n\n')
            return response
        backend_app=web.Application();backend_app.router.add_post('/v1/chat/completions',backend)
        backend_runner,url=await serve(backend_app)
        with tempfile.TemporaryDirectory(prefix='qw3-target-wire-') as runtime:
            root=Path(runtime); gateway=Gateway(driver,root/'gateway')
            gateway_runner,gateway_url=await serve(gateway_app(gateway,'test-gateway-token'))
            desktop=Desktop(gateway_url,'test-gateway-token',root/'desktop')
            controller=Controller(root/'app',desktop,url+'/v1',0,device_platform='linux')
            app_runner,app_url=await serve(web_app(controller));controller.port=int(app_url.rsplit(':',1)[1])
            try:
                await controller.start();await controller.start_task('Qualify one bounded application transition.')
                async with asyncio.timeout(40):
                    while controller.state['status'] in ('starting','running','verifying'):await asyncio.sleep(.02)
                self.assertEqual(controller.state['status'],'completed',controller.state)
                self.assertEqual(driver.effects,1)
                self.assertEqual(controller.state['tool_calls_used'],1)
                output=next(m for m in calls[1]['messages'] if m['role']=='tool')
                text='\n'.join(c['text'] for c in output['content'] if c['type']=='text')
                self.assertIn('"verified": true',text)
                self.assertIn('business completion are not verified',text)
                self.assertIn('"process": "Gedit"',text)
            finally:
                await controller.close();await app_runner.cleanup();await gateway_runner.cleanup()
                await gateway.close();await backend_runner.cleanup()

    async def test_metadata_only_resume_preserves_large_image_history_for_the_next_turn(self):
        requests = []

        async def backend(request):
            body = await request.json()
            # Keep proof markers only; no large image request copies in memory.
            text = json.dumps(body['messages'])
            requests.append({'original_marker': 'original-history-marker-0' in text,
                             'previous_response': 'saved-response-1' in text,
                             'image_count': text.count('image_url')})
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            await chunk(response, {'content': 'saved-response-' + str(len(requests))})
            await chunk(response, finish='stop')
            await response.write(b'data: [DONE]\n\n')
            return response

        backend_app = web.Application(client_max_size=128 * 1024 * 1024)
        backend_app.router.add_post('/v1/chat/completions', backend)
        backend_runner, url = await serve(backend_app)
        bridge_runner, bridge_url = await serve(create_app(url + '/v1'))
        with tempfile.TemporaryDirectory(prefix='qw3-large-history-') as runtime:
            write_config(runtime, bridge_url + '/v1', sys.executable,
                         [str(Path(__file__).with_name('fixture_mcp.py').resolve())])
            image = io.BytesIO()
            Image.frombytes('RGB', (1024, 1024), random.Random(3).randbytes(1024 * 1024 * 3)).save(image, format='PNG')
            data_url = 'data:image/png;base64,' + base64.b64encode(image.getvalue()).decode()
            client = CodexClient(runtime)

            async def completed(turn):
                async with asyncio.timeout(60):
                    while True:
                        event = await client.events.get()
                        self.assertNotEqual(event.get('method'), 'app/disconnected', event)
                        if event.get('method') == 'turn/completed' and event['params']['turn']['id'] == turn:
                            self.assertEqual(event['params']['turn']['status'], 'completed')
                            return
            try:
                await client.start()
                thread = await client.new_thread()
                for index in range(4):
                    await completed(await client.turn(thread, 'original-history-marker-' + str(index), [data_url]))
                total = sum(path.stat().st_size for path in Path(runtime, 'sessions').rglob('*.jsonl'))
                self.assertGreater(total, RPC_MESSAGE_LIMIT)
                await client.close()
                client = CodexClient(runtime)
                await client.start()
                resumed = await client.rpc('thread/resume', {'threadId': thread, 'excludeTurns': True})
                self.assertEqual(resumed['thread']['id'], thread)
                self.assertFalse(resumed['thread']['turns'])
                self.assertLess(len(json.dumps(resumed)), 65536)
                await completed(await client.turn(thread, 'after_resume_probe'))
                self.assertTrue(requests[-1]['original_marker'])
                self.assertTrue(requests[-1]['previous_response'])
                self.assertGreaterEqual(requests[-1]['image_count'], 4)
                self.assertIsNone(client.reader_error)
            finally:
                await client.close()
                await bridge_runner.cleanup()
                await backend_runner.cleanup()

    async def test_verified_action_schema_reaches_backend_through_actual_codex_mcp(self):
        calls=[]
        async def backend(request):
            body=await request.json();calls.append(body)
            response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
            await response.prepare(request)
            await chunk(response,{'content':'Schema transport qualification only.'})
            await chunk(response,finish='stop')
            await response.write(b'data: [DONE]\n\n')
            return response
        app=web.Application();app.router.add_post('/v1/chat/completions',backend)
        backend_runner,url=await serve(app)
        try:
            for platform in ('windows','linux','android'):
                with tempfile.TemporaryDirectory(prefix='qw3-schema-wire-') as runtime:
                    desktop=FixtureDesktop();desktop.start=AsyncMock()
                    pixels=io.BytesIO();Image.new('RGB',(128,96),'blue').save(pixels,format='PNG')
                    desktop.observe.return_value=desktop.latest|{'image':base64.b64encode(pixels.getvalue()).decode()}
                    controller=Controller(runtime,desktop,url+'/v1',0,workflow=True,
                        device_platform=platform,thinking=True,model_output_tokens=16384)
                    app_runner,app_url=await serve(web_app(controller));controller.port=int(app_url.rsplit(':',1)[1])
                    try:
                        offset=len(calls);await controller.start();await controller.start_task('Inspect available tools only; do not send input.')
                        async with asyncio.timeout(40):
                            while len(calls)==offset:await asyncio.sleep(.02)
                        act=next(t['function'] for t in calls[-1]['tools'] if t['function']['name'].endswith('__act'))
                        fields=act['parameters'].get('properties',{})
                        self.assertIn('action',fields,'Actual Codex discarded the action schema before the model received it')
                        self.assertEqual(fields['step']['type'],'object')
                        self.assertEqual(fields['coordinate_space']['enum'],['normalized_1000'])
                        self.assertEqual({b['properties']['kind'].get('const',b['properties']['kind'].get('enum',[None])[0]) for b in fields['action']['oneOf']},
                            {'click_control','click','type','key','drag','scroll','open_app'})
                        self.assertLess(len(json.dumps(act['parameters'])),8000)
                    finally:
                        await controller.close();await app_runner.cleanup()
        finally:await backend_runner.cleanup()

    async def test_completed_reviews_continue_same_codex_history_and_budget(self):
        calls = []
        async def backend(request):
            body = await request.json()
            calls.append(body)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await chunk(response, {"reasoning_content": f"Review thought {len(calls)}."})
            if len(calls) % 2:
                name = next(t["function"]["name"] for t in body["tools"]
                            if t["function"]["name"].endswith("observe"))
                await chunk(response, {"tool_calls": [{"index": 0, "id": f"observe_{len(calls)}",
                    "type": "function", "function": {"name": name, "arguments": "{}"}}]})
                await chunk(response, finish="tool_calls")
            else:
                await chunk(response, {"content": f"Review marker {len(calls) // 2} completed."})
                await chunk(response, finish="stop")
            await response.write(b"data: [DONE]\n\n")
            return response
        app = web.Application()
        app.router.add_post("/v1/chat/completions", backend)
        backend_runner, url = await serve(app)
        with tempfile.TemporaryDirectory(prefix="qw3-followup-integration-") as runtime:
            desktop = FixtureDesktop()
            desktop.start = AsyncMock()
            pixels = io.BytesIO()
            Image.new("RGB", (128, 96), "blue").save(pixels, format="PNG")
            desktop.observe.return_value = desktop.latest | {"image": base64.b64encode(pixels.getvalue()).decode()}
            controller = Controller(runtime, desktop, url + "/v1", 0, tool_call_limit=4,
                                    output_policy="task", thinking=True)
            app_runner, app_url = await serve(web_app(controller))
            controller.port = int(app_url.rsplit(":", 1)[1])
            try:
                await controller.start()
                async def finish():
                    async with asyncio.timeout(40):
                        while controller.state["status"] in ("starting", "running", "verifying"):
                            await asyncio.sleep(.02)
                    self.assertEqual(controller.state["status"], "completed", controller.state)
                    self.assertTrue(desktop.paused)
                await controller.start_task("Initial purchase review marker.")
                await finish()
                task_id, thread_id = controller.state["id"], controller.state["thread_id"]
                for batch in (1, 2, 3):
                    self.assertEqual(controller.state["tool_calls_used"], batch)
                    turn_id = controller.state["turn_id"]
                    await controller.followup(task_id, turn_id, f"batch-{batch:03}", f"New public message batch {batch}.")
                    await finish()
                    self.assertEqual(controller.state["id"], task_id)
                    self.assertEqual(controller.state["thread_id"], thread_id)
                    # Retrying a confirmed owner request must not run again.
                    await controller.followup(task_id, turn_id, f"batch-{batch:03}", f"New public message batch {batch}.")
                self.assertEqual(len(calls), 8)
                self.assertTrue(all(c["enable_thinking"] for c in calls))
                self.assertTrue(controller.state["thinking"])
                self.assertEqual(sum(bool(m.get("reasoning_content")) for m in calls[-1]["messages"]), 7)
                self.assertEqual(controller.state["tool_calls_used"], 4)
                self.assertEqual(controller.budget_remaining(), 0)
                self.assertEqual(len(controller.state["messages"]), 4)
                final_history = json.dumps(calls[-1]["messages"])
                for text in ["Initial purchase review marker.", "Review marker 1 completed.",
                             "New public message batch 1.", "New public message batch 2.", "New public message batch 3."]:
                    self.assertIn(text, final_history)
            finally:
                await controller.close()
                await app_runner.cleanup()
                await backend_runner.cleanup()

    async def test_images_success_error_mcp_rounds_and_interrupt(self):
        calls = []
        waiting, cancelled = asyncio.Event(), asyncio.Event()
        async def backend(request):
            body = await request.json()
            calls.append(body)
            if len(calls) > 3:
                waiting.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            # Exercise repeated connection-liveness events with the real
            # pinned Codex parser, before releasing either tool or text.
            await asyncio.sleep(.1)
            if len(calls) <= 2:
                name = next(t["function"]["name"] for t in body["tools"]
                            if t["function"]["name"].endswith("observe"))
                await chunk(response, {"tool_calls": [{"index": 0, "id": "call_" + str(len(calls)),
                    "type": "function", "function": {"name": name, "arguments": "{}"}}]})
                await chunk(response, finish="tool_calls")
            else:
                await chunk(response, {"content": "Two fixture screenshots received."})
                await chunk(response, finish="stop")
            await response.write(b"data: [DONE]\n\n")
            return response
        app = web.Application()
        app.router.add_post("/v1/chat/completions", backend)
        backend_runner, url = await serve(app)
        bridge_runner, bridge_url = await serve(create_app(url + "/v1", heartbeat_seconds=.02))
        with tempfile.TemporaryDirectory(prefix="qw3-codex-integration-") as runtime:
            write_config(runtime, bridge_url + "/v1", sys.executable,
                         [str(Path(__file__).with_name("fixture_mcp.py").resolve()), "--error-second"])
            client = CodexClient(runtime)
            async def completed(turn):
                while True:
                    event = await asyncio.wait_for(client.events.get(), 40)
                    if event.get("method") == "turn/completed" and event["params"]["turn"]["id"] == turn:
                        return event["params"]["turn"]
                    if event.get("method") == "app/disconnected":
                        self.fail("Codex disconnected")
            try:
                await client.start()
                thread = await client.new_thread()
                image = io.BytesIO()
                Image.new("RGB", (128, 96), "red").save(image, format="PNG")
                data_url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
                turn = await client.turn(thread, "Observe the fixture twice, then describe the result.", [data_url])
                result = await completed(turn)
                self.assertEqual(result["status"], "completed", result)
                self.assertEqual(len(calls), 3)
                self.assertTrue(any("Two fixture" in i.get("text", "") for i in result["items"]))
                self.assertTrue(any(p.get("type") == "image_url" for m in calls[0]["messages"]
                                    if isinstance(m["content"], list) for p in m["content"]))
                outputs = [m for m in calls[2]["messages"] if m["role"] == "tool"]
                self.assertEqual([m["tool_call_id"] for m in outputs], ["call_1", "call_2"])
                for output in outputs:
                    self.assertTrue(any(p.get("type") == "image_url" for p in output["content"]), output)
                self.assertIn("Action rejected without dispatch", json.dumps(outputs[1]))
                self.assertEqual(calls[1]["messages"], calls[2]["messages"][:len(calls[1]["messages"])])
                turn = await client.turn(thread, "Wait for cancellation.")
                await asyncio.wait_for(waiting.wait(), 10)
                await client.interrupt(thread, turn)
                self.assertEqual((await completed(turn))["status"], "interrupted")
                await asyncio.wait_for(cancelled.wait(), 5)
            except BaseException:
                print((Path(runtime) / "app-server.log").read_text()[-8000:], file=sys.stderr)
                raise
            finally:
                await client.close()
                await bridge_runner.cleanup()
                await backend_runner.cleanup()


class CatalogContextTests(unittest.TestCase):
    def test_reduced_backend_capacity_changes_catalog_and_compaction_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            write_config(directory,'http://127.0.0.1:1/v1',context_window=524288)
            catalog=json.loads((Path(directory)/'models.json').read_text())
            self.assertEqual(catalog['models'][0]['context_window'],524288)
            self.assertIn('model_auto_compact_token_limit = 471859',(Path(directory)/'config.toml').read_text())
            with self.assertRaises(ValueError):write_config(directory,'unused',context_window=True)
