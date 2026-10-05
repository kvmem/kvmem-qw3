import copy
import json
from pathlib import Path
import tempfile
import unittest

from aiohttp import web

import query_refresh_replay as replay


def request(pressure=False):
    body = dict(model="scripted-fixture", enable_thinking=True, max_tokens=16384,
                stream=True, stream_options={"include_usage": True},
                messages=[dict(role=role, content=text) for role, text in (
                    ("system", "Fixture policy"), ("developer", "Fixture tools"),
                    ("user", "Fixture task"), ("user", "Fixture initial observation"))],
                tools=[dict(type="function", function=dict(name="mcp__computer__act", parameters=dict(
                    type="object", required=["action"], properties=dict(action=dict(type="string")))))])
    if pressure:
        body["messages"].append(dict(role="assistant", content="Fixture longer history"))
    return json.dumps(body, ensure_ascii=False).encode()


class IdentityTests(unittest.TestCase):
    def test_original_input_is_not_mutated_and_generation_controls_are_rejected(self):
        body = json.loads(request())
        before = copy.deepcopy(body)
        self.assertEqual(replay.exact_body(body, 0), body)
        self.assertEqual(body, before)
        for extra in ({"thinking_budget": 0}, {"kvmem_reselect": "force"}, {"max_tokens": 1}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                replay.exact_body(dict(body, **extra), 0)
        with self.assertRaises(ValueError):
            replay.exact_body(body, False)

    def test_different_task_query_or_roles_cannot_fake_a_warm_sequence(self):
        for index in (2, 3):
            body = json.loads(request(True)); body["messages"][index]["content"] = "Different"
            with self.subTest(index=index), self.assertRaises(ValueError):
                replay.pair_identity(dict(seed=request(), pressure=json.dumps(body, ensure_ascii=False).encode()))
        body = json.loads(request(True)); body["messages"][3]["role"] = "assistant"
        with self.assertRaises(ValueError):
            replay.pair_identity(dict(seed=request(), pressure=json.dumps(body, ensure_ascii=False).encode()))

    def test_whitespace_changes_cannot_replace_the_preserved_wire(self):
        with self.assertRaises(ValueError):
            replay.pair_identity(dict(seed=request() + b' ', pressure=request(True)))


class HttpSequenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sent, self.cancelled = [], []
        self.mode = "success"
        app = web.Application()
        app.router.add_post('/v1/chat/completions', self.backend)
        app.router.add_post('/v1/requests/{request_id}/cancel', self.cancel)
        self.runner = web.AppRunner(app); await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0); await site.start()
        self.endpoint = 'http://127.0.0.1:' + str(site._server.sockets[0].getsockname()[1])

    async def asyncTearDown(self):
        await self.runner.cleanup(); self.tmp.cleanup()

    async def backend(self, request):
        wire = await request.read()
        self.sent.append((wire, request.headers['x-qw3-request-id']))
        if self.mode == 'http-error':
            return web.Response(status=500)
        prompt = 100 if len(json.loads(wire)['messages']) == 4 else 200
        events = [dict(choices=[dict(index=0, delta=dict(reasoning_content='Fixture thought.'), finish_reason=None)]),
                  dict(choices=[dict(index=0, delta=dict(tool_calls=[dict(index=0, id='fixture-call',
                       function=dict(name='mcp__computer__act', arguments='{"action":"click"}'))]), finish_reason=None)]),
                  dict(choices=[dict(index=0, delta={}, finish_reason='tool_calls')],
                       usage=dict(prompt_tokens=prompt, completion_tokens=12, total_tokens=prompt + 12))]
        data = ''.join('data: ' + json.dumps(e) + '\n\n' for e in events)
        if self.mode != 'missing-terminal': data += 'data: [DONE]\n\n'
        return web.Response(text=data, content_type='text/event-stream')

    async def cancel(self, request):
        self.cancelled.append(request.match_info['request_id'])
        return web.json_response(dict(cancelled=True, completed=self.mode != 'unconfirmed-retirement'))

    def inputs(self):
        return dict(seed=request(), pressure=request(True))

    async def run_sequence(self, expected=None):
        output = Path(self.tmp.name) / 'sequence'
        result = await replay.sequence(self.inputs(), output, self.endpoint, seconds=2,
                                       retire_seconds=.1, expected_prompt_tokens=expected)
        return result, output

    async def test_actual_http_sequence_preserves_wire_and_never_executes_generated_calls(self):
        result, output = await self.run_sequence(dict(seed=100, pressure=200))
        self.assertEqual([wire for wire, _ in self.sent], [request(), request(True)])
        self.assertEqual(len(set(identifier for _, identifier in self.sent)), 2)
        self.assertEqual(set(self.cancelled), {identifier for _, identifier in self.sent})
        self.assertTrue(result['termination_confirmed'])
        self.assertEqual(result['tool_calls_executed'], 0)
        for row in result['rows']:
            self.assertEqual(row['result']['valid_generated_tool_calls'], 1)
            self.assertFalse(row['result']['tool_calls'][0]['executed'])
            self.assertTrue(row['result']['original_generation_profile_unchanged'])
            self.assertNotIn('thinking_budget', json.loads((output / row['name'] / 'request.json').read_text()))

    async def test_seed_http_failure_retains_evidence_and_does_not_send_pressure(self):
        self.mode = 'http-error'
        with self.assertRaises(Exception): await self.run_sequence()
        self.assertEqual(len(self.sent), 1)
        result = json.loads((Path(self.tmp.name) / 'sequence/sequence-result.json').read_text())
        self.assertEqual(result['status'], 'halted-no-further-requests')
        self.assertEqual(result['rows'], [])
        self.assertTrue((Path(self.tmp.name) / 'sequence/seed/result.json').exists())

    async def test_unconfirmed_retirement_cannot_start_the_second_request(self):
        self.mode = 'unconfirmed-retirement'
        with self.assertRaises(TimeoutError): await self.run_sequence()
        self.assertEqual(len(self.sent), 1)
        result = json.loads((Path(self.tmp.name) / 'sequence/seed/result.json').read_text())
        self.assertFalse(result['termination_confirmed'])
        self.assertEqual(result['status'], 'failed-retirement-unconfirmed')

    async def test_incomplete_stream_is_not_a_successful_seed(self):
        self.mode = 'missing-terminal'
        with self.assertRaises(ValueError): await self.run_sequence()
        self.assertEqual(len(self.sent), 1)

    async def test_tokenization_mismatch_stops_without_pressure(self):
        with self.assertRaises(ValueError): await self.run_sequence(dict(seed=101, pressure=200))
        self.assertEqual(len(self.sent), 1)

    async def test_existing_attempt_directory_is_never_overwritten(self):
        await self.run_sequence()
        with self.assertRaises(FileExistsError): await self.run_sequence()
        self.assertEqual(len(self.sent), 2)


if __name__ == '__main__':
    unittest.main()
