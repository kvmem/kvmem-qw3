"""Saved-input replay qualification; all HTTP responses are scripted CPU data."""
import argparse
import asyncio
import contextlib
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aiohttp import web
import thinking_budget_replay as http
import thinking_budget_replay_queue as queue


def source():
    schema = {"type": "object", "properties": {"action": {"type": "object", "properties": {
        "kind": {"enum": ["type"]}, "text": {"type": "string"}}, "required": ["kind", "text"],
        "additionalProperties": False}}, "required": ["action"], "additionalProperties": False}
    return dict(model="qw3", messages=[{"role": "user", "content": "immutable saved fixture input"}],
                max_tokens=16384, enable_thinking=True, stream=True, stream_options={"include_usage": True},
                tools=[{"type": "function", "function": {"name": "mcp__computer__act", "parameters": schema}},
                       {"type": "function", "function": {"name": "mcp__computer__observe", "parameters": {"type": "object"}}}])


def chunk(delta=None, finish=None):
    return {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}


def events(mode="tools"):
    data = [chunk({"reasoning_content": "private synthetic reasoning"})]
    if mode == "tools":
        data += [chunk({"tool_calls": [{"index": 0, "id": "call-local", "function": {
            "name": "mcp__computer__act", "arguments": '{"action":'}}]}),
                 chunk({"tool_calls": [{"index": 0, "id": "call-local", "function": {
                     "arguments": '{"kind":"type","text":"never execute"}}'}}]})]
    elif mode == "content": data += [chunk({"content": "synthetic answer"})]
    data += [chunk(finish="tool_calls" if mode == "tools" else "length" if mode == "reasoning" else "stop"),
             {"choices": [], "usage": {"prompt_tokens": 123, "completion_tokens": 16384 if mode == "reasoning" else 9,
                                        "total_tokens": 16507 if mode == "reasoning" else 132}}]
    return data


def stream(mode="tools", done=True):
    value = http.Stream()
    for event in events(mode): value.feed(json.dumps(event).encode())
    if done: value.feed(b"[DONE]")
    return value


class ContractTests(unittest.TestCase):
    def test_only_thinking_budget_changes_in_replay_body(self):
        original = source(); retained = copy.deepcopy(original)
        for budget in (0, 8192):
            body = http.replay_body(original, budget)
            self.assertEqual(body.pop("thinking_budget"), budget)
            self.assertEqual(body, original)
        self.assertEqual(original, retained)

    def test_undeclared_or_boolean_budget_is_rejected(self):
        for budget in (True, -1, 4096, "8192"):
            with self.assertRaises(ValueError): http.replay_body(source(), budget)

    def test_changed_thinking_stream_or_output_budget_is_rejected(self):
        for key, value in (("enable_thinking", False), ("max_tokens", 8192), ("stream", False), ("thinking_budget", 0)):
            body = source(); body[key] = value
            with self.assertRaises(ValueError): http.replay_body(body, 0)

    def test_reasoning_only_length_stop_is_not_answer(self):
        result = stream("reasoning").summary(source()["tools"])
        self.assertTrue(result["reasoning_only_length_stop"])
        self.assertFalse(result["answer_or_tool_output_generated"])

    def test_split_arguments_and_repeated_call_id_are_retained_but_never_executed(self):
        result = stream().summary(source()["tools"])
        self.assertEqual(result["valid_generated_tool_calls"], 1)
        self.assertEqual(result["tool_calls_executed"], 0)
        self.assertFalse(result["tool_calls"][0]["executed"])

    def test_invalid_tool_arguments_remain_failed_structural_output(self):
        s = stream(); s.calls[0]["arguments"] = '{"action":{"kind":"type","text":false}}'
        self.assertEqual(s.summary(source()["tools"])["valid_generated_tool_calls"], 0)

    def test_unknown_function_is_not_accepted_or_executed(self):
        s = stream(); s.calls[0]["name"] = "unoffered_function"
        result = s.summary(source()["tools"])
        self.assertEqual(result["valid_generated_tool_calls"], 0)
        self.assertEqual(result["tool_calls_executed"], 0)

    def test_missing_done_or_usage_cannot_pass(self):
        for s in (stream(done=False), stream()):
            if s.done: s.usage = None
            with self.assertRaises(ValueError): s.summary(source()["tools"])

    def test_actual_usage_above_output_budget_is_rejected(self):
        s = stream(); s.usage["completion_tokens"] = 16385
        with self.assertRaises(ValueError): s.summary(source()["tools"])

    def test_tool_call_identity_change_is_rejected(self):
        s = http.Stream()
        s.feed(json.dumps(chunk({"tool_calls": [{"index": 0, "id": "first", "function": {}}]})).encode())
        with self.assertRaises(ValueError):
            s.feed(json.dumps(chunk({"tool_calls": [{"index": 0, "id": "second", "function": {}}]})).encode())

    def test_reasoning_after_final_output_is_rejected(self):
        s = http.Stream(); s.feed(json.dumps(chunk({"content": "answer"})).encode())
        with self.assertRaises(ValueError): s.feed(json.dumps(chunk({"reasoning_content": "late"})).encode())

    def test_missing_tools_after_tool_finish_is_rejected(self):
        s = stream("content"); s.finish_reason = "tool_calls"
        with self.assertRaises(ValueError): s.summary(source()["tools"])

    def test_predecessor_unknown_or_halted_cannot_start_replay(self):
        p = dict(parent_queue="q", parent_controller={"pid": 123}, parent_protocol_sha256="sha")
        for state in ({"pid": 123, "protocol_sha256": "sha", "status": "halted-no-further-dispatch"},
                      {"pid": 124, "protocol_sha256": "sha", "status": "running"}):
            with patch.object(queue, "read", return_value=state), self.assertRaises(queue.SafetyHalt):
                queue.predecessor_ready(p)

    def test_waiting_parent_requires_actual_live_identity(self):
        p = dict(parent_queue="q", parent_controller={"pid": 123}, parent_protocol_sha256="sha")
        state = dict(pid=123, protocol_sha256="sha", status="waiting-for-forty-eight-and-ten")
        with patch.object(queue, "read", return_value=state), patch.object(queue.base, "require_identity") as check:
            self.assertIsNone(queue.predecessor_ready(p))
        check.assert_called_once_with(p["parent_controller"])

    def test_completed_but_live_parent_still_waits(self):
        p = dict(parent_queue="q", parent_controller={"pid": 123}, parent_protocol_sha256="sha")
        with patch.object(queue, "read", return_value=dict(pid=123, protocol_sha256="sha", status="completed")), \
                patch.object(queue.base, "process_retired", return_value=False):
            self.assertIsNone(queue.predecessor_ready(p))

    def test_four_replays_are_interleaved_with_fixed_original_input_and_output_budget(self):
        rows = queue.trials()
        self.assertEqual([r["thinking_budget"] for r in rows], [0, 8192, 8192, 0])
        self.assertEqual(len({r["id"] for r in rows}), 4)

    def execute_mock_queue(self, unconfirmed_at=None):
        with tempfile.TemporaryDirectory(prefix="qw3-readonly-controller-tests-") as directory:
            root = Path(directory); request = root / "source.json"
            request.write_text(json.dumps(source()))
            p = dict(trials=queue.trials(), source_request=str(request), engine_off_identity={"off": True})
            args = argparse.Namespace(protocol=root / "protocol.json", state_dir=root / "state", wait_seconds=60)
            queue.save(args.protocol, p)
            dispatched = []
            def engine(controller, row):
                controller.state["engine_runtime"] = str(root / "engine" / row["id"])
            async def fake_replay(data, budget, out, **options):
                self.assertEqual(data, request.read_bytes())
                dispatched.append((out.name, budget))
                out.mkdir(parents=True)
                result = dict(termination_confirmed=len(dispatched) != unconfirmed_at,
                              tool_calls_executed=0, business_success_claimed=False)
                queue.save(out / "result.json", result)
                return result
            with patch.object(queue, "validate", return_value=p), \
                    patch.object(queue, "predecessor_ready", return_value={"registered_engine": {"pid": 7}}), \
                    patch.object(queue.image.Controller, "fresh_engine", engine), \
                    patch.object(queue.image.Controller, "parent_idle"), \
                    patch.object(queue.owned, "model_lock", side_effect=contextlib.nullcontext), \
                    patch.object(queue.owned, "register_engine"), patch.object(queue, "replay", fake_replay):
                if unconfirmed_at:
                    with self.assertRaises(queue.SafetyHalt): queue.run(args)
                else: queue.run(args)
            return dispatched, queue.read(args.state_dir / "queue.json")

    def test_full_four_replay_control_sequence_is_serial_and_read_only(self):
        dispatched, state = self.execute_mock_queue()
        self.assertEqual(dispatched, [(r["id"], r["thinking_budget"]) for r in queue.trials()])
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["qw3_replays_completed"], 4)
        self.assertEqual(state["tool_calls_executed"], 0)
        self.assertFalse(state["business_success_claimed"])

    def test_unconfirmed_replay_blocks_all_following_requests(self):
        dispatched, state = self.execute_mock_queue(unconfirmed_at=2)
        self.assertEqual(len(dispatched), 2)
        self.assertEqual(len(state["trials"]), 1)
        self.assertEqual(state["status"], "halted-no-further-dispatch")
        self.assertTrue(state["live_children_and_evidence_preserved"])


class HttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qw3-readonly-replay-tests-")
        self.directory = Path(self.temp.name)
        self.mode = "tools"
        self.complete_cancel = True
        self.requests = []
        self.cancellations = []
        self.request_ids = set()
        app = web.Application()
        async def post(request):
            body = await request.json(); request_id = request.headers.get("x-qw3-request-id")
            self.requests.append(body); self.request_ids.add(request_id)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            if self.mode == "timeout":
                await asyncio.sleep(.2)
                return response
            for event in events("reasoning" if self.mode == "reasoning" else "tools"):
                await response.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            if self.mode != "missing-done": await response.write(b"data: [DONE]\n\n")
            await response.write_eof(); return response
        async def cancel(request):
            request_id = request.match_info["identity"]
            self.cancellations.append(request_id)
            if request_id not in self.request_ids: return web.json_response({"error": "foreign request"}, status=404)
            return web.json_response(dict(cancelled=True, completed=self.complete_cancel))
        app.router.add_post("/v1/chat/completions", post)
        app.router.add_post("/v1/requests/{identity}/cancel", cancel)
        self.runner = web.AppRunner(app); await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0); await site.start()
        self.endpoint = "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])

    async def asyncTearDown(self):
        await self.runner.cleanup(); self.temp.cleanup()

    async def run_replay(self, budget=8192, **options):
        return await http.replay(json.dumps(source(), ensure_ascii=False).encode(), budget,
                                 self.directory / "attempt", self.endpoint, **options)

    async def test_real_http_headers_and_split_tool_stream_never_dispatch_input(self):
        result = await self.run_replay()
        self.assertEqual(result["valid_generated_tool_calls"], 1)
        self.assertEqual(result["tool_calls_executed"], 0)
        self.assertTrue(result["termination_confirmed"])
        self.assertEqual(self.requests[0].pop("thinking_budget"), 8192)
        self.assertEqual(self.requests[0], source())
        self.assertEqual(set(self.cancellations), {result["request_id"]})
        self.assertFalse(result["business_success_claimed"])
        self.assertTrue((self.directory / "attempt/backend.jsonl").exists())

    async def test_real_http_length_without_final_output_is_recorded(self):
        self.mode = "reasoning"
        result = await self.run_replay(0)
        self.assertTrue(result["reasoning_only_length_stop"])
        self.assertFalse(result["answer_or_tool_output_generated"])
        self.assertTrue(result["termination_confirmed"])

    async def test_truncated_stream_is_not_completed_and_requests_own_retirement(self):
        self.mode = "missing-done"
        with self.assertRaises(ValueError): await self.run_replay()
        result = json.loads((self.directory / "attempt/result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["termination_confirmed"])
        self.assertEqual(self.cancellations, [result["request_id"]])

    async def test_timeout_closes_stream_and_confirms_own_request_retirement(self):
        self.mode = "timeout"
        with self.assertRaises(TimeoutError): await self.run_replay(seconds=.05)
        result = json.loads((self.directory / "attempt/result.json").read_text())
        self.assertTrue(result["termination_confirmed"])
        self.assertEqual(result["tool_calls_executed"], 0)

    async def test_unconfirmed_retirement_cannot_release_replay_as_success(self):
        self.complete_cancel = False
        with self.assertRaises(TimeoutError): await self.run_replay(retire_seconds=.05)
        result = json.loads((self.directory / "attempt/result.json").read_text())
        self.assertEqual(result["status"], "failed-retirement-unconfirmed")
        self.assertFalse(result["termination_confirmed"])

    async def test_existing_output_cannot_be_replayed(self):
        (self.directory / "attempt").mkdir()
        with self.assertRaises(FileExistsError): await self.run_replay()
        self.assertFalse(self.requests)

    async def test_remote_endpoint_is_rejected_before_any_request(self):
        with self.assertRaises(ValueError):
            await http.replay(json.dumps(source()).encode(), 8192, self.directory / "attempt", "https://foreign.example")
        self.assertFalse(self.requests)


if __name__ == "__main__": unittest.main()
