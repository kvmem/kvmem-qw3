import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from qw3_computer.codex import CodexClient, CodexTransportError, RPC_MESSAGE_LIMIT


class CodexTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.client = CodexClient(self.temp.name)
        self.stream = asyncio.StreamReader(limit=RPC_MESSAGE_LIMIT)
        self.client.proc = SimpleNamespace(stdout=self.stream, stdin=Mock())
        self.future = asyncio.get_running_loop().create_future()
        self.client.pending[1] = self.future

    async def asyncTearDown(self):
        if not self.future.done(): self.future.cancel()
        self.temp.cleanup()

    async def test_oversized_line_is_classified_and_pending_rpcs_fail_without_unhandled_reader(self):
        self.stream.feed_data(b'x' * (RPC_MESSAGE_LIMIT + 1))
        self.stream.feed_eof()
        await self.client._read()
        with self.assertRaises(CodexTransportError) as raised:
            await self.future
        self.assertEqual(raised.exception.code, "message_too_large")
        event = await self.client.events.get()
        self.assertEqual(event["params"]["code"], "message_too_large")
        log = json.loads((Path(self.temp.name) / "transport-errors.jsonl").read_text())
        self.assertEqual(log["message_limit_bytes"], RPC_MESSAGE_LIMIT)
        self.assertLess(len(json.dumps(log)), 300)

    async def test_invalid_json_never_logs_the_payload(self):
        self.stream.feed_data(b'private-test-marker-is-not-json\n')
        self.stream.feed_eof()
        await self.client._read()
        with self.assertRaises(CodexTransportError) as raised: await self.future
        self.assertEqual(raised.exception.code, "invalid_json")
        self.assertNotIn("private-test-marker", (Path(self.temp.name) / "transport-errors.jsonl").read_text())

    async def test_valid_response_resolves_the_pending_call(self):
        data = json.dumps({"id": 1, "result": {"thread": {"id": "saved"}}}).encode() + b'\n'
        self.stream.feed_data(data); self.stream.feed_eof()
        await self.client._read()
        self.assertEqual((await self.future)["thread"]["id"], "saved")
        self.assertEqual(self.client.last_message_bytes, len(data))
        self.assertIsNone(self.client.reader_error)

    async def test_failed_reader_rejects_new_writes_immediately(self):
        self.client.reader_error = CodexTransportError("reader_failed", "test failure")
        with self.assertRaises(CodexTransportError): await self.client._send({"method": "ignored"})
        self.client.proc.stdin.write.assert_not_called()
