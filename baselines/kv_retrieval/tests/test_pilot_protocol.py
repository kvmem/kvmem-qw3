import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from run_pilot_arm import server_command
from trace_proxy import TraceProxy
from summarize_pilot import align_steps


class PilotProtocolTest(unittest.TestCase):
    def test_summary_first_token_is_not_a_resumed_action(self):
        requests = [
            {"received_at": 11, "message_id": "summary", "api_first_content_s": 5},
            {"received_at": 32, "message_id": "action", "api_first_content_s": 1, "first_tool_call_s": 2}]
        events = [
            {"observed_at": 10, "event": {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "read-1"}]}}},
            {"observed_at": 31, "event": {"type": "system", "subtype": "compact_boundary", "compact_metadata": {"trigger": "auto"}}},
            {"observed_at": 35, "event": {"type": "assistant", "message": {"id": "action", "content": [
                {"type": "tool_use", "id": "read-2"}]}}}]
        rows, metadata = align_steps(requests, events)
        self.assertTrue(rows[0]["is_compaction"])
        self.assertIsNone(rows[0]["observation_to_first_content_s"])
        self.assertEqual(rows[1]["observation_to_first_content_s"], 23)
        self.assertEqual(rows[1]["observation_to_first_tool_call_s"], 24)
        self.assertEqual(metadata["identified_summary_requests"], 1)

    def test_transparent_stream_and_first_content(self):
        payload = b'{"messages":[{"role":"user","content":"hi"}]}'
        observed = []
        class Backend(BaseHTTPRequestHandler):
            def do_POST(self):
                observed.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                events = [{"type": "ping"}, {"type": "content_block_delta",
                    "index": 0, "delta": {"type": "text_delta", "text": "hello"}}, {"type": "message_stop"}]
                for event in events:
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                    self.wfile.flush()
                self.close_connection = True
            def log_message(self, *_):
                pass
        backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        threading.Thread(target=backend.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as directory:
            proxy = TraceProxy(("127.0.0.1", 0), f"http://127.0.0.1:{backend.server_port}", directory)
            threading.Thread(target=proxy.serve_forever, daemon=True).start()
            try:
                conn = http.client.HTTPConnection("127.0.0.1", proxy.server_port, timeout=5)
                conn.request("POST", "/v1/messages?beta=true", body=payload, headers={"Content-Type": "application/json"})
                reply = conn.getresponse()
                wire = reply.read()
                self.assertEqual(reply.status, 200)
                self.assertIn(b'"text": "hello"', wire)
                conn.close()
                path = Path(directory) / "requests.jsonl"
                for _ in range(100):
                    if path.exists() and path.stat().st_size:
                        break
                    time.sleep(.01)
                row = json.loads(path.read_text())
                self.assertEqual(observed, [("/v1/messages?beta=true", payload)])
                self.assertEqual(row["status"], "completed")
                self.assertIsNotNone(row["api_first_content_s"])
                self.assertLessEqual(row["api_first_content_s"], row["request_wall_s"])
            finally:
                proxy.shutdown()
                proxy.server_close()
                backend.shutdown()
                backend.server_close()

    def test_methods_keep_generation_and_retrieval_contract(self):
        plan = {"temperature": 1, "top_p": .95, "top_k": 20, "seed": 73,
            "prefill_chunk": 512, "max_output_tokens": 32768, "thinking_budget": 28672,
            "cpu_kv_cap_gib": 40, "methods": {
                "native_dense": {"context_tokens": 131072},
                "native_kvmem": {"context_tokens": 524288, "selection_budget": 32768,
                    "generation_budget": 8192, "gpu_planning_ratio": .98},
                "infllm": {"context_tokens": 524288, "allocator_cap_gib": 57,
                    "config": "configs/infllm_32k_candidate.json"}}}
        k = server_command("native_kvmem", plan, Path("/tmp/pilot-fixture"), 18241)
        i = server_command("infllm", plan, Path("/tmp/pilot-fixture"), 18241)
        d = server_command("native_dense", plan, Path("/tmp/pilot-fixture"), 18241)
        self.assertIn("--kvmem-query-replay", k)
        self.assertEqual(k[k.index("--kvmem-guided-reselect") + 1], "both")
        self.assertEqual(k[k.index("--mtp-chain") + 1], "1")
        self.assertEqual(d[d.index("--mtp-chain") + 1], "0")
        self.assertIn("--prefix-cache", d)
        self.assertIn("--continuous-batching", d)
        for cmd in (k, i, d):
            self.assertEqual(cmd[cmd.index("--seed") + 1], "73")
            self.assertEqual(cmd[cmd.index("--thinking-budget") + 1], "28672")


if __name__ == "__main__":
    unittest.main()
