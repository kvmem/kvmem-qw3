"""Validate the baseline's reuse of native Claude Code text/tool semantics."""
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


class NativeBridgeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = subprocess.Popen([str(ROOT / "native_build/baseline_native_bridge"),
            str(REPO / "models/Qwen3.8-27B-BF16")], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.process.stdin.close()
        cls.process.wait(timeout=10)
        cls.process.stdout.close()

    def rpc(self, value):
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()
        reply = json.loads(self.process.stdout.readline())
        self.assertTrue(reply["ok"], reply)
        return reply["result"]

    def test_tool_history_and_tokenization(self):
        request = {"model": "baseline", "max_tokens": 128,
            "system": [{"type": "text", "text": "Preserve exact file text."}],
            "tools": [{"name": "Bash", "description": "Run shell", "input_schema": {
                "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}],
            "messages": [
                {"role": "user", "content": "Read the file."},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Inspect current state.", "signature": "opaque"},
                    {"type": "tool_use", "id": "tool_1", "name": "Bash", "input": {"command": "cat file.py"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "  x = 1\n"}]}]}
        result = self.rpc({"op": "render", "request": request, "enable_thinking": True})
        self.assertIn("Inspect current state.", result["prompt"])
        self.assertIn("  x = 1\n", result["prompt"])
        self.assertIn("<function=Bash>", result["prompt"])
        self.assertTrue(result["prompt"].endswith("<|im_start|>assistant\n<think>\n"))
        self.assertEqual(result["input_ids"], self.rpc({"op": "tokenize", "text": result["prompt"]})["input_ids"])
        self.assertEqual(result, self.rpc({"op": "render", "request": request, "enable_thinking": True}))

    def test_native_tool_response_parsing(self):
        result = self.rpc({"op": "parse", "text": '<think>inspect</think>\nUpdate the file.\n<tool_call>\n<function=Bash>\n<parameter=command>\nprintf "a\\nb"\n</parameter>\n</function>\n</tool_call>'})
        self.assertEqual(result["reasoning_content"], "inspect")
        self.assertEqual(result["content"], "Update the file.")
        call = result["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "Bash")
        self.assertEqual(json.loads(call["arguments"]), {"command": 'printf "a\\nb"'})

    def events(self, wire):
        return [json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: ")]

    def test_stream_first_content_is_not_buffered(self):
        self.rpc({"op": "stream_start", "input_tokens": 12, "enable_thinking": False})
        event = self.events(self.rpc({"op": "stream_text", "text": "A"})["wire"])
        self.assertTrue(any(x.get("delta", {}).get("text") == "A" for x in event))
        end = self.events(self.rpc({"op": "stream_finish", "reason": "length"})["wire"])
        self.assertEqual(end[-1]["type"], "message_stop")
        self.assertTrue(any(x.get("delta", {}).get("stop_reason") == "max_tokens" for x in end))

    def test_stream_thinking_split_tool_and_unicode(self):
        tools = [{"type": "function", "function": {"name": "Bash", "parameters": {
            "type": "object", "properties": {"command": {"type": "string"}}}}}]
        self.rpc({"op": "stream_start", "input_tokens": 42, "tools": tools,
                  "enable_thinking": True})
        body = 'inspect</think>\n先读文件。<tool_call>\n<function=Bash>\n<parameter=command>\nprintf "中文\\n"\n</parameter>\n</function>\n</tool_call>'
        wire = ""
        # Exercise every XML/thinking boundary; production token IDs also split
        # UTF-8 bytes, so feed the same text again through native token decoding.
        for char in body:
            wire += self.rpc({"op": "stream_text", "text": char})["wire"]
        finish = self.rpc({"op": "stream_finish"})
        wire += finish["wire"]
        events = self.events(wire)
        reasoning = "".join(x.get("delta", {}).get("thinking", "") for x in events)
        content = "".join(x.get("delta", {}).get("text", "") for x in events)
        arguments = "".join(x.get("delta", {}).get("partial_json", "") for x in events)
        self.assertEqual(reasoning, "inspect")
        self.assertIn("先读文件。", content)
        self.assertNotIn("<tool_call>", content)
        self.assertEqual(json.loads(arguments), {"command": 'printf "中文\\n"'})
        self.assertTrue(any(x.get("delta", {}).get("stop_reason") == "tool_use" for x in events))
        self.assertEqual(finish["raw_completion"], "<think>\n" + body)
        self.rpc({"op": "stream_start", "input_tokens": 42, "tools": tools,
                  "enable_thinking": True})
        token_wire = ""
        for token in self.rpc({"op": "tokenize", "text": body})["input_ids"]:
            token_wire += self.rpc({"op": "stream_token", "token_id": token})["wire"]
        token_wire += self.rpc({"op": "stream_finish"})["wire"]
        token_args = "".join(x.get("delta", {}).get("partial_json", "") for x in self.events(token_wire))
        self.assertEqual(json.loads(token_args), json.loads(arguments))

    def test_count_tokens_without_generation_limit(self):
        result = self.rpc({"op": "render", "require_max_tokens": False,
            "request": {"model": "baseline", "messages": [{"role": "user", "content": "hi"}]}})
        self.assertGreater(len(result["input_ids"]), 0)

    def test_multiple_tools_preserved_until_end_of_generation(self):
        tools = [{"type": "function", "function": {"name": "Bash", "parameters": {
            "type": "object", "properties": {"command": {"type": "string"}}}}}]
        self.rpc({"op": "stream_start", "input_tokens": 4, "tools": tools, "enable_thinking": False})
        wire = ""
        for command in ("ls", "pwd"):
            wire += self.rpc({"op": "stream_text", "text": '<tool_call>\n<function=Bash>\n<parameter=command>\n' + command + '\n</parameter>\n</function>\n</tool_call>\n'})["wire"]
        final = self.rpc({"op": "stream_finish"})
        wire += final["wire"]
        events = self.events(wire)
        starts = [x for x in events if x["type"] == "content_block_start" and x["content_block"]["type"] == "tool_use"]
        self.assertEqual(len(starts), 2)
        self.assertEqual(final["finish_reason"], "tool_calls")

    def test_truncated_stream_is_explicit_error(self):
        tools = [{"type": "function", "function": {"name": "Bash", "parameters": {
            "type": "object", "properties": {"command": {"type": "string"}}}}}]
        self.rpc({"op": "stream_start", "input_tokens": 4, "tools": tools, "enable_thinking": False})
        self.rpc({"op": "stream_text", "text": '<tool_call>\n<function=Bash>\n<parameter=command>\nls'})
        self.process.stdin.write(json.dumps({"op": "stream_finish", "reason": "length"}) + "\n")
        self.process.stdin.flush()
        reply = json.loads(self.process.stdout.readline())
        self.assertFalse(reply["ok"])
        self.assertIn("incomplete", reply["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
