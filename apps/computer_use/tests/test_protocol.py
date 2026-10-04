import base64
import copy
import unittest

from qw3_computer.protocol import ProtocolError, convert_request

IMAGE = "data:image/png;base64," + base64.b64encode(b"fixture image bytes").decode()


def request(inputs=None):
    return {"model": "qw3-computer", "stream": True, "store": False,
            "instructions": "Stable system instruction", "input": inputs or [],
            "tools": [{"type": "function", "name": "observe", "parameters": {"type": "object"}}]}


class ProtocolTests(unittest.TestCase):
    def test_computer_action_schema_loss_fails_before_backend_conversion(self):
        from qw3_computer.tool_schema import tools_for_platform
        r=request([{'role':'user','content':'Inspect'}])
        for schema in ({},{'type':'object','properties':{}},
                       {'properties':{'desktop_id':{},'observation_id':{},'action_id':{},'action':{'type':'string'}}}):
            r['tools']=[{'type':'function','name':'mcp__computer__act','parameters':schema}]
            with self.subTest(schema=schema),self.assertRaisesRegex(ProtocolError,'before model dispatch'):
                convert_request(r,'qw3')
        for platform in ('windows','linux','android'):
            for workflow in (False,True):
                schema=tools_for_platform(platform,workflow)[1]['inputSchema']
                r['tools']=[{'type':'function','name':'mcp__computer__act','parameters':schema}]
                self.assertEqual(convert_request(r,'qw3').body['tools'][0]['function']['parameters'],schema)

    def test_opt_in_thinking_preserves_reasoning_tool_and_final_message(self):
        thought = {"type": "reasoning", "summary": [], "content": [
            {"type": "reasoning_text", "text": "Check the actual window before typing."}]}
        call = {"type": "function_call", "name": "observe", "call_id": "check1", "arguments": "{}"}
        output = {"type": "function_call_output", "call_id": "check1", "output": "Notepad is focused."}
        history = [{"role": "user", "content": "Inspect"}, thought, call, output,
                   thought, {"role": "assistant", "content": "Verified."}, {"role": "user", "content": "Continue"}]
        body = convert_request(request(history) | {"reasoning": {"effort": "medium"}}, "qw3",
                               allow_thinking=True).body
        self.assertTrue(body["enable_thinking"])
        assistant = [m for m in body["messages"] if m["role"] == "assistant"]
        self.assertEqual(len(assistant), 2)
        self.assertEqual(assistant[0]["tool_calls"][0]["id"], "check1")
        self.assertEqual(assistant[1]["content"], "Verified.")
        self.assertTrue(all(m["reasoning_content"] == thought["content"][0]["text"] for m in assistant))
        # Explicit none still disables generation but keeps prior reasoning.
        off = convert_request(request(history), "qw3", allow_thinking=True).body
        self.assertFalse(off["enable_thinking"])
        self.assertEqual(off["messages"], body["messages"])

    def test_thinking_rejects_unrecoverable_or_summary_only_history(self):
        for thought in ({"type": "reasoning", "summary": []},
                        {"type": "reasoning", "encrypted_content": "foreign-opaque-state"},
                        {"type": "reasoning", "summary": [], "content": [{"type": "output_text", "text": "x"}]}):
            with self.subTest(thought=thought), self.assertRaises(ProtocolError):
                convert_request(request([thought]), "qw3", allow_thinking=True)

    def test_full_history_keeps_tool_image_order_and_prefix(self):
        history = [{"role": "user", "content": [{"type": "input_text", "text": "Inspect"},
                                                     {"type": "input_image", "image_url": IMAGE}]},
                   {"type": "function_call", "name": "observe", "call_id": "call_1", "arguments": "{}"},
                   {"type": "function_call_output", "call_id": "call_1", "output": [
                       {"type": "input_text", "text": "observation-1"},
                       {"type": "input_image", "image_url": IMAGE}]}]
        a = convert_request(request(history), "qw3").body
        b = convert_request(request(history + [{"role": "user", "content": "Continue"}]), "qw3").body
        self.assertEqual(a["messages"], b["messages"][:-1])
        self.assertEqual(a["messages"][-1]["tool_call_id"], "call_1")
        self.assertEqual([p["type"] for p in a["messages"][-1]["content"]], ["text", "image_url"])
        changed = copy.deepcopy(history)
        changed[-1]["output"][-1]["image_url"] = "data:image/png;base64," + base64.b64encode(b"new image").decode()
        self.assertNotEqual(a["messages"], convert_request(request(changed), "qw3").body["messages"])

    def test_unsupported_semantics_fail_explicitly(self):
        for update in ({"previous_response_id": "resp_x"}, {"store": True}, {"stream": False},
                       {"reasoning": {"effort": "high"}}, {"input": [{"type": "reasoning", "summary": []}]},
                       {"tools": [{"type": "custom", "name": "shell"}]},
                       {"tools": [{"type": "function", "name": "observe", "strict": True}]}):
            with self.subTest(update=update), self.assertRaises(ProtocolError):
                convert_request(request() | update, "qw3")

    def test_orphan_duplicate_and_missing_results_fail(self):
        call = {"type": "function_call", "name": "observe", "call_id": "x", "arguments": "{}"}
        output = {"type": "function_call_output", "call_id": "x", "output": "ok"}
        for history in ([output], [call], [call, call, output], [call, output, output],
                        [call, {"role": "user", "content": "continue"}, output]):
            with self.subTest(history=history), self.assertRaises(ProtocolError):
                convert_request(request(history), "qw3")

    def test_namespaces_are_bijective_and_collisions_fail(self):
        r = request()
        r["tools"] = [{"type": "namespace", "name": "computer", "tools": r["tools"]}]
        c = convert_request(r, "qw3")
        self.assertEqual(c.names, {"computer__observe": ("computer", "observe")})
        r["tools"].append({"type": "function", "name": "computer__observe"})
        with self.assertRaises(ProtocolError):
            convert_request(r, "qw3")
