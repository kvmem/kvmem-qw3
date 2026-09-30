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
