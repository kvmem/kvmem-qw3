#!/usr/bin/env python3
"""Transport retries must wait for the previous QW3 request to release its KV."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import httpx
import openai

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'third_party' / 'OSWorld-V2'))
from mm_agents.qwen35vl_agent import Qwen35VLAgent


class RetryTest(unittest.TestCase):
    def run_case(self, cancel_fails=False):
        agent = Qwen35VLAgent.__new__(Qwen35VLAgent)
        agent.base_url = 'http://127.0.0.1:18081/v1'
        agent.api_key = 'test'
        agent.max_tokens, agent.temperature, agent.top_p = 1, 0, 1
        client = MagicMock()
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='OK', tool_calls=None), finish_reason='stop')])
        client.chat.completions.create.side_effect = [
            openai.APITimeoutError(request=httpx.Request('POST', agent.base_url)), response]

        def cancel(*args):
            self.assertEqual(client.chat.completions.create.call_count, 1)
            if cancel_fails:
                raise RuntimeError('no ack')

        with patch.dict(os.environ, {'QW3_OSWORLD_REQUEST_CANCEL': '1'}), \
                patch('mm_agents.qwen35vl_agent.openai.OpenAI', return_value=client) as factory, \
                patch.object(agent, '_cancel_qw3_request', side_effect=cancel), \
                patch('mm_agents.qwen35vl_agent.time.sleep'):
            if cancel_fails:
                with self.assertRaisesRegex(RuntimeError, 'no ack'):
                    agent.call_llm({'messages': []}, 'test')
                self.assertEqual(client.chat.completions.create.call_count, 1)
            else:
                self.assertEqual(agent.call_llm({'messages': []}, 'test'), 'OK')
                self.assertEqual(client.chat.completions.create.call_count, 2)
                calls = client.chat.completions.create.call_args_list
                self.assertNotEqual(calls[0].kwargs['extra_headers'], calls[1].kwargs['extra_headers'])
            self.assertEqual(factory.call_args.kwargs['max_retries'], 0)
            client.close.assert_called_once()

    def test_wait_before_retry(self):
        self.run_case()

    def test_no_retry_without_ack(self):
        self.run_case(True)

    def test_summary_reads_content_and_passes_stop_without_replaying_tools(self):
        agent = Qwen35VLAgent.__new__(Qwen35VLAgent)
        agent.base_url, agent.api_key = 'http://127.0.0.1:18081/v1', 'test'
        agent.max_tokens, agent.temperature, agent.top_p = 100, 0, 1
        content = 'The desktop contains the requested files. The form remains unfinished.'
        raw = content + '<tool_call>do not replay this command</tool_call>'
        message = SimpleNamespace(content=content, tool_calls=[{'function': {'name': 'computer_use'}}],
                                  qw3_raw_completion=raw)
        response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason='tool_calls')])
        client = MagicMock()
        client.chat.completions.create.return_value = response
        with patch('mm_agents.qwen35vl_agent.openai.OpenAI', return_value=client):
            returned = agent.call_llm({'messages': [], 'response_text_only': True,
                                       'stop': ['<tool_call']}, 'test')
        self.assertEqual(returned, content)
        self.assertEqual(agent.last_api_metadata['summary_response']['raw_completion'], raw)
        self.assertEqual(client.chat.completions.create.call_args.kwargs['stop'], ['<tool_call'])
        self.assertNotIn('response_text_only', client.chat.completions.create.call_args.kwargs)
        self.assertEqual(agent._extract_openai_message(message), raw)


if __name__ == '__main__':
    unittest.main()
