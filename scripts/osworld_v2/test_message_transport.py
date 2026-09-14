import base64
from copy import deepcopy
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'third_party/OSWorld-V2'))
sys.path.insert(0, str(ROOT / 'scripts/osworld_v2'))
from mm_agents.qw3_message_transport import MessagePrefixTransport
from mm_agents.qwen35vl_agent import Qwen35VLAgent


def ack(key, count):
    return NS(model_extra={'qw3_messages_cache': {'id': key, 'count': count}})


class PrefixTransportTest(unittest.TestCase):
    def test_append_and_tail_rewrite_are_lossless(self):
        state = MessagePrefixTransport()
        messages = [{'role': 'system', 'content': 'a'}, {'role': 'user', 'content': 'b'}]
        self.assertEqual(state.prepare(messages)[0], messages)
        state.acknowledge(messages, ack('first', 2))
        appended = messages + [{'role': 'assistant', 'content': 'c'}]
        wire, extra = state.prepare(appended)
        self.assertEqual(messages[:extra['qw3_messages_prefix']['count']] + wire, appended)
        messages[1]['content'] = 'edited in place'
        wire, extra = state.prepare(messages)
        self.assertEqual(extra['qw3_messages_prefix']['count'], 1)
        self.assertEqual(wire, messages[1:])
        messages[0]['content'] = 'changed system'
        self.assertNotIn('qw3_messages_prefix', state.prepare(messages)[1])

    def test_no_ack_reset_and_changed_image(self):
        state = MessagePrefixTransport()
        messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'A'}}]}]
        state.acknowledge(messages, ack('x', 1))
        messages[0]['content'][0]['image_url']['url'] = 'B'
        self.assertEqual(state.prepare(messages)[0], messages)
        state.acknowledge(messages, NS(model_extra={}))
        self.assertIsNone(state.cache_id)

    def test_cached_image_metadata_and_reset(self):
        agent = Qwen35VLAgent(history_mode='full')
        agent.screenshots = [base64.b64encode(b'raw image').decode()]
        url = agent._screenshot_url(0)
        self.assertIs(url, agent._screenshot_url(0))
        messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}}]}]
        expected = agent._sanitize_messages_for_dump(messages)
        self.assertEqual(agent._sanitize_messages_for_dump(messages, agent._image_dump_metadata), expected)
        with patch('mm_agents.qwen35vl_agent.base64.b64decode', side_effect=AssertionError('old image decoded again')):
            self.assertEqual(agent._sanitize_messages_for_dump(messages, agent._image_dump_metadata), expected)
        agent.screenshots[0] = base64.b64encode(b'new image').decode()
        self.assertNotEqual(agent._screenshot_url(0), url)
        agent.reset()
        self.assertEqual(agent._image_dump_metadata, {})
        self.assertEqual(agent._screenshot_urls, [])

    def test_missing_server_cache_reuploads_without_changing_model_request(self):
        agent = Qwen35VLAgent(history_mode='full')
        agent.use_message_prefix_transport = True
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'first'}]}]
        agent.message_transport.acknowledge(messages, ack('evicted', 1))
        full = messages + [{'role': 'user', 'content': [{'type': 'text', 'text': 'next'}]}]
        calls = []
        class Miss(Exception):
            status_code = 409
            body = {'code': 'qw3_messages_cache_miss'}
        response = ack('new', 2)
        response.choices = [NS(message=NS(content='Action: done.', tool_calls=[], reasoning_content=None), finish_reason='stop')]
        response.id, response.created, response.model = 'response', 0, 'test'
        response.usage = {'prompt_tokens': 5, 'completion_tokens': 3}
        def create(**kwargs):
            calls.append(deepcopy(kwargs))
            if len(calls) == 1:
                raise Miss()
            return response
        client = NS(chat=NS(completions=NS(create=create)), close=lambda: None)
        with patch('mm_agents.qwen35vl_agent.openai.OpenAI', return_value=client):
            result = agent.call_llm({'messages': full, 'max_tokens': 17, 'temperature': .5}, 'test')
        self.assertEqual(result, 'Action: done.')
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]['messages'], full[1:])
        self.assertEqual(calls[1]['messages'], full)
        self.assertEqual(calls[0]['max_tokens'], calls[1]['max_tokens'])
        self.assertNotIn('qw3_messages_prefix', calls[1]['extra_body'])
        self.assertTrue(agent.last_api_metadata['message_transport']['full_retry'])
        self.assertEqual(agent.message_transport.cache_id, 'new')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PrefixTransportTest)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_trace_instrumentation'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
