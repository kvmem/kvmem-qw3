"""Budget and information-flow regressions for the new memory-QA protocol."""
import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_qwen35_longmemeval as run


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(f"<{m['role']}>\n{m['content']}" for m in messages) + '\n<assistant>'


class ProtocolTest(unittest.TestCase):
    def test_raw_token_transport_rejects_a_different_id_mapping(self):
        tok = run.AutoTokenizer.from_pretrained(run.TOKENIZER, local_files_only=True)
        audit = run.validate_raw_token_vocabulary(tok)
        self.assertTrue(audit['all_hf_token_ids_match'])
        self.assertTrue(audit['all_merge_rules_match'])
        vocab = tok.get_vocab()
        first, second = list(vocab)[:2]
        vocab[first], vocab[second] = vocab[second], vocab[first]
        with patch.object(tok, 'get_vocab', return_value=vocab):
            with self.assertRaisesRegex(AssertionError, 'token ID mapping differs'):
                run.validate_raw_token_vocabulary(tok)

    def test_native_probe_matches_declared_hf_normalization_without_changing_ids(self):
        tok = run.AutoTokenizer.from_pretrained(run.TOKENIZER, local_files_only=True)
        text = 'The formula says where\u0304u is bounded.\nQuestion: what is bounded?'
        before = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        normalized = run.native_tokenizer_probe_text(tok, text)
        self.assertIn('wher\u0113u', normalized)
        self.assertEqual(before['input_ids'], run.ids(tok, normalized))
        self.assertEqual(before, tok(text, add_special_tokens=False, return_offsets_mapping=True))
        self.assertEqual(tok.decode(before['input_ids']), normalized)

    def test_labels_and_question_do_not_change_history(self):
        sample = {'haystack_dates': ['2022-01-01'], 'haystack_sessions': [[
            {'role': 'user', 'content': 'I moved to Paris.', 'has_answer': True}]],
            'question': 'FIRST', 'answer': 'SECRET', 'answer_session_ids': ['first']}
        first = run.sessions(sample)
        sample.update(question='SECOND', answer='OTHER', answer_session_ids=[])
        sample['haystack_sessions'][0][0]['has_answer'] = False
        self.assertEqual(first, run.sessions(sample))
        self.assertNotIn('SECRET', json.dumps(first))

    def test_compaction_is_question_blind_and_final_rag_cap_is_complete(self):
        sample = {'question_id': 'fixture', 'question_type': 'single-session-user',
                  'question': 'QUESTION_ONLY_AT_END', 'answer': 'GOLD_NEVER_SENT',
                  'question_date': '2022-12-01', 'haystack_dates': ['2022-01-01'] * 3,
                  'haystack_sessions': [[{'role': 'user', 'content': 'h' * 14000}]] * 3}
        calls = []
        @contextlib.contextmanager
        def fake_server(*args):
            yield 'unused'
        def fake_generate(base, prompt, directory, name, **kwargs):
            text = ''.join(map(chr, prompt))
            calls.append((name, text))
            self.assertNotIn('GOLD_NEVER_SENT', text)
            if name.startswith('compact_'):
                self.assertNotIn('QUESTION_ONLY_AT_END', text)
                self.assertLessEqual(len(prompt), 32768)
                return {'answer': 'summary ' * 100, 'request_wall_sec': 1}
            self.assertIn('QUESTION_ONLY_AT_END', text)
            self.assertLessEqual(len(prompt), 32768)
            return {'answer': 'answer', 'request_wall_sec': 1}
        class Embeddings:
            def call(self, body):
                assert all(name.startswith('compact_') for name, _ in calls)
                assert len(calls) == 3
                return {'ranked': list(range(60)), 'scores': [1.] * 60, 'query_sec': .01}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks = [{'session_index': i, 'chunk_index': 0, 'text': 'x' * 1000,
                       'date': '2022-01-01', 'block_id': str(i)} for i in range(60)]
            run.save(root / 'indices/fixture/index.json', {'chunks': chunks, 'history_index_sec': 1})
            with patch.object(run, 'OUT', root), patch.object(run, 'server', fake_server), patch.object(run, 'generate', fake_generate):
                result = run.run_compact_rag(sample, CharacterTokenizer(), root / 'arm', Embeddings())
            self.assertEqual(result['compaction_rounds'], 3)
            self.assertGreater(result['selected_chunks'], 0)
            self.assertLess(result['selected_chunks'], 60)
            self.assertGreater(result['final_prompt_tokens'], 31000)


if __name__ == '__main__':
    unittest.main()
