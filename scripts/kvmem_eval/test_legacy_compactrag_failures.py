import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_longmemeval_9b_legacy_compactrag as reader
import score_longmemeval_legacy_compactrag as scorer


def response(content='', reasoning='Repeated thinking', finish='stop'):
    return {'finish_reason':finish,'input_tokens':100,'output_tokens':50,'wall_sec':1,
            'raw':{'choices':[{'message':{'content':content,'reasoning':reasoning}}]}}


class FailureHandlingTests(unittest.TestCase):
    def test_sampling_correction_preserves_legacy_pipeline_and_omitted_caps(self):
        with patch.object(reader,'SAMPLING_POLICY','legacy'):
            old=reader.config('http://127.0.0.1:1')
        with patch.object(reader,'SAMPLING_POLICY','qwen35-thinking'):
            corrected=reader.config('http://127.0.0.1:1')
            payload={'temperature':corrected['llm']['temperature'], **corrected['llm']['extra_body']}
            reader.validate_request(payload)
            self.assertNotIn('max_tokens',payload)
            self.assertNotIn('thinking_budget',payload)
            self.assertEqual(corrected['llm']['max_tokens'],old['llm']['max_tokens'])
            self.assertIsNone(corrected['llm']['max_tokens'])
            self.assertEqual(corrected['compact']['summary_temperature'],1.0)
            self.assertEqual({k:v for k,v in corrected['compact'].items() if k!='summary_temperature'},
                             {k:v for k,v in old['compact'].items() if k!='summary_temperature'})
            self.assertEqual({k:v for k,v in corrected.items() if k not in ('llm','compact')},
                             {k:v for k,v in old.items() if k not in ('llm','compact')})
            for key in reader.RECOMMENDED_SAMPLING:
                payload[key]=99
                with self.assertRaises(AssertionError):reader.validate_request(payload)
                payload[key]=reader.RECOMMENDED_SAMPLING[key]

    def test_valid_content_is_preserved_exactly(self):
        text='  Valid memory.\n'
        self.assertEqual(reader.checked_content(response(text),'summary',Path('request.json')),text)

    def test_reasoning_is_never_accepted_as_summary_or_answer(self):
        for stage in ['summary','answer']:
            with self.subTest(stage=stage), self.assertRaises(reader.ModelGenerationFailure) as error:
                reader.checked_content(response(),'summary' if stage=='summary' else 'answer',Path('r'))
            self.assertEqual(error.exception.reason,stage+'_empty_final_content')

    def test_truncated_summary_stops_before_next_compaction(self):
        calls=[]
        def compact_then_continue():
            calls.append('first')
            memory=reader.checked_content(response('Partial summary',finish='length'),'summary',Path('r'))
            calls.append(memory)
        with self.assertRaises(reader.ModelGenerationFailure):compact_then_continue()
        self.assertEqual(calls,['first'])
        self.assertEqual(reader.checked_content(response('Partial answer',finish='length'),'answer',Path('r')),'Partial answer')

    def test_incomplete_stream_is_not_scored_as_model_failure(self):
        with self.assertRaises(RuntimeError):
            reader.checked_content(response('partial',finish=None),'summary',Path('r'))

    def test_resume_uses_first_failed_request_and_keeps_denominator(self):
        with tempfile.TemporaryDirectory() as name:
            out=Path(name);sample={'question_id':'x','question_type':'single-session-user'}
            d=out/'trials/x/compact_rag';d.mkdir(parents=True)
            p=d/'summary_00.response.json';reader.save(p,response(finish='length'))
            # A downstream interrupted attempt must not erase the original failure.
            reader.save(d/'summary_01.response.json',response(finish=None))
            failure=reader.previous_model_failure(d)
            self.assertEqual(failure.response_path,p)
            failed=reader.failure_result(sample,failure);reader.save(d/'result.json',failed)
            reader.save(out/'trials/x/kvmem/result.json',{'status':'completed','answer':'correct'})
            jobs=scorer.terminal_jobs(out,[sample])
            self.assertEqual(len(jobs),2)
            self.assertEqual(sum(j['pipeline_failure'] for j in jobs),1)
            self.assertEqual(next(j['answer'] for j in jobs if j['pipeline_failure']),'')
            # No model response can be silently edited under a cached failed score.
            reader.save(p,response('valid'))
            with self.assertRaises(AssertionError):scorer.terminal_jobs(out,[sample])


if __name__=='__main__':unittest.main()
