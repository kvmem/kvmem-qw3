import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import prepare_dataset
from audit_new import collection_digest


class SourceIdentityTest(unittest.TestCase):
    def test_retried_software_task_is_one_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for suffix in ['a/task-x/attempt_02/pier_jobs/task-x/trial/agent',
                           'b/task-x/pier_jobs/task-x/trial/agent',
                           'c/task-y/attempt_02/pier_jobs/task-y/trial/agent']:
                p=root/'benchmark/mini_swe_deepswe/recordings'/suffix
                p.mkdir(parents=True);(p/'mini-swe-agent.trajectory.json').write_text('{"messages":[]}')
            with patch.object(prepare_dataset,'ROOT',root), patch.object(prepare_dataset,'replay_cuts',return_value=[{}]*12):
                selected,count=prepare_dataset.prepare_code(20,12,random.Random(0))
            self.assertEqual(count,2)
            self.assertEqual({r['independent_unit'] for r in selected},{'task-x','task-y'})

    def test_lineage_allows_label_correction_but_not_frame_or_protocol_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'origin.json'
            protocol=dict(seed=1,model='m',kv_dtype='fp16',ctx=65536,max_decode_tokens=2048,
                          prefill_query_tokens=64,thinking_budget=512,prefill_instrumentation='batched')
            row=dict(id='code_attempt_02',domain='code',independent_unit='attempt_02',
                     source='/root/task-x/attempt_02/pier_jobs/task-x/trial',frames=[{'cut':3}])
            dialogue=dict(id='dialogue_1',domain='dialogue',independent_unit='session-hash',
                          source='/data/dialogue.json',frames=[{'cut':3}])
            p.write_text(json.dumps(protocol|{'trajectories':[row,dialogue]}))
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            primary=protocol|{'collection_manifests':{'original':{'path':str(p),'sha256':digest}}}
            corrected=row|{'independent_unit':'task-x','collection_cohort':'original'}
            self.assertEqual(collection_digest(corrected,primary,'new-hash',{}),digest)
            with self.assertRaisesRegex(AssertionError,'independence unit'):
                collection_digest(dialogue|{'collection_cohort':'original','independent_unit':'other'},primary,'new-hash',{})
            bad=copy.deepcopy(corrected);bad['frames'][0]['cut']=4
            with self.assertRaisesRegex(AssertionError,'source/frame'):
                collection_digest(bad,primary,'new-hash',{})
            with self.assertRaisesRegex(AssertionError,'protocol'):
                collection_digest(corrected,primary|{'ctx':32768},'new-hash',{})
            p.write_text(p.read_text()+' ')
            with self.assertRaisesRegex(AssertionError,'checksum'):
                collection_digest(corrected,primary,'new-hash',{})


if __name__=='__main__':unittest.main()
