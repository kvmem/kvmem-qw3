import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from trace_io import iter_segments, read_request, expected_decode_queries


class TraceReaderTest(unittest.TestCase):
    def test_eos_and_output_cap_have_different_query_counts(self):
        self.assertEqual(expected_decode_queries(17,32),17)
        self.assertEqual(expected_decode_queries(32,32),31)
        self.assertEqual(expected_decode_queries(1,1),0)
        with self.assertRaises(ValueError):expected_decode_queries(33,32)

    def rows(self, sample, position):
        return [dict(kind='attention_mass',mode='global',sample=sample,position=position,
            layer=layer,block_tokens=128,block_ids=[0,1,-1],mass=mass)
            for layer,mass in [(3,[1,3,0]),(7,[3,1,0])]]

    def test_request_byte_interval_and_layer_average(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'trace.jsonl';a=self.rows(0,127)+self.rows(1,128)
            prefix=''.join(json.dumps(r)+'\n' for r in a).encode()
            suffix=''.join(json.dumps(r)+'\n' for r in self.rows(0,300)).encode()
            p.write_bytes(prefix+suffix)
            segs=list(iter_segments(p));self.assertEqual(len(segs),2)
            self.assertEqual(segs[0]['positions'].tolist(),[127,128])
            np.testing.assert_allclose(segs[0]['mass'],[[.5,.5],[.5,.5]])
            cut=list(iter_segments(p,len(prefix),len(prefix+suffix)))
            self.assertEqual(len(cut),1);self.assertEqual(cut[0]['positions'].tolist(),[300])

    def test_missing_layer_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'trace.jsonl'
            rows=self.rows(0,127)+self.rows(1,128)[:1]
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'inconsistent layers'):list(iter_segments(p))

    def test_layer_major_prefill_then_token_major_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'trace.jsonl'
            prefill=self.rows(254,254)+self.rows(255,255)
            prefill.sort(key=lambda row:row['layer'])
            rows=prefill+self.rows(0,256)
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            d=read_request(p,0,p.stat().st_size)
            self.assertEqual(d['positions'].tolist(),[254,255,256])
            np.testing.assert_allclose(d['mass'],np.full((3,2),.5))
            p.write_text(p.read_text()+json.dumps(rows[-1])+'\n')
            with self.assertRaisesRegex(ValueError,'duplicate layer/position'):
                read_request(p,0,p.stat().st_size)


if __name__=='__main__':unittest.main()
