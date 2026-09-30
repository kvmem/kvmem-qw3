import json
from pathlib import Path
import tempfile
import unittest

from windows_long_task import inspect_retrieval


class LongEvidenceTests(unittest.TestCase):
    def test_same_length_neighbor_and_alternate_mask_do_not_qualify_answer(self):
        history = {"backend_trace_tag": "answer", "prompt_tokens": 150000}
        meta = {"type": "meta", "trace_tag": "answer", "prompt_tokens": 150000, "mask": -1,
                "selected": 1024, "block_count": 1200, "scorer_fallback_reason": "none",
                "query_ready": True, "index_ready": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retrieval.jsonl"
            for altered in ({"trace_tag": "neighbor"}, {"prompt_tokens": 150001}, {"mask": 0}):
                path.write_text(json.dumps(meta | altered) + "\n")
                with self.assertRaisesRegex(AssertionError, "Missing retrieval evidence"):
                    inspect_retrieval(path, history)
            # A later unrelated record must not displace the answer's record.
            path.write_text(json.dumps(meta) + "\n" + json.dumps(meta | {"trace_tag": "neighbor"}) + "\n")
            self.assertEqual(inspect_retrieval(path, history), meta)
            for altered in ({"query_ready": False}, {"index_ready": False},
                            {"selected": 1200}, {"scorer_fallback_reason": "unavailable"}):
                path.write_text(json.dumps(meta | altered) + "\n")
                with self.assertRaises(AssertionError):
                    inspect_retrieval(path, history)
