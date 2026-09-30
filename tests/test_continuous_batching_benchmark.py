"""Check benchmark request parameters without starting a model or HTTP service."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import continuous_batching_benchmark as bench


class Response:
    status = 200

    def read(self):
        return b'{"choices":[{"text":"ok"}],"usage":{"completion_tokens":1,"total_tokens":2}}'


class Connection:
    payloads = []

    def __init__(self, *args, **kwargs):
        pass

    def request(self, method, url, body, headers):
        self.payloads.append(json.loads(body))

    def getresponse(self):
        return Response()

    def close(self):
        pass


class SamplingParametersTest(unittest.TestCase):
    def setUp(self):
        Connection.payloads = []

    @patch.object(bench.http.client, "HTTPConnection", Connection)
    def test_default_remains_greedy(self):
        result = bench.post_completion("http://unused:1", "a", "prompt", 8, False, 1)
        self.assertTrue(result.ok)
        self.assertEqual(Connection.payloads[0]["temperature"], 0)

    @patch.object(bench.http.client, "HTTPConnection", Connection)
    def test_sampling_reaches_serial_and_concurrent_requests(self):
        sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 1000}
        for concurrent in (False, True):
            with self.subTest(concurrent=concurrent):
                Connection.payloads = []
                _, results = bench.send_requests(
                    "http://unused:1", {"a": "first", "b": "second"},
                    17, True, 1, concurrent, sampling)
                self.assertEqual(len(results), 2)
                self.assertTrue(all(r.ok for r in results))
                self.assertEqual(len(Connection.payloads), 2)
                for payload in Connection.payloads:
                    self.assertEqual({key: payload[key] for key in sampling}, sampling)
                    self.assertEqual(payload["max_tokens"], 17)
                    self.assertTrue(payload["ignore_eos"])


if __name__ == "__main__":
    unittest.main()
