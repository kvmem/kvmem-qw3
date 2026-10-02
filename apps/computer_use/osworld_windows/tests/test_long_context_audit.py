"""Host-only artifact-audit controls; fixtures are never actual-model results."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from osworld_windows.grade_long_context import grade


class LongContextAuditTests(unittest.TestCase):
    def fixture(self, root, compact_tokens=960000):
        directory = root / "long-context-test"
        native = directory / "native"
        native.mkdir(parents=True)
        engine = dict(kvmem="on", binary_sha256="fixture-not-a-real-engine")
        (root / "engine").mkdir()
        (root / "engine/manifest.json").write_text(json.dumps(engine))
        (root / "engine/retrieval.jsonl").write_text(json.dumps(dict(type="meta", trace_tag="compact", prompt_tokens=compact_tokens)) + "\n")
        markers = [hashlib.sha256((directory.name + ":" + str(index)).encode()).hexdigest()[:16] for index in (1, 2)]
        (native / "observations.jsonl").write_text("".join(json.dumps(dict(sequence=i, marker=m)) + "\n" for i, m in enumerate(markers, 1)))
        (native / "events.jsonl").write_text(json.dumps(dict(method="item/completed", params=dict(item=dict(type="contextCompaction")))) + "\n")
        (native / "result.json").write_text(json.dumps(dict(real_long_context_inference=True, initial_health=dict(platform="windows"),
            answer=dict(observations=2, first_marker=markers[0], latest_marker=markers[1]), turn=dict(status="completed"))))
        (directory / "result.json").write_text(json.dumps(dict(engine=engine, observations_requested=2,
            launch_receipt=dict(started=True), recording=dict(complete=True), final_health=dict(paused=True))))
        receipts = []
        for name, started, tokens, tools, text in [("before", 1, 950100, [{"function": {"name": "observe"}}], "Observe"),
            ("compact", 2, compact_tokens, [], "You are performing a CONTEXT CHECKPOINT COMPACTION."),
            ("after", 3, 5000, [{"function": {"name": "observe"}}], "Continue")]:
            body = dict(kvmem_trace_tag=name, tools=tools, messages=[dict(role="user", content=text)])
            trace = native / "guest-runtime/inference" / name
            trace.mkdir(parents=True)
            (trace / "request.json").write_text(json.dumps(body))
            (trace / "metrics.json").write_text(json.dumps(dict(started_at=started, status="completed", usage=dict(input_tokens=tokens))))
            canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
            receipts.append(dict(event="request_terminal", backend_trace_tag=name, status="complete",
                                 canonical_request_sha256=hashlib.sha256(canonical).hexdigest()))
        (directory / "relay").mkdir()
        (directory / "relay/requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in receipts))
        return directory

    def test_contract_control_and_small_compaction_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self.fixture(Path(tmp))
            self.assertTrue(grade(directory)["passed"])
        with tempfile.TemporaryDirectory() as tmp:
            directory = self.fixture(Path(tmp), compact_tokens=500)
            result = grade(directory)
            self.assertFalse(result["passed"])
            self.assertFalse(result["checks"]["near_1m_compaction_request"])

    def test_changed_request_cannot_reuse_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self.fixture(Path(tmp))
            path = directory / "native/guest-runtime/inference/compact/request.json"
            body = json.loads(path.read_text()); body["temperature"] = 1
            path.write_text(json.dumps(body))
            self.assertFalse(grade(directory)["checks"]["all_backend_requests_bound"])

    def test_usage_disagreement_with_engine_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self.fixture(Path(tmp))
            (Path(tmp) / "engine/retrieval.jsonl").write_text(json.dumps(dict(type="meta", trace_tag="compact", prompt_tokens=9000)) + "\n")
            self.assertFalse(grade(directory)["checks"]["compaction_usage_matches_engine"])

    def test_lost_marker_is_not_successful_compaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self.fixture(Path(tmp))
            path = directory / "native/result.json"
            native = json.loads(path.read_text()); native["answer"]["first_marker"] = None
            path.write_text(json.dumps(native))
            self.assertFalse(grade(directory)["checks"]["summary_retains_first_and_latest"])


if __name__ == "__main__":
    unittest.main()
