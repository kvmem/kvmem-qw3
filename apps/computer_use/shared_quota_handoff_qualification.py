"""CPU-only terminal-route qualification for the unchanged live successors."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import unittest


def run(destination):
    import shared_quota_successors as adapter
    q = adapter.RUNTIME
    source_root = Path(__file__).parent
    names = ("shared_quota_handoff_qualification.py", "tests/test_shared_quota_terminal_handoff.py",
             "shared_quota_successors.py", "shared_quota_memory.py", "runner_runtime_successors.py",
             "retired_051_successors.py", "execution_retest_queue.py", "execution_retest_queue_v2.py",
             "visual_context_retest_queue.py", "kvmem_osworld_queue.py")
    destination.mkdir(parents=True, exist_ok=False)
    pins = {}
    for name in names:
        source = source_root / name
        target = destination / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        pins[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    suite = unittest.defaultTestLoader.discover(str(source_root / "tests"), pattern="test_shared_quota_terminal_handoff.py")
    transcript = io.StringIO()
    result = unittest.TextTestRunner(stream=transcript, verbosity=2).run(suite)
    (destination / "tests.log").write_text(transcript.getvalue())
    unchanged = all(hashlib.sha256((source_root / name).read_bytes()).hexdigest() == digest for name, digest in pins.items())
    report = dict(kind="shared_quota_terminal_handoff_cpu_qualification", checked_at=time.time(),
                  passed=result.wasSuccessful() and unchanged and result.testsRun >= 12 and not result.skipped,
                  tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  source_sha256=pins, sources_unchanged=unchanged, model_calls=0, device_inputs=0,
                  scope="Immutable SDK/image successful terminal callbacks, fresh eight/ten receipt validation, negative inventory and ownership checks. Static admission/historical forty/host probes are explicit CPU fixtures; no business gain claimed.")
    (destination / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output.resolve())
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)
