import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from computer_use_vision_profile import terminate_owned_group, validate_metrics


class VisionMetricTests(unittest.TestCase):
    def metrics(self):
        return dict(rows=2, dim=1024, weights_mib=10, repeats=1, gpu_mean_ms=1,
                    gpu_min_ms=1, cpu_mean_ms=5, cpu_min_ms=5, partial_hit_ms=1,
                    reordered_hit_ms=.01, mean_abs=.001, rmse=.001, max_abs=.01, cosine=.9999)

    def test_invalid_metrics_cannot_pass(self):
        validate_metrics(self.metrics())
        for key, value in [('cosine', float('nan')), ('gpu_mean_ms', -1),
                           ('partial_hit_ms', float('inf')), ('repeats', 3),
                           ('mean_abs', .03), ('cosine', .998), ('dim', True)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_metrics({**self.metrics(), key: value})

    def test_missing_timings_cannot_pass(self):
        values = self.metrics()
        del values['reordered_hit_ms']
        with self.assertRaises(ValueError):
            validate_metrics(values)


@unittest.skipUnless(sys.platform == 'linux', 'The GPU profiler uses Linux process groups')
class OwnedWorkerCleanupTests(unittest.TestCase):
    def check_group(self, ignore_parent_term):
        child_code = ('import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                      'print(os.getpid(),flush=True); time.sleep(60)')
        parent_code = 'import subprocess,sys,signal,time; '
        if ignore_parent_term:
            parent_code += 'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
        parent_code += f'subprocess.Popen([sys.executable,"-c",{child_code!r}]); time.sleep(60)'
        proc = subprocess.Popen([sys.executable, '-c', parent_code],
                                start_new_session=True, stdout=subprocess.PIPE, text=True)
        self.addCleanup(proc.stdout.close)
        self.addCleanup(terminate_owned_group, proc, .2)
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            self.assertTrue(selector.select(5), 'Child did not reach its ready signal')
        child_pid = int(proc.stdout.readline())
        self.assertEqual(os.getpgid(child_pid), proc.pid)
        terminate_owned_group(proc, .2)
        self.assertIsNotNone(proc.returncode)
        # An orphan can remain a zombie until the environment reaps it, but
        # it must not keep running or retain GPU/lock descriptors.
        def child_running():
            try:
                return Path(f'/proc/{child_pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
            except FileNotFoundError:
                return False
        deadline = time.monotonic() + 2
        while child_running() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(child_running())

    def test_worker_stops_when_parent_exits_on_term(self):
        self.check_group(False)

    def test_parent_and_worker_ignoring_term_are_killed(self):
        self.check_group(True)


if __name__ == '__main__':
    unittest.main()
