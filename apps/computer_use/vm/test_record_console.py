"""Recorder timing/failure checks; run on the Linux VM host, not the Windows client."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from vm import record_console


class RecorderTests(unittest.TestCase):
    def test_slow_capture_never_backdates_future_screen(self):
        now = [0]
        encoded = []
        encoder = Mock()
        encoder.stdin.write.side_effect = encoded.append
        encoder.wait.return_value = 0
        encoder.poll.return_value = 0
        console = Mock(width=2, height=2)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        def capture():
            now[0] += 250_000_000  # slower than the 100 ms video interval
            return Image.new("RGB", (2, 2), colors.pop(0))

        console.screenshot.side_effect = capture
        stop = Mock()
        stop.is_set.return_value = False
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "encoder"
            executable.write_bytes(b"fixture")
            with patch.object(record_console, "Console", return_value=console), \
                 patch.object(record_console.time, "monotonic_ns", side_effect=lambda: now[0]), \
                 patch.object(record_console.subprocess, "Popen", return_value=encoder), \
                 patch.object(record_console.subprocess, "run"):
                result = record_console.record(Path(directory) / "recording", executable, 0.51, 10, stop=stop)
            # The first screen arrives at 0.25 s: frames 0.0–0.2 must
            # remain black, and only slots at/after 0.3 may show red.
            self.assertEqual(encoded, [bytes(12)] * 3 + [bytes((255, 0, 0)) * 4] * 2 +
                             [bytes((0, 255, 0)) * 4] * 3)
            self.assertEqual(result["elapsed_seconds"], 0.75)
            self.assertEqual(result["video_frames"], 8)
            rows = [json.loads(line) for line in (Path(directory) / "recording/capture.jsonl").read_text().splitlines()]
            self.assertEqual([row["available_from_frame"] for row in rows], [3, 5, 8])
        console.close.assert_called_once()

    def test_capture_failure_retains_explicit_failure_manifest(self):
        console = Mock(width=2, height=2)
        console.screenshot.side_effect = TimeoutError("VM screen stalled")
        encoder = Mock()
        encoder.poll.return_value = 0
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "encoder"
            executable.write_bytes(b"fixture")
            output = Path(directory) / "recording"
            with patch.object(record_console, "Console", return_value=console), \
                 patch.object(record_console.subprocess, "Popen", return_value=encoder):
                with self.assertRaisesRegex(TimeoutError, "screen stalled"):
                    record_console.record(output, executable, 1)
            result = json.loads((output / "recording.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["complete"])
            self.assertEqual(result["captures"], 0)
        console.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
