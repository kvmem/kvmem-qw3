"""Graceful stopping must not suppress an unrelated capture failure."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import graceful_recoverable_vnc_recording as q
from test_recoverable_vnc_recording import MultiRFB, raw


class GracefulNegativeTests(unittest.TestCase):
    def test_interrupted_capture_without_stop_event_remains_failed(self):
        import imageio_ffmpeg
        with tempfile.TemporaryDirectory() as folder, MultiRFB([((8, 6), [raw(8, 6, (255, 0, 0))])]) as server:
            output = Path(folder) / "video"
            with patch.object(q.GracefulStopConsole, "screenshot", side_effect=InterruptedError("unrelated capture error")):
                with self.assertRaisesRegex(InterruptedError, "unrelated"):
                    q.record(output, imageio_ffmpeg.get_ffmpeg_exe(), owner_check=lambda _: True,
                             host="127.0.0.1", port=server.port, duration=1, fps=2,
                             stop=threading.Event(), max_wait_seconds=3, reconnect_budget=0)
            receipt = json.loads((output / "recoverable-recording.json").read_text())
            self.assertFalse(receipt["encoding_complete"])
            self.assertEqual(receipt["validated_complete_captures"], 0)
            self.assertFalse(receipt["graceful_stop_discarded_capture"])
            self.assertFalse(receipt["fresh_sampling_uninterrupted"])

    def test_changed_base_source_rejected_before_endpoint_or_encoder(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(q, "BASE_SHA256", "0" * 64), patch.object(q, "ORIGINAL_RECORD") as invoke:
            with self.assertRaisesRegex(ValueError, "source changed"):
                q.record(Path(folder) / "video", "/unused/encoder", owner_check=lambda _: True)
            invoke.assert_not_called()
            self.assertFalse((Path(folder) / "video").exists())


if __name__ == "__main__":
    unittest.main()
