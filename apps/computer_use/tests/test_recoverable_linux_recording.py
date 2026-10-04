"""Task wrapper with actual subprocess/RFB/codec and a local Docker fixture.

The fixture never calls Docker, connects a VM, requests a model or sends input.
"""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import av
import recoverable_linux_recording as q
from test_recoverable_vnc_recording import MultiRFB, raw


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "task"
        self.output.mkdir()
        self.container = "c" * 64
        self.info = {"Id": self.container, "Name": "/qw3-test-recording", "Image": "fixed-image",
                     "Config": {"Labels": {"qw3.goal": "cross-device-20261003"}},
                     "State": {"Running": True, "StartedAt": "fixed-boot"},
                     "NetworkSettings": {"Networks": {"bridge": {"IPAddress": "127.0.0.1"}}}}
        self.write_info()
        docker = self.root / "docker"
        docker.write_text("#!/usr/bin/env python3\nimport pathlib,sys\n"
                          "assert sys.argv[1:]==['inspect','qw3-test-recording']\n"
                          "print((pathlib.Path(__file__).parent/'info.json').read_text())\n")
        docker.chmod(0o700)
        self.env = patch.dict(os.environ, {"PATH": str(self.root) + os.pathsep + os.environ["PATH"]})
        self.env.start()
        self.addCleanup(self.env.stop)

    def write_info(self):
        (self.root / "info.json").write_text(json.dumps([self.info]))

    def wrapper(self, server, **options):
        return q.RecoverableLinuxRecording(self.output, self.container, "qw3-test-recording",
                  port=server.port, profile=q.RecordingProfile(**({"duration_seconds": 2,
                      "startup_seconds": 3, "max_wait_seconds": 5, "reconnect_budget": 1} | options)))

    def movie(self):
        with av.open(str(self.output / "continuous.mp4")) as source:
            return list(source.decode(video=0))

    def test_registered_worker_full_movie_and_repeat_finish_share_one_source(self):
        with MultiRFB([((8, 6), [raw(8, 6, (255, 0, 0))])]) as server:
            wrapper = self.wrapper(server)
            launch = wrapper.start()
            self.assertTrue(wrapper.first_frame_ready())
            self.assertEqual(launch["identity"]["command"][2], "graceful_recoverable_vnc_recording")
            wrapper.worker.wait(timeout=10)
            receipt = wrapper.finish()
            self.assertTrue(receipt["encoding_complete"])
            self.assertTrue(receipt["fresh_sampling_uninterrupted"])
            self.assertFalse(receipt["qualified_for_task_latency_comparison"])
            source, target = self.output / "video/screen.mp4", self.output / "continuous.mp4"
            self.assertEqual(source.stat().st_ino, target.stat().st_ino)
            self.assertEqual(wrapper.finish(), receipt)
            frames = self.movie()
            self.assertEqual(len(frames), receipt["video_frames"])
            self.assertGreater(frames[-1].to_ndarray(format="rgb24")[1, 1, 0], 160)
            self.assertTrue(all(kind in (0, 2, 3) for kind in server.requests))

    def test_worker_reconnect_does_not_promote_gap_to_fresh_sampling(self):
        with MultiRFB([((8, 6), [raw(8, 6, (255, 0, 0)), "stall"]),
                       ((8, 6), [raw(8, 6, (0, 0, 255))])]) as server:
            wrapper = self.wrapper(server, duration_seconds=6)
            wrapper.start()
            wrapper.worker.wait(timeout=15)
            receipt = wrapper.finish()
            self.assertTrue(receipt["encoding_complete"])
            self.assertFalse(receipt["fresh_sampling_uninterrupted"])
            self.assertEqual(receipt["reconnects"], 1)
            gaps = receipt["publication_evidence"]["video_observation_gaps"]
            self.assertEqual(len(gaps), 1)
            self.assertTrue(gaps[0]["displayed_frames_are_stale"])
            self.assertFalse(receipt["full_task_continuous_observation_proven"])
            self.assertGreater(self.movie()[-1].to_ndarray(format="rgb24")[1, 1, 2], 160)

    def test_sigterm_during_half_frame_encodes_tail_without_minting_capture(self):
        with MultiRFB([((8, 6), [raw(8, 6, (255, 0, 0)), ("drip", raw(8, 6, (0, 0, 255)))])]) as server:
            wrapper = self.wrapper(server, duration_seconds=30)
            wrapper.start()
            time.sleep(.7)
            receipt = wrapper.finish()
            self.assertTrue(receipt["encoding_complete"])
            self.assertTrue(receipt["graceful_stop_discarded_capture"])
            self.assertEqual(receipt["validated_complete_captures"], 1)
            self.assertFalse(receipt["fresh_sampling_uninterrupted"])
            gaps = receipt["publication_evidence"]["video_observation_gaps"]
            self.assertEqual(len(gaps), 1)
            self.assertFalse(gaps[0]["recovered"])
            self.assertTrue(gaps[0]["displayed_frames_are_stale"])
            pixels = self.movie()[-1].to_ndarray(format="rgb24")[1, 1]
            self.assertGreater(pixels[0], 160)
            self.assertLess(pixels[2], 80)
            captures = (self.output / "video/capture.jsonl").read_text().splitlines()
            self.assertEqual(len(captures), 1)

    def test_no_complete_frame_cannot_pass_start_and_owned_worker_retires(self):
        with MultiRFB([((8, 6), ["stall"])]) as server:
            wrapper = self.wrapper(server, startup_seconds=.5)
            with self.assertRaises(TimeoutError):
                wrapper.start()
            failure = json.loads((self.output / "recorder-start-failure.json").read_text())
            self.assertTrue(failure["owned_worker_retired"])
            self.assertIsNotNone(wrapper.worker.poll())
            self.assertFalse((self.output / "continuous.mp4").exists())

    def test_corrupted_terminal_media_rejected_on_first_and_repeat_finish(self):
        with MultiRFB([((8, 6), [raw(8, 6, (255, 0, 0))])]) as server:
            wrapper = self.wrapper(server)
            wrapper.start(); wrapper.worker.wait(timeout=10)
            with (self.output / "video/screen.mp4").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(RuntimeError, "evidence failed"):
                wrapper.finish()
            with self.assertRaisesRegex(RuntimeError, "Prior"):
                wrapper.finish()
            self.assertFalse((self.output / "continuous.mp4").exists())

    def test_container_reboot_after_recording_rejects_publication(self):
        with MultiRFB([((8, 6), [raw(8, 6, (255, 0, 0))])]) as server:
            wrapper = self.wrapper(server)
            wrapper.start(); wrapper.worker.wait(timeout=10)
            self.info["State"]["StartedAt"] = "replacement-boot"
            self.write_info()
            with self.assertRaisesRegex(RuntimeError, "restarted"):
                wrapper.finish()
            self.assertFalse((self.output / "continuous.mp4").exists())

    def test_existing_target_and_original_media_are_preserved(self):
        with MultiRFB([((8, 6), [raw(8, 6, (255, 0, 0))])]) as server:
            wrapper = self.wrapper(server)
            wrapper.start(); wrapper.worker.wait(timeout=10)
            (self.output / "continuous.mp4").write_bytes(b"other-owner")
            with self.assertRaises(FileExistsError):
                wrapper.finish()
            self.assertEqual((self.output / "continuous.mp4").read_bytes(), b"other-owner")
            self.assertGreater((self.output / "video/screen.mp4").stat().st_size, 100)

    def test_foreign_container_label_never_launches_recorder(self):
        self.info["Config"]["Labels"] = {}
        self.write_info()
        wrapper = q.RecoverableLinuxRecording(self.output, self.container, "qw3-test-recording")
        with self.assertRaisesRegex(RuntimeError, "endpoint changed"):
            wrapper.start()
        self.assertIsNone(wrapper.worker)
        self.assertFalse((self.output / "recorder.log").exists())

    def test_invalid_or_changed_profile_source_fails_before_launch(self):
        for values in ({"duration_seconds": float("nan")}, {"fps": True},
                       {"max_wait_seconds": 2}, {"reconnect_budget": -1}):
            with self.assertRaises(ValueError):
                q.RecordingProfile(**values)
        wrapper = q.RecoverableLinuxRecording(self.output, self.container, "qw3-test-recording")
        with patch.object(q, "RECORDER_SHA256", "0" * 64), patch.object(q.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "recorder changed"):
                wrapper.start()
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
