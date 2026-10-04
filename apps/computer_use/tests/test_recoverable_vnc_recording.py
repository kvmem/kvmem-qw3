"""Actual local RFB connections, timeout/partial-frame faults and codec evidence."""
import json
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import recoverable_vnc_recording as q
from test_resizable_vnc_recording import RFB, raw, resize, update


class MultiRFB:
    def __init__(self, plans):
        self.plans, self.requests, self.errors, self.accepted = plans, [], [], 0
        self.stop = threading.Event()
        self.listener = socket.socket(); self.listener.bind(("127.0.0.1", 0)); self.listener.listen(5)
        self.listener.settimeout(.05)
        self.port = self.listener.getsockname()[1]
        self.worker = threading.Thread(target=self.serve, daemon=True)

    def __enter__(self):
        self.worker.start(); return self

    def serve(self):
        for size, replies in self.plans:
            while not self.stop.is_set():
                try:
                    sock, _ = self.listener.accept(); break
                except socket.timeout: continue
                except OSError: return
            else: return
            self.accepted += 1
            try:
                with sock:
                    sock.settimeout(1); sock.sendall(b"RFB 003.008\n"); RFB.read(sock, 12)
                    sock.sendall(b"\x01\x01"); RFB.read(sock, 1); sock.sendall(b"\0"*4); RFB.read(sock, 1)
                    sock.sendall(struct.pack(">HH", *size)+b"\0"*16+struct.pack(">I", 3)+b"cpu")
                    index = 0
                    while not self.stop.is_set():
                        kind = RFB.read(sock, 1)[0]
                        if kind == 0:
                            RFB.read(sock, 19); self.requests.append(kind)
                        elif kind == 2:
                            count = struct.unpack(">H", RFB.read(sock, 3)[1:])[0]
                            RFB.read(sock, count*4); self.requests.append(kind)
                        elif kind == 3:
                            RFB.read(sock, 9); self.requests.append(kind)
                            reply = replies[min(index, len(replies)-1)]; index += 1
                            if reply == "stall": continue
                            if isinstance(reply, tuple):
                                kind, data = reply
                                if kind == "partial": sock.sendall(data); break
                                if kind == "drip":
                                    for byte in data:
                                        if self.stop.wait(.04): break
                                        sock.sendall(bytes((byte,)))
                                else: raise AssertionError("Unknown scripted reply")
                            else: sock.sendall(reply)
                        else: raise AssertionError(f"Input message {kind}")
            except (EOFError, ConnectionError, socket.timeout): pass
            except BaseException as exc: self.errors.append(repr(exc))

    def __exit__(self, *_):
        self.stop.set(); self.listener.close(); self.worker.join(2)
        if self.worker.is_alive(): raise AssertionError("Owned scripted server did not retire")
        if self.errors: raise AssertionError(self.errors)
        if any(k not in (0, 2, 3) for k in self.requests): raise AssertionError("Input was sent")


class RecoveryTests(unittest.TestCase):
    def client(self, server, **options):
        args = dict(owner_check=lambda _: True, canvas=(8, 6), max_wait_seconds=.8,
                    io_timeout=.05, reconnect_budget=1)
        args.update(options)
        return q.RecoverableConsole(server.port, "127.0.0.1", **args)

    def test_header_timeout_reconnects_to_full_new_geometry(self):
        captures = []
        with MultiRFB([((2, 2), [raw(2, 2, (255, 0, 0)), "stall"]),
                       ((6, 4), [raw(6, 4, (0, 255, 0))])]) as server:
            client = self.client(server, on_capture=captures.append)
            try:
                self.assertEqual(client.screenshot().getpixel((1, 1)), (255, 0, 0))
                image = client.screenshot()
                self.assertEqual(image.getpixel((5, 3)), (0, 255, 0))
                self.assertEqual(image.getpixel((7, 5)), (0, 0, 0))
                self.assertEqual([c["capture"] for c in captures], [0, 1])
                self.assertEqual([c["connection"] for c in captures], [1, 2])
                self.assertEqual(captures[1]["framebuffer"], [6, 4])
                self.assertEqual(client.reconnects, 1)
                self.assertTrue(client.gaps[0]["recovered"])
                self.assertFalse(client.gaps[0]["held_frames_are_new_observations"])
                self.assertTrue(all(c["full_framebuffer_received"] for c in captures))
            finally: client.close()

    def test_partial_pixels_are_discarded_before_reconnect(self):
        captures = []
        with MultiRFB([((2, 2), [("partial", raw(2, 2, (255, 0, 0))[:18])]),
                       ((2, 2), [raw(2, 2, (0, 0, 255))])]) as server:
            client = self.client(server, on_capture=captures.append)
            try:
                image = client.screenshot()
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))
                self.assertEqual(len(captures), 1)
                self.assertEqual(client.gaps[0]["first_error_type"], "EOFError")
                self.assertFalse(client.gaps[0]["missing_pixels_reconstructed"])
            finally: client.close()

    def test_lifetime_budget_is_not_reset_after_successful_recovery(self):
        with MultiRFB([((2, 2), ["stall"]), ((2, 2), [raw(2, 2, (0, 255, 0)), "stall"])]) as server:
            client = self.client(server)
            try:
                client.screenshot()
                with self.assertRaisesRegex(TimeoutError, "Lifetime"):
                    client.screenshot()
                self.assertEqual(client.reconnects, 1)
                self.assertEqual([g["recovered"] for g in client.gaps], [True, False])
                self.assertEqual(server.accepted, 2)
            finally: client.close()

    def test_zero_reconnect_budget_preserves_failed_gap(self):
        captures = []
        with MultiRFB([((2, 2), ["stall"])]) as server:
            client = self.client(server, reconnect_budget=0, on_capture=captures.append)
            try:
                with self.assertRaises(TimeoutError): client.screenshot()
                self.assertEqual(captures, [])
                self.assertEqual(client.reconnects, 0)
                self.assertFalse(client.gaps[0]["recovered"])
            finally: client.close()

    def test_identity_change_prevents_a_reconnect(self):
        checks = []
        def owner(_):
            checks.append(True); return len(checks) < 3
        with MultiRFB([((2, 2), ["stall"]), ((2, 2), [raw(2, 2, (0, 255, 0))])]) as server:
            client = self.client(server, owner_check=owner)
            try:
                with self.assertRaisesRegex(RuntimeError, "identity"): client.screenshot()
                self.assertEqual(server.accepted, 1)
                self.assertEqual(client.reconnects, 0)
                self.assertFalse(client.gaps[0]["recovered"])
            finally: client.close()

    def test_identity_change_after_frame_rejects_its_pixels(self):
        checks, captures = [], []
        def owner(_):
            checks.append(True); return len(checks) < 3
        with MultiRFB([((2, 2), [raw(2, 2, (0, 255, 0))])]) as server:
            client = self.client(server, owner_check=owner, on_capture=captures.append)
            try:
                with self.assertRaises(RuntimeError): client.screenshot()
                self.assertEqual(captures, [])
                self.assertEqual(client.captures, 0)
            finally: client.close()

    def test_protocol_errors_are_not_retried(self):
        for reply in (resize(9, 4), update([(0, 0, 2, 2, 7, None)]),
                      update([(0, 0, 1, 1, 0, (255, 0, 0))])):
            with self.subTest(reply=reply[:16]), MultiRFB([((2, 2), [reply])]) as server:
                client = self.client(server)
                try:
                    with self.assertRaises(RuntimeError): client.screenshot()
                    self.assertEqual(client.reconnects, 0)
                    self.assertEqual(client.captures, 0)
                finally: client.close()

    def test_slow_partial_reads_cannot_extend_absolute_deadline(self):
        with MultiRFB([((2, 2), [("drip", raw(2, 2, (255, 0, 0)))])]) as server:
            client = self.client(server, max_wait_seconds=.25, io_timeout=.08, reconnect_budget=0)
            try:
                started = time.monotonic()
                with self.assertRaises(TimeoutError): client.screenshot()
                self.assertLess(time.monotonic() - started, .7)
                self.assertEqual(client.captures, 0)
            finally: client.close()

    def test_stop_during_gap_does_not_claim_a_fresh_frame(self):
        stop = threading.Event()
        with MultiRFB([((2, 2), ["stall"])]) as server:
            client = self.client(server, stop=stop, io_timeout=.1)
            timer = threading.Timer(.04, stop.set); timer.start()
            try:
                with self.assertRaises(InterruptedError): client.screenshot()
                self.assertEqual(client.captures, 0)
                self.assertFalse(client.gaps[0]["recovered"])
            finally: client.close(); timer.join()

    def test_invalid_configuration_and_changed_source_fail_before_network(self):
        with patch.object(q.socket, "create_connection") as connect:
            for options in ({"canvas": (True, 2)}, {"max_wait_seconds": float("inf")},
                            {"reconnect_budget": True}, {"owner_check": None}):
                with self.assertRaises(ValueError):
                    q.RecoverableConsole(1, "127.0.0.1", **({"owner_check": lambda _: True} | options))
            with patch.dict(q.SOURCE_SHA256, {"resizable_vnc_recording.py": "0"*64}):
                with self.assertRaises(ValueError):
                    q.RecoverableConsole(1, "127.0.0.1", owner_check=lambda _: True)
            connect.assert_not_called()

    def test_owned_endpoint_rejects_same_id_restart_and_foreign_labels(self):
        info = {"Id": "a"*64, "Name": "/qw3-test", "Image": "sha256:"+"b"*64,
                "Config": {"Labels": {"qw3.goal": "cross-device-20261003"}},
                "State": {"Running": True, "StartedAt": "first"},
                "NetworkSettings": {"Networks": {"bridge": {"IPAddress": "172.17.0.8"}}}}
        guard = q.OwnedEndpoint("qw3-test", "a"*64, "172.17.0.8")
        with patch.object(q.subprocess, "check_output", return_value=json.dumps([info])):
            self.assertTrue(guard(time.monotonic()+1))
            info["State"]["StartedAt"] = "second"
        with patch.object(q.subprocess, "check_output", return_value=json.dumps([info])):
            with self.assertRaises(RuntimeError): guard(time.monotonic()+1)
        info["Config"]["Labels"]["qw3.goal"] = "foreign"
        with patch.object(q.subprocess, "check_output", return_value=json.dumps([info])):
            with self.assertRaises(RuntimeError): guard(time.monotonic()+1)

    def test_real_codec_retains_expanded_pixels_and_declares_stale_interval(self):
        import av
        import imageio_ffmpeg
        with tempfile.TemporaryDirectory() as folder, MultiRFB([
            ((2, 2), [raw(2, 2, (255, 0, 0)), "stall"]),
            ((6, 4), [raw(6, 4, (0, 255, 0))])]) as server:
            output = Path(folder)/"video"
            result = q.record(output, imageio_ffmpeg.get_ffmpeg_exe(), owner_check=lambda _: True,
                canvas=(8, 6), port=server.port, fps=10, duration=.45, io_timeout=.05, max_wait_seconds=.8)
            self.assertTrue(result["encoding_complete"])
            self.assertFalse(result["fresh_sampling_uninterrupted"])
            self.assertFalse(result["full_task_continuous_observation_proven"])
            self.assertFalse(result["qualified_for_task_latency_comparison"])
            self.assertEqual(len(result["transport_gaps"]), 1)
            self.assertTrue(result["transport_gaps"][0]["recovered"])
            captures = [json.loads(line) for line in (output/"framebuffer-geometry.jsonl").read_text().splitlines()]
            slots = [json.loads(line) for line in (output/"capture.jsonl").read_text().splitlines()]
            gap = result["transport_gaps"][0]
            fresh = slots[gap["fresh_capture"]]
            self.assertGreaterEqual(fresh["captured_monotonic_ns"], gap["until_monotonic_ns"])
            self.assertEqual(captures[1]["framebuffer"], [6, 4])
            with av.open(str(output/"screen.mp4")) as video:
                frames = list(video.decode(video=0))
                self.assertTrue(all((f.width, f.height)==(8, 6) for f in frames))
                self.assertTrue(any(f.to_ndarray(format="rgb24")[3, 5, 1]>180 for f in frames))
            with self.assertRaises(FileExistsError):
                q.record(output, imageio_ffmpeg.get_ffmpeg_exe(), owner_check=lambda _: True)

    def test_failed_protocol_recording_cannot_claim_uninterrupted_sampling(self):
        import imageio_ffmpeg
        with tempfile.TemporaryDirectory() as folder, MultiRFB([
            ((2, 2), [update([(0, 0, 2, 2, 7, None)])])]) as server:
            output = Path(folder)/"video"
            with self.assertRaises(RuntimeError):
                q.record(output, imageio_ffmpeg.get_ffmpeg_exe(), owner_check=lambda _: True,
                    canvas=(8, 6), port=server.port, fps=10, duration=.4, io_timeout=.05, max_wait_seconds=.8)
            metadata = json.loads((output/"recoverable-recording.json").read_text())
            self.assertFalse(metadata["encoding_complete"])
            self.assertFalse(metadata["fresh_sampling_uninterrupted"])
            self.assertFalse(metadata["geometry_validated_per_capture"])
            self.assertFalse(metadata["qualified_for_task_latency_comparison"])


if __name__ == "__main__": unittest.main()
