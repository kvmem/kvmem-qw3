"""Real local RFB transport and codec checks; no VM input or model requests."""
import json
from pathlib import Path
from unittest.mock import patch
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from PIL import Image
import resizable_vnc_recording as q


def update(rectangles):
    data = struct.pack(">BBH", 0, 0, len(rectangles))
    for x, y, w, h, encoding, color in rectangles:
        data += struct.pack(">HHHHi", x, y, w, h, encoding)
        if encoding == 0:
            r, g, b = color; data += bytes((b, g, r, 0))*w*h
    return data


def raw(width, height, color): return update([(0, 0, width, height, 0, color)])
def resize(width, height): return update([(0, 0, width, height, -223, None)])


class RFB:
    def __init__(self, size, replies):
        self.size, self.replies, self.requests, self.errors = size, replies, [], []
        self.listener = socket.socket(); self.listener.bind(("127.0.0.1", 0)); self.listener.listen(1)
        self.port = self.listener.getsockname()[1]; self.worker = threading.Thread(target=self.serve, daemon=True)
    def __enter__(self): self.worker.start(); return self
    @staticmethod
    def read(sock, count):
        data = b""
        while len(data) < count:
            chunk = sock.recv(count-len(data))
            if not chunk: raise EOFError()
            data += chunk
        return data
    def serve(self):
        try:
            with self.listener.accept()[0] as sock:
                sock.settimeout(3); sock.sendall(b"RFB 003.008\n"); self.read(sock, 12)
                sock.sendall(b"\x01\x01"); self.read(sock, 1); sock.sendall(b"\0"*4); self.read(sock, 1)
                sock.sendall(struct.pack(">HH", *self.size)+b"\0"*16+struct.pack(">I", 3)+b"cpu")
                index = 0
                while True:
                    kind = self.read(sock, 1)[0]
                    if kind == 0:
                        self.read(sock, 19); self.requests.append((kind, None))
                    elif kind == 2:
                        count = struct.unpack(">H", self.read(sock, 3)[1:])[0]
                        encodings = struct.unpack(">"+"i"*count, self.read(sock, 4*count))
                        self.requests.append((kind, encodings))
                    elif kind == 3:
                        fields = struct.unpack(">BHHHH", self.read(sock, 9)); self.requests.append((kind, fields))
                        sock.sendall(self.replies[min(index, len(self.replies)-1)]); index += 1
                    else: raise AssertionError(f"Input or undeclared client message {kind}")
        except (EOFError, ConnectionResetError, BrokenPipeError): pass
        except BaseException as exc: self.errors.append(repr(exc))
    def __exit__(self, *_):
        self.listener.close(); self.worker.join(4)
        if self.worker.is_alive(): raise AssertionError("Scripted RFB worker did not retire")
        if self.errors: raise AssertionError(self.errors)


class CaptureTests(unittest.TestCase):
    def test_resize_delivers_pixels_outside_old_rectangle_without_scaling(self):
        events = []
        with RFB((2, 2), [raw(2, 2, (255, 0, 0)), resize(6, 4), raw(6, 4, (0, 255, 0))]) as server:
            client = q.ResizableConsole(server.port, canvas=(8, 6), on_capture=events.append)
            try:
                first = client.screenshot(); second = client.screenshot()
                self.assertEqual(first.size, (8, 6)); self.assertEqual(first.getpixel((1, 1)), (255, 0, 0))
                self.assertEqual(first.getpixel((5, 3)), (0, 0, 0))
                self.assertEqual(second.getpixel((5, 3)), (0, 255, 0))
                self.assertEqual(second.getpixel((7, 5)), (0, 0, 0))
                self.assertEqual(events[1]["resize_notifications"], [[6, 4]])
                self.assertTrue(events[1]["desktop_size_extension_confirmed"])
                self.assertFalse(events[1]["pixel_scaling"])
                self.assertTrue(any(kind==2 and values==(0, -223) for kind, values in server.requests))
                self.assertTrue(all(kind in (0, 2, 3) for kind, _ in server.requests))
            finally: client.close()

    def test_shrink_clears_pixels_outside_new_framebuffer(self):
        with RFB((6, 4), [raw(6, 4, (0, 255, 0)), resize(2, 2), raw(2, 2, (0, 0, 255))]) as server:
            client = q.ResizableConsole(server.port, canvas=(8, 6))
            try:
                client.screenshot(); image = client.screenshot()
                self.assertEqual(image.getpixel((1, 1)), (0, 0, 255)); self.assertEqual(image.getpixel((5, 3)), (0, 0, 0))
            finally: client.close()

    def test_partial_raw_rectangles_must_cover_complete_framebuffer(self):
        left = update([(0, 0, 2, 3, 0, (255, 0, 0))]); right = update([(2, 0, 2, 3, 0, (0, 255, 0))])
        with RFB((4, 3), [left, right]) as server:
            client = q.ResizableConsole(server.port, canvas=(8, 6))
            try:
                image = client.screenshot(); self.assertEqual(image.getpixel((3, 2)), (0, 255, 0))
                self.assertEqual(len([r for r in server.requests if r[0]==3]), 2)
            finally: client.close()

    def rejection(self, reply, size=(2, 2)):
        events = []
        with RFB(size, [reply]) as server:
            client = q.ResizableConsole(server.port, canvas=(8, 6), on_capture=events.append)
            try:
                with self.assertRaises(RuntimeError): client.screenshot()
                self.assertEqual(events, [])
            finally: client.close()

    def test_oversize_resize_cannot_be_cropped(self): self.rejection(resize(10, 4))
    def test_zero_size_resize_is_rejected(self): self.rejection(resize(0, 4))
    def test_unannounced_out_of_bounds_rectangle_is_rejected(self): self.rejection(raw(6, 4, (0, 255, 0)))
    def test_unsupported_encoding_is_rejected(self): self.rejection(update([(0, 0, 2, 2, 7, None)]))
    def test_resize_not_last_in_update_is_rejected(self): self.rejection(update([(0, 0, 4, 3, -223, None), (0, 0, 2, 2, 0, (0, 0, 0))]))
    def test_empty_updates_are_bounded(self): self.rejection(update([]))
    def test_incomplete_screen_never_becomes_successful_capture(self): self.rejection(update([(0, 0, 1, 1, 0, (255, 0, 0))]))
    def test_excessive_rectangle_count_rejected_before_pixel_read(self): self.rejection(struct.pack(">BBH", 0, 0, 4097))

    def test_initial_framebuffer_larger_than_declared_canvas_is_rejected(self):
        with RFB((10, 4), [b""]) as server:
            with self.assertRaises(RuntimeError): q.ResizableConsole(server.port, canvas=(8, 6))

    def test_initial_oversize_is_rejected_before_image_allocation(self):
        with RFB((65535, 65535), [b""]) as server, patch.object(q.Image, "new") as allocate:
            with self.assertRaises(RuntimeError): q.ResizableConsole(server.port, canvas=(8, 6))
            allocate.assert_not_called()

    def test_invalid_canvas_is_rejected_before_connection(self):
        for canvas in ((True, 2), (0, 2), (2, 2161), (3841, 2)):
            with self.subTest(canvas=canvas), self.assertRaises(ValueError): q.ResizableConsole(1, canvas=canvas)

    def test_geometry_messages_discard_clipboard_and_bell(self):
        reply=b"\x02\x03\0\0\0"+struct.pack(">I", 3)+b"cpu"+raw(2, 2, (255, 0, 0))
        with RFB((2, 2), [reply]) as server:
            client=q.ResizableConsole(server.port, canvas=(8, 6))
            try: self.assertEqual(client.screenshot().getpixel((1, 1)), (255, 0, 0))
            finally: client.close()

    def test_real_codec_keeps_expanded_framebuffer_and_geometry_events(self):
        import av
        import imageio_ffmpeg
        with tempfile.TemporaryDirectory() as folder, RFB((2, 2), [raw(2, 2, (255, 0, 0)), resize(6, 4), raw(6, 4, (0, 255, 0))]) as server:
            output=Path(folder)/"recording"
            result=q.record(output, imageio_ffmpeg.get_ffmpeg_exe(), canvas=(8, 6), port=server.port, fps=10, duration=.65)
            self.assertTrue(result["complete"]); self.assertTrue(result["whole_framebuffer_validated"])
            events=[json.loads(line) for line in (output/"framebuffer-geometry.jsonl").read_text().splitlines()]
            self.assertEqual(events[0]["framebuffer"], [2, 2]); self.assertTrue(any(e["framebuffer"]==[6, 4] for e in events))
            self.assertTrue(all(e["input_events"]==0 for e in events))
            with av.open(str(output/"screen.mp4")) as video:
                frames=list(video.decode(video=0)); self.assertGreaterEqual(len(frames), 4)
                self.assertTrue(all((f.width, f.height)==(8, 6) for f in frames))
                self.assertTrue(any(f.to_ndarray(format="rgb24")[3, 5, 1]>180 for f in frames))

    def test_owned_cli_sigterm_finishes_video_and_closes_console(self):
        import imageio_ffmpeg
        with tempfile.TemporaryDirectory() as folder, RFB((4, 4), [raw(4, 4, (0, 255, 0))]) as server:
            output=Path(folder)/"recording"
            command=[sys.executable, "-m", "resizable_vnc_recording", "--output", str(output),
                     "--ffmpeg", imageio_ffmpeg.get_ffmpeg_exe(), "--port", str(server.port),
                     "--canvas-width", "8", "--canvas-height", "6", "--fps", "10", "--duration", "20"]
            child=subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline=time.monotonic()+8
                while not (output/"framebuffer-geometry.jsonl").exists():
                    if child.poll() is not None: self.fail(child.communicate()[1])
                    if time.monotonic()>=deadline: self.fail("Owned recorder did not reach first complete capture")
                    time.sleep(.02)
                child.terminate(); stdout, stderr=child.communicate(timeout=10)
                self.assertEqual(child.returncode, 0, stderr)
                result=json.loads((output/"resizable-recording.json").read_text())
                self.assertTrue(result["complete"]); self.assertEqual(json.loads(stdout)["input_events"], 0)
                self.assertTrue(all(kind in (0, 2, 3) for kind, _ in server.requests))
            finally:
                if child.poll() is None: child.terminate(); child.wait(timeout=10)


if __name__ == "__main__": unittest.main()
