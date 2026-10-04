"""Isolated read-only recording candidate with explicit desktop-size handling.

The frozen recorder is reused without editing it. Captures are placed unscaled
at the top left of a declared fixed canvas. Oversize or incomplete screens fail
instead of silently clipping. RFC 6143 section 7.8.2 defines DesktopSize.
"""
import argparse
import ipaddress
import json
from pathlib import Path
import socket
import signal
import struct
import threading
import time
from unittest.mock import patch

from PIL import Image
from vm.vnc_admin import Console
from vm import record_console


def full_coverage(rectangles, width, height):
    """Check the union of delivered raw rectangles, without a pixel-sized map."""
    edges = sorted({0, height, *(y for _, y, _, _ in rectangles), *(y+h for _, y, _, h in rectangles)})
    for top, bottom in zip(edges, edges[1:]):
        if bottom == top: continue
        spans = sorted((x, x+w) for x, y, w, h in rectangles if y <= top and y+h >= bottom)
        end = 0
        for left, right in spans:
            if left > end: return False
            end = max(end, right)
        if end < width: return False
    return True


class ResizableConsole(Console):
    def __init__(self, port=15900, host="127.0.0.1", *, canvas=(1920, 1080), on_capture=None):
        if (len(canvas) != 2 or any(type(v) is not int or v <= 0 for v in canvas)
                or canvas[0] > 3840 or canvas[1] > 2160):
            raise ValueError("Declare a bounded fixed canvas before recording")
        self.canvas, self.on_capture, self.capture_index = canvas, on_capture, 0
        self.desktop_size_confirmed = False
        if not ipaddress.ip_address(host).is_private: raise ValueError("Use an owned private console")
        self.socket = socket.create_connection((host, port), 10)
        self.socket.settimeout(3)
        try:
            if self.read(12) != b"RFB 003.008\n": raise RuntimeError("Require qualified RFB 3.8 console")
            self.socket.sendall(b"RFB 003.008\n")
            modes = self.read(self.read(1)[0])
            if 1 not in modes: raise RuntimeError("Expected owned no-auth console")
            self.socket.sendall(b"\x01")
            if self.read(4) != b"\0"*4: raise RuntimeError("Console handshake failed")
            self.socket.sendall(b"\x01")
            self.actual_width, self.actual_height = struct.unpack(">HH", self.read(4))
            # Bound dimensions before allocating any initial framebuffer.
            self._size(self.actual_width, self.actual_height)
            self.read(16); name_length = struct.unpack(">I", self.read(4))[0]
            if name_length > 1048576: raise RuntimeError("Oversized console name")
            self.read(name_length)
            self.framebuffer = Image.new("RGB", (self.actual_width, self.actual_height))
            self.width, self.height = canvas
            self.socket.sendall(struct.pack(">BBBBBBBBHHHBBBxxx", 0,0,0,0,32,24,0,1,255,255,255,16,8,0))
            self.socket.sendall(struct.pack(">BBHii", 2, 0, 2, 0, -223))
        except BaseException:
            self.close(); raise

    def _size(self, width, height):
        if not (0 < width <= self.canvas[0] and 0 < height <= self.canvas[1]):
            raise RuntimeError("Framebuffer exceeds declared canvas; refuse cropped recording")

    def _update(self):
        for _ in range(256):
            kind = self.read(1)[0]
            if kind == 2: continue
            if kind == 3:
                self.read(3); length = struct.unpack(">I", self.read(4))[0]
                if length > 1048576: raise RuntimeError("Oversized clipboard message")
                self.read(length); continue
            if kind != 0: raise RuntimeError(f"Unsupported read-only console message {kind}")
            self.read(1); count = struct.unpack(">H", self.read(2))[0]
            if count > 4096: raise RuntimeError("Too many framebuffer rectangles")
            rectangles, resized, pixels = [], False, 0
            for index in range(count):
                x, y, width, height, encoding = struct.unpack(">HHHHi", self.read(12))
                if encoding == -223:
                    if index != count-1: raise RuntimeError("DesktopSize must be the last rectangle")
                    self._size(width, height)
                    new = Image.new("RGB", (width, height)); new.paste(self.framebuffer, (0, 0))
                    self.framebuffer = new; self.actual_width, self.actual_height = width, height
                    self.desktop_size_confirmed = True
                    rectangles, resized = [], True
                    continue
                if encoding != 0: raise RuntimeError(f"Unsupported console encoding {encoding}")
                if (width <= 0 or height <= 0 or x+width > self.actual_width or y+height > self.actual_height):
                    raise RuntimeError("Raw rectangle outside announced framebuffer")
                pixels += width*height
                if pixels > 2*self.canvas[0]*self.canvas[1]: raise RuntimeError("Excessive raw update data")
                patch_image = Image.frombytes("RGB", (width, height), self.read(width*height*4), "raw", "BGRX")
                self.framebuffer.paste(patch_image, (x, y)); rectangles.append((x, y, width, height))
            return rectangles, resized
        raise RuntimeError("Console message deadline exceeded")

    def screenshot(self):
        rectangles, transitions = [], []
        for _ in range(16):
            self.socket.sendall(struct.pack(">BBHHHH", 3, 0, 0, 0, self.actual_width, self.actual_height))
            update, resized = self._update()
            if resized:
                rectangles = []
                transitions.append([self.actual_width, self.actual_height])
                continue
            rectangles.extend(update)
            if not full_coverage(rectangles, self.actual_width, self.actual_height): continue
            result = Image.new("RGB", self.canvas); result.paste(self.framebuffer, (0, 0))
            event = dict(capture=self.capture_index, captured_monotonic_ns=time.monotonic_ns(),
                         canvas=list(self.canvas), framebuffer=[self.actual_width, self.actual_height],
                         resize_notifications=transitions, full_framebuffer_received=True,
                         desktop_size_extension_confirmed=self.desktop_size_confirmed,
                         evidence_scope="announced RFB framebuffer; guest display agreement qualified separately",
                         pixel_scaling=False, top_left_offset=[0, 0], input_events=0)
            if self.on_capture is not None: self.on_capture(event)
            self.capture_index += 1
            return result
        raise RuntimeError("Complete framebuffer was not received; no successful capture")


def record(output, ffmpeg, *, canvas=(1920, 1080), host="127.0.0.1", port=15900, fps=2, duration=None, stop=None):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError("Recording output already exists")
    events = []
    def capture(event):
        events.append(event)
        with (output / "framebuffer-geometry.jsonl").open("a") as log: log.write(json.dumps(event)+"\n")
    factory = lambda port, host: ResizableConsole(port, host, canvas=canvas, on_capture=capture)
    with patch.object(record_console, "Console", factory):
        result = record_console.record(output, ffmpeg, duration=duration, fps=fps, port=port, host=host, stop=stop)
    if not events or result.get("complete") is not True:
        raise RuntimeError("Recording lacks validated framebuffer captures")
    result.update(resizable_capture=True, declared_canvas=list(canvas), pixel_scaling=False,
                  framebuffer_geometry_events=len(events), whole_framebuffer_validated=True)
    (output / "resizable-recording.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--ffmpeg", required=True)
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=15900)
    p.add_argument("--canvas-width", type=int, default=1920); p.add_argument("--canvas-height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=2); p.add_argument("--duration", type=float, required=True)
    args = p.parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    result = record(args.output, args.ffmpeg, canvas=(args.canvas_width, args.canvas_height), host=args.host,
                    port=args.port, fps=args.fps, duration=args.duration, stop=stop)
    print(json.dumps(dict(complete=result["complete"], captures=result["captures"], input_events=0)))


if __name__ == "__main__": main()
