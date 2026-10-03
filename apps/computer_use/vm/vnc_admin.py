"""Local QEMU console helper for VM provisioning, never exposed to the model."""
import argparse
from pathlib import Path
import socket
import struct

from PIL import Image


class Console:
    def __init__(self, port=15900, host="127.0.0.1"):
        # Administrative local VM consoles only; never exposed to Computer MCP.
        import ipaddress
        if not ipaddress.ip_address(host).is_private:
            raise ValueError("Console host must be a private or loopback IP")
        self.socket = socket.create_connection((host, port), 10)
        self.socket.settimeout(20)
        self.read(12)
        self.socket.sendall(b"RFB 003.008\n")
        modes = self.read(self.read(1)[0])
        if 1 not in modes:
            raise RuntimeError("Expected the local QEMU console's no-auth mode")
        self.socket.sendall(b"\x01")
        if self.read(4) != b"\0" * 4:
            raise RuntimeError("Console handshake failed")
        self.socket.sendall(b"\x01")
        self.width, self.height = struct.unpack(">HH", self.read(4))
        self.read(16)
        self.read(struct.unpack(">I", self.read(4))[0])
        self.socket.sendall(struct.pack(">BBBBBBBBHHHBBBxxx", 0,0,0,0,32,24,0,1,255,255,255,16,8,0))
        self.socket.sendall(struct.pack(">BBHi", 2,0,1,0))
        self.frame = Image.new("RGB", (self.width, self.height))

    def read(self, count):
        data = b""
        while len(data) < count:
            chunk = self.socket.recv(count - len(data))
            if not chunk:
                raise EOFError("VNC connection closed")
            data += chunk
        return data

    def screenshot(self):
        self.socket.sendall(struct.pack(">BBHHHH", 3,0,0,0,self.width,self.height))
        while True:
            kind = self.read(1)[0]
            if kind == 2:
                continue
            if kind == 3:
                self.read(3)
                self.read(struct.unpack(">I", self.read(4))[0])
                continue
            if kind != 0:
                raise RuntimeError(f"Unsupported console message {kind}")
            self.read(1)
            count = struct.unpack(">H", self.read(2))[0]
            for _ in range(count):
                x,y,width,height,encoding = struct.unpack(">HHHHi", self.read(12))
                if encoding != 0:
                    raise RuntimeError(f"Unsupported console encoding {encoding}")
                self.frame.paste(Image.frombytes("RGB", (width,height), self.read(width*height*4), "raw", "BGRX"), (x,y))
            if count:
                return self.frame.copy()

    def close(self):
        self.socket.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=15900)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    console = Console(args.port)
    try:
        if args.screenshot:
            console.screenshot().save(args.screenshot)
    finally:
        console.close()
