#!/usr/bin/env python3
"""Tiny TCP relay used to expose a local QW3 port through Pier's safe port 80."""

from __future__ import annotations

import argparse
import socket
import socketserver
import threading


def copy_stream(source: socket.socket, destination: socket.socket) -> None:
    try:
        while data := source.recv(1024 * 1024):
            destination.sendall(data)
    except (ConnectionError, OSError):
        pass
    finally:
        try:
            destination.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class RelayHandler(socketserver.BaseRequestHandler):
    target: tuple[str, int]

    def handle(self) -> None:
        with socket.create_connection(self.target) as upstream:
            upload = threading.Thread(
                target=copy_stream, args=(self.request, upstream), daemon=True
            )
            download = threading.Thread(
                target=copy_stream, args=(upstream, self.request), daemon=True
            )
            upload.start()
            download.start()
            upload.join()
            download.join()


class ThreadedRelay(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", required=True)
    parser.add_argument("--listen-port", required=True, type=int)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", required=True, type=int)
    args = parser.parse_args()

    RelayHandler.target = (args.target_host, args.target_port)
    with ThreadedRelay((args.listen_host, args.listen_port), RelayHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
