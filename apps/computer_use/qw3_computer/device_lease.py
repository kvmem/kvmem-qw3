"""One input owner per concrete endpoint/device, across application runtimes."""
import hashlib
import os
from pathlib import Path
import tempfile


class DeviceLease:
    def __init__(self, identity):
        digest = hashlib.sha256(identity.encode()).hexdigest()
        self.path = Path(tempfile.gettempdir()) / ("qw3-device-" + digest + ".lock")
        self.handle = None

    def acquire(self):
        handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.path.stat().st_size == 0:
                    handle.write(b"0");handle.flush()
                handle.seek(0);msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError("This device already has an input owner; stop that task before attaching another runtime") from exc
        self.handle = handle
        return self

    def close(self):
        if self.handle:
            self.handle.close();self.handle=None
