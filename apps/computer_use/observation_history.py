"""Task-owned archived observations, without capture, input or latest promotion."""
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3

from PIL import Image


IDENTITY_KEYS = ("desktop_id", "device_id", "session_id", "display_id", "observation_id", "sha256")


def observation_reference(metadata):
    if not isinstance(metadata, dict):
        return None
    result = {key: metadata.get(key) for key in IDENTITY_KEYS}
    if (any(not isinstance(v, str) or not 1 <= len(v) <= 512 for v in result.values())
            or not re.fullmatch(r"[a-f0-9]{64}", result["sha256"])):
        return None
    return result


class ObservationHistory:
    def __init__(self, desktop, runtime):
        self.desktop = desktop
        self.db = sqlite3.connect(Path(runtime) / "observation_history.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS pins(task TEXT,owner TEXT,id TEXT,data TEXT,PRIMARY KEY(task,owner,id))")
        self.db.commit()

    def owner(self):
        # RegisteredDesktop delegates db/runtime to its selected device. Never
        # accept a model-supplied path, task identity or database connection.
        return str(Path(self.desktop.runtime).resolve())

    def metadata(self, identity):
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity):
            raise ValueError("Use an observed observation_id, never a path")
        row = self.desktop.db.execute("SELECT metadata FROM observations WHERE task=? AND id=?",
                                      (self.desktop.task_id, identity)).fetchone()
        if not row:
            raise ValueError("Observation is not in this task's selected-device history")
        result = json.loads(row[0])
        if not isinstance(result, dict) or result.get("observation_id") != identity:
            raise ValueError("Archived observation identity does not match")
        return result

    def image(self, metadata):
        digest = metadata.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Archived screenshot digest is invalid")
        path = Path(self.desktop.runtime) / "screenshots" / (digest + ".png")
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Archived screenshot is missing or invalid")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Archived screenshot bytes changed")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.size != (metadata.get("width"), metadata.get("height")):
                raise ValueError("Archived screenshot dimensions changed")
            image.verify()
        return base64.b64encode(data).decode()

    def reference_pins(self):
        refs = []
        for identity, saved in self.db.execute("SELECT id,data FROM pins WHERE task=? AND owner=? ORDER BY rowid",
                                             (self.desktop.task_id, self.owner())):
            current = observation_reference(self.metadata(identity))
            if current is None or current != json.loads(saved):
                raise ValueError("Pinned observation metadata changed")
            refs.append(current)
        return refs

    def invoke(self, arguments):
        if not isinstance(arguments, dict):
            raise ValueError("history arguments must be an object")
        operation = arguments.get("operation")
        if operation == "list":
            if set(arguments) - {"operation", "before_rowid"}:
                raise ValueError("History list accepts only before_rowid")
            cursor = arguments.get("before_rowid")
            if cursor is not None and (type(cursor) is not int or not 1 <= cursor <= 2**63-1):
                raise ValueError("History cursor must be a positive SQLite row identifier")
            rows = self.desktop.db.execute(
                "SELECT rowid,id,metadata FROM observations WHERE task=? AND (? IS NULL OR rowid<?) ORDER BY rowid DESC LIMIT 21",
                (self.desktop.task_id, cursor, cursor)).fetchall()
            items = []
            for _, identity, value in rows[:20]:
                metadata = json.loads(value)
                front = metadata.get("foreground", {})
                items.append({"observation_id": identity, "sha256": metadata.get("sha256"),
                    "captured_at": metadata.get("captured_at"),
                    "foreground": {k: v[:256] for k,v in front.items()
                                   if k in {"handle", "process", "title"} and isinstance(v, str)},
                    "reference": observation_reference(metadata)})
            return {"operation": "list", "observations": items,
                    "next_before_rowid": rows[19][0] if len(rows) > 20 else None,
                    "pins": self.reference_pins(), "input_dispatched": False,
                    "scope": "Previously observed task-owned selected-device history only; not current device state."}
        if operation not in {"read", "pin", "unpin"} or set(arguments) != {"operation", "observation_id"}:
            raise ValueError("Use history list/read/pin/unpin with the documented fields")
        identity = arguments["observation_id"]
        metadata = self.metadata(identity)
        if operation == "read":
            return {"operation": "read", "input_dispatched": False,
                "observation": metadata | {"image": self.image(metadata), "historical_read_only": True},
                "history": {"task_id": self.desktop.task_id, "current_state_proven": False,
                            "promoted_to_latest": False, "input_target_eligible": False}}
        if operation == "pin":
            reference = observation_reference(metadata)
            if reference is None:
                raise ValueError("This observation lacks the complete device/session/image identity required for pinning")
            self.image(metadata)  # Validate before changing reference memory.
            pins = self.reference_pins()
            if reference not in pins and len(pins) >= 4:
                raise ValueError("At most four reference observations may be pinned; unpin an obsolete one first")
            self.db.execute("INSERT OR REPLACE INTO pins VALUES(?,?,?,?)",
                (self.desktop.task_id, self.owner(), identity, json.dumps(reference)))
        else:
            self.db.execute("DELETE FROM pins WHERE task=? AND owner=? AND id=?",
                            (self.desktop.task_id, self.owner(), identity))
        self.db.commit()
        return {"operation": operation, "pins": self.reference_pins(), "input_dispatched": False,
                "scope": "Reference memory only; does not change device state, latest observation or task budget."}

    def close(self):
        self.db.close()
