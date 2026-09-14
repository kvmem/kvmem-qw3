"""Serialized RPC to the frozen native text frontend; no model work here."""
import json
from pathlib import Path
import subprocess
import threading


class NativeBridge:
    def __init__(self, model, binary=None):
        root = Path(__file__).resolve().parents[1]
        self.process = subprocess.Popen([str(binary or root / "native_build/baseline_native_bridge"),
            str(model)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lock = threading.Lock()

    def call(self, op, **kwargs):
        with self.lock:
            self.process.stdin.write(json.dumps(dict(op=op, **kwargs), ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"native bridge exited: {self.process.poll()}")
            reply = json.loads(line)
            if not reply["ok"]:
                raise ValueError(reply["error"])
            return reply["result"]

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.process.stdout.close()


def validate_text_request(request):
    """Fail explicitly on unsupported modalities/features instead of dropping them."""
    if request.get("stop_sequences"):
        raise ValueError("custom stop_sequences are not yet validated by this baseline")
    if request.get("n", 1) != 1:
        raise ValueError("only one completion is supported")
    def content(value):
        if isinstance(value, str):
            return
        if not isinstance(value, list):
            raise ValueError("content must be text or a block list")
        for block in value:
            kind = block.get("type")
            if kind not in ("text", "thinking", "redacted_thinking", "tool_use", "tool_result"):
                raise ValueError(f"unvalidated content block type: {kind}")
            if kind == "tool_result":
                content(block.get("content", ""))
    if "system" in request:
        content(request["system"])
    for message in request.get("messages", []):
        content(message.get("content", ""))


def stream_events(wire):
    for line in wire.splitlines():
        if line.startswith("data: "):
            yield json.loads(line[6:])


def meaningful_content(event):
    if event.get("type") == "content_block_start":
        return event.get("content_block", {}).get("type") == "tool_use"
    delta = event.get("delta", {})
    return bool(delta.get("text") or delta.get("thinking") or delta.get("partial_json"))


def action_content(event):
    if event.get("type") == "content_block_start":
        return event.get("content_block", {}).get("type") == "tool_use"
    return bool(event.get("delta", {}).get("text"))
