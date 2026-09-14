"""Reconstruct a held-out text trajectory; explicitly not original wire bodies.

Use exact recorded assistant/tool content and the recorded task prompt, rendered
with one newly captured, frozen Claude tool/system template for all methods.
No tools execute during replay. This artifact cannot establish task success.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

from kv_baselines.frontend import NativeBridge, validate_text_request

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def prepare(arm, template_path, out, max_tokens):
    task_id = arm.parent.name
    pilot = json.loads((ROOT / "tasks_pilot4.json").read_text())
    if task_id in {t["task_id"] for t in pilot["tasks"]}:
        raise ValueError("calibration must be held out from the pilot task outcomes")
    if out.exists():
        raise FileExistsError("preserve the existing frozen trace")
    out.mkdir(parents=True)
    stream_path, prompt_path = arm / "claude.stream.jsonl", arm / "prompt.md"
    template = json.loads(template_path.read_text())
    history = [{"role": "user", "content": prompt_path.read_text()}]
    rows = []
    last_id = None
    bridge = NativeBridge(REPO / "models/Qwen3.8-27B-BF16")
    try:
        for line_number, line in enumerate(stream_path.open(), 1):
            event = json.loads(line)
            if event.get("parent_tool_use_id") is not None:
                continue
            if event.get("type") == "system" and event.get("subtype") == "compact_boundary":
                break
            if event.get("type") == "user":
                message = event["message"]
                history.append({"role": "user", "content": copy.deepcopy(message["content"])})
            elif event.get("type") == "assistant":
                message = event["message"]
                message_id = message["id"]
                if message_id != last_id:
                    body = copy.deepcopy(template)
                    body["messages"] = copy.deepcopy(history)
                    body["max_tokens"] = 64
                    body["stream"] = True
                    validate_text_request(body)
                    rendered = bridge.call("render", request=body)
                    length = len(rendered["input_ids"])
                    if length > max_tokens:
                        break
                    raw = (json.dumps(body, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                    filename = f"request_{len(rows):04d}.json.gz"
                    (out / filename).write_bytes(gzip.compress(raw, mtime=0))
                    rows.append({"index": len(rows), "path": filename,
                        "prompt_tokens": length, "body_sha256": hashlib.sha256(raw).hexdigest(),
                        "source_line": line_number, "source_message_id": message_id,
                        "recorded_input_tokens": message.get("usage", {}).get("input_tokens")})
                    history.append({"role": "assistant", "content": []})
                    last_id = message_id
                for block in message.get("content", []):
                    if block not in history[-1]["content"]:
                        history[-1]["content"].append(copy.deepcopy(block))
        if len(rows) < 4:
            raise ValueError("too few held-out requests")
        manifest = {"status": "prepared_not_measured", "task_id": task_id,
            "construction": "reconstructed from recorded block-wise transcript, with current frozen Claude Code template",
            "original_wire_request": False, "claude_template_version": "2.1.148",
            "task_success_claim": False, "executes_tools": False, "synthetic_repeated_padding": False,
            "max_prompt_tokens": max_tokens, "generation_cap": 64,
            "output_warning": "Fixed short output cap is for controlled request timing, not task evaluation. Truncated tool streams must be reported separately.",
            "source_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (stream_path, prompt_path, template_path)}, "requests": rows}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"requests": len(rows), "first_prompt_tokens": rows[0]["prompt_tokens"],
            "last_prompt_tokens": rows[-1]["prompt_tokens"], "task_id": task_id}))
    finally:
        bridge.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-prompt-tokens", type=int, default=131072)
    a = p.parse_args()
    prepare(a.arm.resolve(), a.template.resolve(), a.out.resolve(), a.max_prompt_tokens)


if __name__ == "__main__":
    main()
