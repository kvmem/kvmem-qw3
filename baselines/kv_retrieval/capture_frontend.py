"""Capture current Claude Code's default text tool schema without model/GPU work.

The existing capture fixture responds with Bash(true), then DONE. This produces
a template for a reconstructed fixed trace, never a task-success measurement.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("native_capture_fixture", REPO / "benchmark/claude_deepswe_ab/capture_claude_requests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    server = module.CaptureServer(("127.0.0.1", 0), args.out / "requests.jsonl")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = dict(os.environ)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    env.update(ANTHROPIC_BASE_URL=f"http://127.0.0.1:{server.server_port}",
        ANTHROPIC_API_KEY="local-fixture", ANTHROPIC_AUTH_TOKEN="local-fixture",
        CLAUDE_CONFIG_DIR=str(args.out / "claude-config"), CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
        DISABLE_COMPACT="1", CLAUDE_CODE_MAX_CONTEXT_TOKENS="1048576")
    model = "Qwen3.8-27B-BF16[1m]"
    for key in ("ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
        env[key] = model
    cmd = ["/home/chaidi/.local/share/claude/versions/2.1.148", "-p", "--output-format", "stream-json",
        "--verbose", "--include-partial-messages", "--allowedTools", "Bash",
        "--disallowedTools", "Agent,EnterPlanMode,AskUserQuestion", "--setting-sources", "project,local",
        "--settings", "{}", "--strict-mcp-config", "--model", model, "--effort", "xhigh",
        "--", "Run true through Bash, then reply DONE. This is an interface validation fixture."]
    try:
        with (args.out / "claude.stream.jsonl").open("w") as output, (args.out / "stderr.log").open("w") as errors:
            subprocess.run(cmd, env=env, cwd=args.out, stdout=output, stderr=errors, timeout=45, check=True)
        rows = [json.loads(s) for s in (args.out / "requests.jsonl").read_text().splitlines()]
        bodies = [r["body"] for r in rows if r["path"] == "/v1/messages"]
        if len(bodies) < 2:
            raise RuntimeError("capture did not exercise a tool round trip")
        names = [t["name"] for t in bodies[0]["tools"]]
        if not {"Bash", "Read", "Edit", "Write"}.issubset(names) or "Agent" in names:
            raise RuntimeError(f"unexpected default tool schema: {names}")
        (args.out / "template.json").write_text(json.dumps(bodies[0], ensure_ascii=False, indent=2) + "\n")
        (args.out / "result.json").write_text(json.dumps({"status": "passed", "synthetic_response_fixture": True,
            "claude_version": "2.1.148", "tools": names, "model_requests": len(bodies)}, indent=2) + "\n")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
