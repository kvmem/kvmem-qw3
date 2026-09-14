"""Real pinned Claude Code read-tool round trip, separate from task outcomes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, default=18227)
    args = p.parse_args()
    args.out = args.out.resolve()
    workspace = args.out / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "fixture.txt").write_text("BASELINE_PROTOCOL_OK_73\n")
    environment = dict(os.environ)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environment.pop(key, None)
    command = [str(ROOT / ".venv/bin/python"), str(ROOT / "serve_infllm.py"),
        "--model", str(REPO / "models/Qwen3.8-27B-BF16"), "--out", str(args.out / "server"),
        "--port", str(args.port), "--shared-gpu", "--temperature", "0", "--max-context", "65536",
        "--max-tokens", "1024", "--thinking-budget", "256"]
    report = {"status": "starting", "correctness_only": True, "performance_claim": False,
        "task_success_claim": False, "server_command": command}
    def save():
        (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    log = (args.out / "server.log").open("w")
    server = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=log)
    base = f"http://127.0.0.1:{args.port}"
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode}")
            try:
                with http.open(base + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(.5)
        else:
            raise TimeoutError("server readiness")
        client_environment = dict(environment, ANTHROPIC_BASE_URL=base,
            ANTHROPIC_API_KEY="local-baseline", ANTHROPIC_AUTH_TOKEN="local-baseline",
            CLAUDE_CONFIG_DIR=str(args.out / "claude-config"),
            CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_COMPACT="1",
            CLAUDE_CODE_MAX_CONTEXT_TOKENS="65536")
        model_id = "Qwen3.8-27B-BF16[1m]"
        for name in ("ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
            client_environment[name] = model_id
        client = ["/home/chaidi/.local/share/claude/versions/2.1.148", "-p",
            "--output-format", "stream-json", "--verbose", "--include-partial-messages",
            "--tools", "Read", "--allowedTools", "Read", "--disallowedTools", "Agent,EnterPlanMode,AskUserQuestion",
            "--setting-sources", "project,local", "--settings", "{}", "--strict-mcp-config",
            "--model", model_id, "--effort", "xhigh", "--debug-file", str(args.out / "claude.debug.log"),
            "--", "Use the Read tool to read fixture.txt in the current directory. Reply with its first line verbatim. Do not guess the content."]
        report.update(status="running_claude", client_command=client)
        save()
        with (args.out / "claude.stream.jsonl").open("w") as output, (args.out / "claude.stderr.log").open("w") as errors:
            result = subprocess.run(client, cwd=workspace, env=client_environment,
                stdout=output, stderr=errors, timeout=180)
        report["client_exit_code"] = result.returncode
        events = [json.loads(line) for line in (args.out / "claude.stream.jsonl").read_text().splitlines()]
        uses = [block for event in events if event.get("type") == "assistant"
            for block in event.get("message", {}).get("content", []) if block.get("type") == "tool_use"]
        outcomes = [event for event in events if event.get("type") == "result"]
        assert result.returncode == 0 and outcomes and not outcomes[-1].get("is_error"), outcomes
        assert any(u.get("name") == "Read" for u in uses), uses
        assert "BASELINE_PROTOCOL_OK_73" in outcomes[-1].get("result", ""), outcomes[-1]
        rows = [json.loads(line) for line in (args.out / "server/requests.jsonl").read_text().splitlines()]
        assert len(rows) >= 2 and all(r["status"] == "completed" for r in rows), [(r["status"], r.get("error")) for r in rows]
        assert rows[-1]["resume_origin"] > 0, rows[-1]["reuse_reason"]
        report.update(status="passed_real_claude_read_tool_roundtrip", client_result=outcomes[-1],
            model_requests=len(rows), read_tool_calls=len(uses), reuse_origins=[r["resume_origin"] for r in rows])
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        log.close()
        report["finished"] = time.time()
        save()


if __name__ == "__main__":
    main()
