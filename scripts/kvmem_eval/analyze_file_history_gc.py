#!/usr/bin/env python3
"""Offline evidence for file-history GC; never executes recorded tool commands.

Reads canonical Claude streams from one cohort/seed. Counts explicit completed
Read/Edit/Write events, exact visible Read repetitions, edit/reread patterns,
and literal inverse edits. Bash operations are deliberately not interpreted as
file-version facts. The output does not estimate GC regret or causal necessity.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import median


def content_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(content_text(x) for x in value)
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return ""


def read_view(text):
    # Compare displayed source without line-number formatting or reminders.
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S)
    numbered = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+)(?:\t|→)(.*)$", line)
        if match:
            numbered.append((int(match[1]), match[2]))
    body = "\n".join(line for _, line in numbered) if numbered else text.strip()
    return {
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "chars": len(body),
        "line_begin": numbered[0][0] if numbered else None,
        "line_end": numbered[-1][0] if numbered else None,
    }


def load_events(path):
    calls = {}
    completed = set()
    events = []
    malformed = 0
    result = None
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, raw in enumerate(stream, 1):
            # Stream fragments duplicate the later aggregate messages.
            if re.search(r'"type"\s*:\s*"stream_event"', raw[:90]):
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                malformed += 1
                continue
            if row.get("type") == "result":
                result = {k: row.get(k) for k in ("subtype", "is_error")}
            message = row.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    call_id = block.get("id")
                    if call_id and call_id not in calls:
                        calls[call_id] = {
                            "id": call_id,
                            "name": block.get("name"),
                            "input": block.get("input", {}),
                            "call_line": line_number,
                        }
                elif block.get("type") == "tool_result":
                    call_id = block.get("tool_use_id")
                    if call_id not in calls or call_id in completed:
                        continue
                    completed.add(call_id)
                    output = content_text(block.get("content"))
                    event = {
                        **calls[call_id],
                        "event": len(events) + 1,
                        "result_line": line_number,
                        "error": bool(block.get("is_error", False)),
                        "output_excerpt": output[:1200],
                    }
                    if event["name"] == "Read":
                        event["view"] = read_view(output)
                    events.append(event)
    return events, malformed, result


def analyze(path, root):
    events, malformed, result = load_events(path)
    counts = Counter()
    last_read = {}
    previous_edits = defaultdict(list)
    file_edits = defaultdict(list)
    duplicate = []
    reread = []
    reverse = []
    returns = []
    read_history = defaultdict(list)
    bash_history = []
    for event in events:
        name = event["name"]
        counts[f"completed_{name}"] += 1
        if event["error"]:
            counts[f"error_{name}"] += 1
            continue
        args = event["input"]
        file_path = args.get("file_path") if isinstance(args, dict) else None
        if name in ("Edit", "Write") and file_path:
            file_edits[file_path].append(event)
            if name == "Edit":
                old = args.get("old_string", "")
                new = args.get("new_string", "")
                if old != new and old and new:
                    for prior in reversed(previous_edits[file_path]):
                        p = prior["input"]
                        if p.get("old_string") == new and p.get("new_string") == old:
                            reverse.append({
                                "path": file_path, "first": prior["event"],
                                "reverse": event["event"],
                                "first_line": prior["call_line"],
                                "reverse_line": event["call_line"],
                                "old_excerpt": old[:700], "new_excerpt": new[:700],
                                "same_replace_all": p.get("replace_all", False) == args.get("replace_all", False),
                            })
                            break
                previous_edits[file_path].append(event)
        elif name == "Read" and file_path:
            key = (file_path, args.get("offset", 1), args.get("limit"))
            prior = last_read.get(key)
            if prior and event["view"]["sha256"] == prior["view"]["sha256"] and event["view"]["chars"]:
                duplicate.append({"path": file_path, "first": prior["event"], "again": event["event"],
                                  "first_line": prior["call_line"], "again_line": event["call_line"],
                                  "visible_chars": event["view"]["chars"]})
            any_prior = read_history[file_path][-1] if read_history[file_path] else None
            if any_prior:
                between = [e for e in file_edits[file_path] if any_prior["event"] < e["event"] < event["event"]]
                if between:
                    reread.append({"path": file_path, "first": any_prior["event"], "again": event["event"],
                                   "first_line": any_prior["call_line"], "again_line": event["call_line"],
                                   "edits": [e["event"] for e in between],
                                   "same_requested_range": key == (file_path, any_prior["input"].get("offset", 1), any_prior["input"].get("limit")),
                                   "first_view": any_prior["view"], "again_view": event["view"]})
            history = [r for r in read_history[file_path] if (r["input"].get("offset", 1), r["input"].get("limit")) == key[1:]]
            if len(history) >= 2 and history[-1]["view"]["sha256"] != event["view"]["sha256"]:
                matching = next((r for r in reversed(history[:-1]) if r["view"]["sha256"] == event["view"]["sha256"]), None)
                if matching and event["view"]["chars"]:
                    returns.append({"path": file_path, "first": matching["event"], "changed": history[-1]["event"],
                                    "restored": event["event"], "first_line": matching["call_line"],
                                    "changed_line": history[-1]["call_line"], "restored_line": event["call_line"]})
            last_read[key] = event
            read_history[file_path].append(event)
        elif name == "Bash":
            command = str(args.get("command", ""))
            if re.search(r"\bgit\s+(?:show|diff|checkout|restore|reset|revert)\b", command):
                bash_history.append({"event": event["event"], "call_line": event["call_line"], "command": command[:1800]})
    reverse_after_reread = [x for x in reverse if any(r["path"] == x["path"] and x["first"] < r["again"] < x["reverse"] for r in reread)]
    metrics_path = path.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    counts.update({"read_edit_reread": len(reread), "identical_visible_read": len(duplicate),
                   "literal_inverse_edit": len(reverse), "inverse_after_reread": len(reverse_after_reread),
                   "read_view_returns_to_previous": len(returns), "bash_history_commands": len(bash_history)})
    relative = path.relative_to(root)
    task = relative.parts[1]
    return {"task": task, "cohort_stream": str(path.absolute()),
            "stream": str(path.resolve()), "source_bytes": path.stat().st_size,
            "malformed_lines": malformed, "terminal_result": result,
            "agent_exit": metrics.get("agent_exit"), "reward": metrics.get("reward"),
            "server_selection_metrics": {key: value for key, value in metrics.get("server", {}).items()
                                         if "reselection" in key or "selection_pass" in key},
            "counts": dict(counts), "read_edit_reread": reread,
            "identical_visible_reads": duplicate, "inverse_edits": reverse,
            "inverse_after_reread": reverse_after_reread, "read_view_returns": returns,
            "bash_history_commands": bash_history, "events": events}


def selection_summary(rows):
    values = [row.get("server_selection_metrics", {}).get("kvmem_semantic_reselections") for row in rows]
    values = [value for value in values if isinstance(value, int)]
    return {
        "tasks_with_semantic_reselection_metric": len(values),
        "semantic_reselections_histogram": dict(sorted(Counter(values).items())),
        "semantic_reselections_median": median(values) if values else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", default="1000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--include-events", action="store_true", help="Include detailed tool inputs/output excerpts")
    args = parser.parse_args()
    files = sorted(args.root.glob(f"trials/*/seed-{args.seed}/runs/*/claude_kvmem_128k_gen64k/claude.stream.jsonl"))
    all_rows = [analyze(path, args.root) for path in files]
    rows = [row for row in all_rows if args.include_incomplete or row["terminal_result"] is not None]
    excluded = [row["task"] for row in all_rows if not args.include_incomplete and row["terminal_result"] is None]
    counts = Counter()
    tasks_with = Counter()
    for row in rows:
        counts.update(row["counts"])
        tasks_with.update(key for key, value in row["counts"].items() if value)
    rereads = [event for row in rows for event in row["read_edit_reread"]]
    summary = {"cohort": str(args.root.resolve()), "seed": args.seed,
               "discovered_streams": len(files), "streams": len(rows), "excluded_incomplete": excluded,
               "source_aliases": sum(row["stream"] != row["cohort_stream"] for row in rows),
               "source_bytes": sum(r["source_bytes"] for r in rows), "counts": dict(counts),
               "tasks_with": dict(tasks_with),
               "reread_same_requested_range": sum(e["same_requested_range"] for e in rereads),
               "reread_different_requested_range": sum(not e["same_requested_range"] for e in rereads),
               "terminal_results": dict(Counter(str(r["terminal_result"]) for r in rows)),
               "malformed_lines": sum(r["malformed_lines"] for r in rows),
               **selection_summary(rows)}
    if not args.include_events:
        for row in rows:
            row.pop("events", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "tasks": rows}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
