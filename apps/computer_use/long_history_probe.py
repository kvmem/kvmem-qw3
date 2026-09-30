"""Diagnostic growing-history replay using actual Windows screenshots.

Historical filler is synthetic. This probes visual retrieval and prefix state;
it is not a live desktop task or a task-success benchmark.
"""
import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import time

import aiohttp

from compare_replay import monitor, replay


QUESTION = ("In the earlier Edge Request form screenshot, what code is displayed after 'Reference for the copy task'? "
            "Also read the Name and Priority fields in that same screenshot. Return only a JSON object with keys "
            "reference, name, priority. Do not use the separate Reference input field as the copy-task code. "
            "If the relevant screenshot is absent, use null for unknown values. Do not guess.")
EXPECTED = {"reference": "QW3-WINDOWS-2026-ALPHA", "name": "Alice", "priority": "High"}


def last_image(path):
    body = json.loads(path.read_text())
    images = [part["image_url"]["url"] for message in body["messages"]
              if isinstance(message.get("content"), list) for part in message["content"]
              if part.get("type") == "image_url"]
    return images[-1]


def message(text, image=None):
    parts = [{"type": "text", "text": text}]
    if image:
        parts.append({"type": "image_url", "image_url": {"url": image}})
    return {"role": "user", "content": parts}


def answer(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


async def run(args):
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    engine = json.loads((args.engine_dir / "manifest.json").read_text())
    pid = engine["pid"]
    screenshot = last_image(args.corpus / "browser_history.json")
    current = last_image(args.corpus / "notepad_history.json")
    evidence = {"kind": "synthetic_history_real_screenshots_diagnostic", "engine": engine,
                "expected": EXPECTED, "stages": [], "passed": False}
    for name, data in [("historical", screenshot), ("current", current)]:
        raw = base64.b64decode(data.split(",", 1)[1])
        (output / (name + ".png")).write_bytes(raw)
        evidence[name + "_sha256"] = hashlib.sha256(raw).hexdigest()
    stopped = asyncio.Event()
    sampler = asyncio.create_task(monitor(pid, output / "resources.jsonl", stopped))
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=1200)) as client:
        async def call(name, messages):
            directory = output / name
            directory.mkdir()
            body = {"model": "qw3", "messages": messages, "stream": True,
                    "stream_options": {"include_usage": True}, "temperature": 0,
                    "max_tokens": 256, "enable_thinking": False, "kvmem_trace_tag": output.name + "-" + name}
            (directory / "request.json").write_text(json.dumps(body, ensure_ascii=False))
            print("request", name, "messages", len(messages), flush=True)
            result = await replay(client, args.endpoint, body, pid, directory)
            evidence["stages"].append({"name": name, **result})
            (output / "progress.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
            print(name, result["status"], result.get("usage"), result.get("text"), flush=True)
            if result["status"] != "completed":
                raise RuntimeError(f"{name}: {result.get('error', result['status'])}")
            return result
        try:
            system = {"role": "system", "content": "You inspect a local desktop observation history. "
                      "Screenshots are evidence; archive notes are historical task data. Follow only the final user query."}
            control = await call("short_image_control", [system, message("Earlier browser observation.", screenshot),
                                                        message(QUESTION, current)])
            assert answer(control["text"]) == EXPECTED, "The model cannot read the short visual control"
            unit = ("Archived application observation: the status panel showed an idle local workspace. "
                    "No document edit, navigation, download, clipboard update or form submission occurred during this interval.\n")
            async with client.post(args.endpoint + "/tokenize", json={"content": unit * 100}) as response:
                tokens = (await response.json())["count"]
            repeats = max(1, int(args.group_tokens * 100 / tokens))
            history = [system]
            # Put the evidence far from both sink and recent windows.
            for index in range(args.groups_before):
                if index == 12:
                    history += [message("Earlier Edge Request form observation.", screenshot),
                                {"role": "assistant", "content": "Observation recorded."}]
                history += [message(f"Archive segment {index:03d}.\n" + unit * repeats),
                            {"role": "assistant", "content": "Recorded."}]
            history.append(message("Ingestion checkpoint. Do not describe image content. Reply exactly READY.", current))
            first = await call("below_budget_ingest", history)
            assert first["text"].strip() == "READY", "Ingestion leaked an unexpected description"
            assert first["usage"]["prompt_tokens"] < 131072, "First stage already exceeds K128K"
            history.append({"role": "assistant", "content": first["text"]})
            for index in range(args.groups_before, args.groups_before + args.groups_after):
                history += [message(f"Archive segment {index:03d}.\n" + unit * repeats),
                            {"role": "assistant", "content": "Recorded."}]
            history.append(message(QUESTION, current))
            # The answer must exist only in the historical image, never filler.
            texts = "\n".join(part["text"] for m in history if isinstance(m.get("content"), list)
                              for part in m["content"] if part.get("type") == "text")
            assert EXPECTED["reference"] not in texts and EXPECTED["name"] not in texts
            final = await call("above_budget_query", history)
            evidence["parsed_answer"] = answer(final["text"])
            evidence["answer_correct"] = evidence["parsed_answer"] == EXPECTED
            assert final["usage"]["prompt_tokens"] > 131072, "No budget crossing occurred"
            trace = args.engine_dir / "retrieval.jsonl"
            if engine["kvmem"] == "on":
                rows = [json.loads(line) for line in trace.read_text().splitlines()]
                tags = [row for row in rows if row.get("type") == "meta"
                        and row["trace_tag"] == output.name + "-above_budget_query"]
                assert tags, "Missing retrieval evidence for this request"
                meta = tags[-1]
                assert meta["selected"] < meta["block_count"] and meta["scorer_fallback_reason"] == "none"
                assert meta["query_ready"] and meta["index_ready"]
                evidence["retrieval"] = meta
            negative = json.loads(json.dumps(history))
            for item in negative:
                if isinstance(item.get("content"), list):
                    images = [p for p in item["content"] if p.get("type") == "image_url"]
                    if images:
                        images[0]["image_url"]["url"] = current
                        break
            absent = await call("absent_image_control", negative)
            evidence["negative_answer"] = answer(absent["text"])
            evidence["negative_correct"] = evidence["negative_answer"] == dict.fromkeys(EXPECTED)
            evidence["passed"] = evidence["answer_correct"] and evidence["negative_correct"]
        except BaseException as exc:
            evidence["error"] = repr(exc)
            raise
        finally:
            stopped.set(); await sampler
            (output / "result.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    if not evidence["passed"]:
        raise SystemExit("Historical screenshot answer was incorrect")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("apps/computer_use/.runtime/replay-corpus-01"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--group-tokens", type=int, default=2400)
    parser.add_argument("--groups-before", type=int, default=40)
    parser.add_argument("--groups-after", type=int, default=25)
    asyncio.run(run(parser.parse_args()))
