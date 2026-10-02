"""Frozen neutral-screen diagnostics: original versus fixed enlarged quadrants.

Answers and boxes stay in the host grader. This does not control a desktop and
must not be reported as task completion or a test of a whole agent workflow.
"""
import argparse
import asyncio
import base64
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import time
from urllib.parse import urlparse

import aiohttp
from PIL import Image

from qw3_computer.protocol import read_sse


KINDS = {"text", "point", "state", "spatial"}
APPLICATIONS = {"Edge", "Notepad", "Calc", "Writer", "Impress", "GIMP"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def image_item(pixels):
    stream = io.BytesIO()
    pixels.save(stream, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()}}


def request_for(sample, path, variant):
    with Image.open(path) as source:
        screen = source.convert("RGB")
    width, height = screen.size
    content = [{"type": "text", "text":
        f"屏幕原始尺寸为 {width}×{height} 像素。请仅根据提供的屏幕回答；不要操作软件。\n"
        + sample["question"] + "\n仅返回符合下列结构的 JSON，不要附带说明：\n"
        + json.dumps(sample["answer_schema"], ensure_ascii=False)}]
    if variant == "original":
        content.append(image_item(screen))
        areas = [{"box": [0, 0, width, height], "scale": 1, "input_pixels": width * height}]
    elif variant == "quadrants":
        areas = []
        for row in range(2):
            for column in range(2):
                box = [column * width // 2, row * height // 2, (column + 1) * width // 2, (row + 1) * height // 2]
                tile = screen.crop(box)
                tile = tile.resize((tile.width * 2, tile.height * 2), Image.Resampling.LANCZOS)
                # Every screen uses the same four tiles; the target box is not
                # consulted for cropping. Coordinate origins are not answers.
                content.append({"type": "text", "text":
                    f"下图为原屏幕区域 [{box[0]},{box[1]},{box[2]},{box[3]}] 的 2 倍放大图。"
                    "若回答坐标，必须映射回原屏幕：原坐标=本图像素坐标/2+区域左上角坐标。"})
                content.append(image_item(tile))
                areas.append({"box": box, "scale": 2, "input_pixels": tile.width * tile.height})
    else:
        raise ValueError("Unknown visual variant")
    return {"model": "qw3", "messages": [{"role": "user", "content": content}], "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": 4096, "enable_thinking": False,
            "temperature": 0, "top_p": 1}, areas


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def score(sample, text):
    result = {"passed": False, "format_valid": False}
    try:
        value = text.strip()
        if value.startswith("```json\n") and value.endswith("\n```"):
            value = value[8:-4]
        answer = json.loads(value)
        if not isinstance(answer, dict):
            raise ValueError("Answer must be an object")
        result["answer"] = answer
        if sample["kind"] == "point":
            if set(answer) != {"x", "y"} or any(type(answer[k]) not in (int, float) or not math.isfinite(answer[k]) for k in answer):
                raise ValueError("Point must contain two finite pixel coordinates")
            x, y = answer["x"], answer["y"]
            left, top, right, bottom = sample["expected_box"]
            result.update(format_valid=True, passed=left <= x < right and top <= y < bottom,
                          center_distance_pixels=math.hypot(x - (left + right) / 2, y - (top + bottom) / 2))
        elif sample["kind"] == "text":
            if set(answer) != {"text"} or not isinstance(answer["text"], str):
                raise ValueError("Text answer must be a single string field")
            actual, expected = answer["text"].strip(), sample["expected"]["text"].strip()
            edits = edit_distance(expected, actual)
            result.update(format_valid=True, passed=actual == expected, character_edits=edits,
                          expected_characters=len(expected), character_accuracy=max(0, 1 - edits / max(1, len(expected))))
        else:
            expected = sample["expected"]
            matches = {key: key in answer and type(answer[key]) is type(value) and answer[key] == value for key, value in expected.items()}
            result.update(format_valid=set(answer) == set(expected), field_matches=matches,
                          passed=set(answer) == set(expected) and all(matches.values()))
    except (ValueError, TypeError, KeyError) as exc:
        result["error"] = str(exc)
    return result


def validate_corpus(corpus, directory):
    samples = corpus["samples"]
    if len(samples) != 24 or len({s["id"] for s in samples}) != 24:
        raise ValueError("P1 requires 24 unique frozen neutral samples")
    if Counter(s["kind"] for s in samples) != Counter({kind: 6 for kind in KINDS}):
        raise ValueError("P1 requires six samples in each of its four categories")
    for kind in KINDS:
        if Counter(s["split"] for s in samples if s["kind"] == kind) != Counter(development=3, heldout=3):
            raise ValueError("Each category must retain three development and three held-out samples")
        if Counter(s["app"] for s in samples if s["kind"] == kind) != Counter({app: 1 for app in APPLICATIONS}):
            raise ValueError("Every category must cover all six planned applications")
    for sample in samples:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", sample["id"]):
            raise ValueError("Unsafe sample identity")
        path = (directory / sample["image"]).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != sample["image_sha256"]:
            raise ValueError("Frozen neutral screenshot identity changed")
        if not sample.get("ground_truth_reviewed") or not sample.get("question") or not sample.get("answer_schema"):
            raise ValueError("Screen, question, answer schema and reviewed ground truth are required before running")
        with Image.open(path) as screen:
            if sample["kind"] == "point":
                left, top, right, bottom = sample["expected_box"]
                if not (0 <= left < right <= screen.width and 0 <= top < bottom <= screen.height):
                    raise ValueError("Ground-truth box is outside the original screen")
            elif not isinstance(sample.get("expected"), dict) or not sample["expected"]:
                raise ValueError("Nonempty structured ground truth is required")
    return samples


async def run(args):
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    samples = validate_corpus(corpus, args.corpus.parent)
    engine = json.loads((args.engine / "manifest.json").read_text())
    actual = [v.decode() for v in Path(f"/proc/{engine['pid']}/cmdline").read_bytes().split(b"\0") if v]
    if actual != engine["command"] or engine["kvmem"] != "off":
        raise ValueError("P1 requires the owned live KVMem-off baseline engine")
    endpoint = urlparse(args.endpoint)
    port = int(engine["command"][engine["command"].index("--port") + 1])
    if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost"} or endpoint.port != port or endpoint.path:
        raise ValueError("P1 must use the manifest's local QW3 endpoint")
    order = [(s, variant) for s in samples for variant in ("original", "quadrants")]
    random.Random(20261003).shuffle(order)
    report = {"kind": "P1_neutral_visual_diagnostic", "status": "running", "model_used": True,
              "agent_task_success_claimed": False, "corpus_sha256": digest(args.corpus), "engine": engine,
              "source_sha256": digest(Path(__file__)), "temperature": 0, "thinking": False,
              "planned_requests": 48, "started_at": time.time(), "trials": [],
              "order": [{"sample": s["id"], "variant": variant} for s, variant in order]}
    save(output / "result.json", report)
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=180, sock_read=90)) as client:
        for index, (sample, variant) in enumerate(order):
            directory = output / f"{index:02d}-{sample['id']}-{variant}"; directory.mkdir()
            if digest(args.corpus.parent / sample["image"]) != sample["image_sha256"]:
                raise ValueError("Screenshot changed after corpus validation; no further requests dispatched")
            body, areas = request_for(sample, args.corpus.parent / sample["image"], variant)
            save(directory / "request.json", body)
            trial = {"sample": sample["id"], "variant": variant, "kind": sample["kind"], "app": sample["app"],
                     "split": sample["split"], "status": "running", "areas": areas,
                     "request_sha256": digest(directory / "request.json"), "started_at": time.time()}
            answer = ""; reasoning = ""; done = False; started = time.monotonic()
            try:
                async with client.post(args.endpoint + "/v1/chat/completions", json=body) as response:
                    response.raise_for_status()
                    with (directory / "stream.jsonl").open("w", encoding="utf-8") as stream:
                        async for data in read_sse(response.content):
                            if data == "[DONE]":
                                done = True; break
                            stream.write(data + "\n"); stream.flush()
                            chunk = json.loads(data)
                            if chunk.get("usage"): trial["usage"] = chunk["usage"]
                            for choice in chunk.get("choices", []):
                                delta = choice.get("delta", {})
                                if delta.get("content"): trial.setdefault("first_text_seconds", time.monotonic() - started)
                                answer += delta.get("content") or ""
                                reasoning += delta.get("reasoning_content") or ""
                                if choice.get("finish_reason"): trial["finish_reason"] = choice["finish_reason"]
                if not done or reasoning:
                    raise ValueError("Stream did not terminate cleanly or thinking-off was ignored")
                trial.update(status="complete", answer=answer, score=score(sample, answer))
                if trial.get("finish_reason") == "length":
                    trial["score"]["passed"] = False
                    trial["budget_exhausted"] = True
            except Exception as exc:
                trial.update(status="infrastructure_error", error=f"{type(exc).__name__}: {exc}", answer=answer)
            trial["elapsed_seconds"] = time.monotonic() - started
            save(directory / "result.json", trial)
            report["trials"].append(trial); save(output / "result.json", report)
            if trial["status"] != "complete":
                report["status"] = "halted_for_transport_analysis"
                break
        else:
            report["status"] = "complete"
    report["finished_at"] = time.time(); save(output / "result.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    result = asyncio.run(run(parser.parse_args()))
    print(json.dumps({"status": result["status"], "requests": len(result["trials"])}))
