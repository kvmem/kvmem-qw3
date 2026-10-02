"""Frozen follow-up of the P1 coordinate-unit finding, with new target controls."""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import random
import time
from urllib.parse import urlparse

import aiohttp
from PIL import Image

from qw3_computer.coordinates import normalized_point
from qw3_computer.protocol import read_sse
from visual_diagnostic_run import APPLICATIONS, digest, request_for, save, score


def request(sample, image, variant):
    if variant == "pixels":
        return request_for(sample, image, "original")[0]
    modified = sample | {"question": sample["question"].replace("原屏幕像素坐标", "位置"),
        "answer_schema": {"coordinate_space": "normalized_1000", "x": "integer 0..1000", "y": "integer 0..1000"}}
    body = request_for(modified, image, "original")[0]
    body["messages"][0]["content"][0]["text"] += (
        "\n定位契约：横轴、纵轴都用 0 到 1000 的归一化坐标。左上为 (0,0)，右下为 (1000,1000)。"
        "请直接输出归一化坐标，不要换算为屏幕像素；程序会按实际截图尺寸转换。"
        "coordinate_space 必须为 normalized_1000。")
    return body


def grade(sample, answer, variant, dimensions):
    if variant == "pixels":
        return score(sample, answer)
    try:
        value = answer.strip()
        if value.startswith("```json\n") and value.endswith("\n```"):
            value = value[8:-4]
        raw = json.loads(value)
        if not isinstance(raw, dict) or set(raw) != {"coordinate_space", "x", "y"} or raw["coordinate_space"] != "normalized_1000":
            raise ValueError("Missing explicit coordinate-space contract")
        pixel = normalized_point(raw["x"], raw["y"], *dimensions)
        return score(sample, json.dumps(pixel)) | {"raw_answer": raw, "pixel_answer": pixel}
    except (ValueError, TypeError, KeyError) as exc:
        return {"passed": False, "format_valid": False, "error": str(exc)}


async def run(args):
    corpus = json.loads(args.corpus.read_text())
    samples = corpus["samples"]
    if len(samples) != 12 or len({s["id"] for s in samples}) != 12:
        raise ValueError("Expected six development and six new-target samples")
    for split in ("development", "heldout"):
        if Counter(s["app"] for s in samples if s["split"] == split) != Counter({a: 1 for a in APPLICATIONS}):
            raise ValueError("Each split must cover all six apps")
    for sample in samples:
        path = (args.corpus.parent / sample["image"]).resolve()
        if not path.is_relative_to(args.corpus.parent.resolve()) or digest(path) != sample["image_sha256"] or not sample["ground_truth_reviewed"]:
            raise ValueError("Frozen reviewed screenshot required")
        with Image.open(path) as image:
            left, top, right, bottom = sample["expected_box"]
            if not 0 <= left < right <= image.width or not 0 <= top < bottom <= image.height:
                raise ValueError("Invalid target bounds")
    engine = json.loads((args.engine / "manifest.json").read_text())
    actual = [v.decode() for v in Path(f"/proc/{engine['pid']}/cmdline").read_bytes().split(b"\0") if v]
    endpoint = urlparse(args.endpoint)
    port = int(engine["command"][engine["command"].index("--port") + 1])
    if actual != engine["command"] or engine["kvmem"] != "off" or endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost"} or endpoint.port != port or endpoint.path:
        raise ValueError("Owned local baseline engine required")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    order = [(s, variant) for s in samples for variant in ("pixels", "normalized_1000")]
    random.Random(20261004).shuffle(order)
    report = {"kind": "P1_coordinate_contract_followup", "status": "running", "started_at": time.time(),
              "corpus_sha256": digest(args.corpus), "source_sha256": digest(Path(__file__)),
              "conversion_sha256": digest(Path(__file__).parent / "qw3_computer/coordinates.py"),
              "engine": engine, "planned_requests": 24, "trials": [],
              "order": [{"sample": s["id"], "variant": v} for s, v in order]}
    save(out / "result.json", report)
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=180, sock_read=90)) as client:
        for index, (sample, variant) in enumerate(order):
            directory = out / f"{index:02d}"; directory.mkdir()
            trial = {"sample": sample["id"], "app": sample["app"], "split": sample["split"], "variant": variant}
            answer = ""; reasoning = ""; done = False; started = time.monotonic()
            try:
                path = args.corpus.parent / sample["image"]
                if digest(path) != sample["image_sha256"]: raise ValueError("Screenshot changed")
                with Image.open(path) as image: dimensions = image.size
                body = request(sample, path, variant); save(directory / "request.json", body)
                trial["request_sha256"] = digest(directory / "request.json")
                async with client.post(args.endpoint + "/v1/chat/completions", json=body) as response:
                    response.raise_for_status()
                    with (directory / "stream.jsonl").open("w") as stream:
                        async for data in read_sse(response.content):
                            if data == "[DONE]": done = True; break
                            stream.write(data + "\n"); stream.flush(); chunk = json.loads(data)
                            if chunk.get("usage"): trial["usage"] = chunk["usage"]
                            for choice in chunk.get("choices", []):
                                delta = choice.get("delta", {})
                                if delta.get("content"): trial.setdefault("first_text_seconds", time.monotonic() - started)
                                answer += delta.get("content") or ""
                                reasoning += delta.get("reasoning_content") or ""
                                if choice.get("finish_reason"): trial["finish_reason"] = choice["finish_reason"]
                if not done or reasoning: raise ValueError("Incomplete stream or unexpected thinking")
                trial.update(status="complete", answer=answer, score=grade(sample, answer, variant, dimensions))
                if trial.get("finish_reason") == "length": trial["score"]["passed"] = False
            except Exception as exc:
                trial.update(status="infrastructure_error", answer=answer, error=f"{type(exc).__name__}: {exc}")
            trial["elapsed_seconds"] = time.monotonic() - started
            save(directory / "result.json", trial); report["trials"].append(trial); save(out / "result.json", report)
            if trial["status"] != "complete": report["status"] = "halted"; break
        else: report["status"] = "complete"
    report["finished_at"] = time.time(); save(out / "result.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "engine", "output"): parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    result = asyncio.run(run(parser.parse_args()))
    print(json.dumps({"status": result["status"], "requests": len(result["trials"])}))
