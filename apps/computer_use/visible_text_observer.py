"""Optional, read-only CPU text evidence; not enabled in frozen GUI campaigns.

This observer neither dispatches input nor changes accessibility predicates.
Every result belongs to one exact screenshot and observation. Missing text is
never evidence of absence, and an OCR match is not business-task acceptance.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


IDENTITY_FIELDS = ("desktop_id", "observation_id", "session_id", "display_id",
                   "sha256", "width", "height", "rotation")
OUTPUT_LIMIT = 512 * 1024


def identity(observation):
    result = {key: observation.get(key) for key in IDENTITY_FIELDS}
    if any(not isinstance(result[key], str) or not result[key]
           for key in ("desktop_id", "observation_id", "sha256")):
        raise ValueError("Screenshot observation identity is required")
    if not re.fullmatch(r"[a-f0-9]{64}", result["sha256"]):
        raise ValueError("Invalid screenshot digest")
    if any(type(result[key]) is not int or not 1 <= result[key] <= 8192
           for key in ("width", "height")) or result["width"] * result["height"] > 24_000_000:
        raise ValueError("Screenshot dimensions exceed the observation bound")
    return result


def region_bounds(region, observation):
    if (not isinstance(region, (list, tuple)) or len(region) != 4
            or any(type(value) is not int for value in region)):
        raise ValueError("Region must be four integer pixel coordinates")
    x, y, width, height = region
    if not (x >= 0 and y >= 0 and width > 0 and height > 0
            and x + width <= observation["width"] and y + height <= observation["height"]):
        raise ValueError("Region must be entirely inside the screenshot")
    return list(region)


def valid_lines(lines, region):
    if not isinstance(lines, list) or len(lines) > 128:
        raise ValueError("Too many OCR lines")
    x, y, width, height = region
    for line in lines:
        if (not isinstance(line, dict) or not isinstance(line.get("text"), str)
                or not 1 <= len(line["text"]) <= 2048):
            raise ValueError("Invalid OCR text")
        confidence = line.get("confidence")
        if (type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1):
            raise ValueError("Invalid OCR confidence")
        bounds = line.get("bounds")
        if (not isinstance(bounds, list) or len(bounds) != 4
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in bounds)):
            raise ValueError("Invalid OCR bounds")
        left, top, right, bottom = bounds
        if not (x <= left < right <= x + width and y <= top < bottom <= y + height):
            raise ValueError("OCR text is outside its evidence region")
    return lines


def positive_text_check(text, evidence, observation, *, minimum_confidence=0.95, maximum_lines=3):
    """Exact visible phrase, allowing only adjacent wrapped lines/whitespace.

    No fuzzy, substring, case, punctuation, phone, or absence equivalence.
    Cropped/outdated/low-confidence evidence cannot establish the phrase.
    """
    if not isinstance(text, str) or not 1 <= len(text) <= 512:
        raise ValueError("Expected text must contain 1..512 characters")
    if (type(minimum_confidence) not in (int, float) or not math.isfinite(minimum_confidence)
            or not 0 <= minimum_confidence <= 1 or type(maximum_lines) is not int
            or not 1 <= maximum_lines <= 3):
        raise ValueError("Invalid positive text check bounds")
    outcome = {"passed": False, "basis": "positive_visible_cpu_ocr", "matching_lines": [],
               "business_acceptance": False, "absence_supported": False}
    try:
        current = identity(observation)
        if evidence.get("status") != "ok" or evidence.get("identity") != current:
            return outcome | {"reason": "unavailable_or_stale_evidence"}
        region = region_bounds(evidence.get("region"), observation)
        lines = valid_lines(evidence.get("lines"), region)
    except (ValueError, AttributeError, TypeError, KeyError):
        return outcome | {"reason": "invalid_evidence"}
    ordered = sorted(enumerate(lines), key=lambda item: (item[1]["bounds"][1], item[1]["bounds"][0]))
    expected = " ".join(text.split())
    for start in range(len(ordered)):
        selected = []
        for index, line in ordered[start:start + maximum_lines]:
            if line["confidence"] < minimum_confidence:
                break
            if selected:
                previous = selected[-1][1]
                l, t, r, b = line["bounds"]
                pl, pt, pr, pb = previous["bounds"]
                line_height = max(pb - pt, b - t)
                # Wrapped lines share a left edge, have no overlap or column
                # jump, and are within one line height vertically.
                if not (t >= pb - line_height * 0.15 and t - pb <= line_height
                        and abs(l - pl) <= line_height * 0.75):
                    break
            selected.append((index, line))
            if " ".join(" ".join(item[1]["text"].split()) for item in selected) == expected:
                return outcome | {"passed": True, "matching_lines": [item[0] for item in selected],
                                  "minimum_observed_confidence": min(item[1]["confidence"] for item in selected)}
    return outcome | {"reason": "expected_phrase_not_established"}


async def _read_bounded(stream, limit):
    pieces = []
    length = 0
    while data := await stream.read(16384):
        length += len(data)
        if length > limit:
            raise ValueError("OCR worker output exceeded its bound")
        pieces.append(data)
    return b"".join(pieces)


async def observe_text(image_path, observation, recipe_path, *, python_executable=sys.executable,
                       dependency_path=None, region=None, timeout_seconds=8):
    """One owned subprocess, a bounded CPU timeout, and no device/model input."""
    current = identity(observation)
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 30:
        raise ValueError("OCR timeout must be0..30 seconds")
    selected_region = region_bounds([0, 0, current["width"], current["height"]] if region is None else region, current)
    fallback = {"identity": current, "region": selected_region, "lines": [],
                "basis": "positive_visible_cpu_ocr", "absence_supported": False}
    request = {"image_path": str(Path(image_path).resolve()), "observation": current,
               "recipe_path": str(Path(recipe_path).resolve()), "region": selected_region}
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    if dependency_path is not None:
        env["PYTHONPATH"] = str(Path(dependency_path).resolve())
    process = await asyncio.create_subprocess_exec(str(python_executable), "-u", str(Path(__file__).resolve()),
        "--worker", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, env=env, limit=OUTPUT_LIMIT)
    readers = []
    try:
        process.stdin.write(json.dumps(request).encode())
        await process.stdin.drain()
        process.stdin.close()
        # One newline-delimited response, rather than waiting for native-library
        # shutdown or inherited pipe handles to make stdout reach EOF.
        readers = [asyncio.create_task(_read_bounded(process.stderr, 65536))]
        stdout = await asyncio.wait_for(process.stdout.readline(), timeout_seconds)
        if len(stdout) > OUTPUT_LIMIT or not stdout.endswith(b"\n"):
            raise ValueError("Invalid OCR response frame")
        result = json.loads(stdout)
        if readers[0].done() and readers[0].exception() is not None:
            raise ValueError("OCR worker stderr exceeded its bound")
        if result.get("status") != "ok":
            return fallback | {"status": "unavailable", "error_type": result.get("error_type", "WorkerError")}
        if result.get("identity") != current or result.get("region") != selected_region:
            raise ValueError("Worker returned mismatched evidence")
        valid_lines(result.get("lines"), selected_region)
        return result
    except asyncio.TimeoutError:
        return fallback | {"status": "timeout"}
    except (ValueError, OSError, TypeError, KeyError):
        return fallback | {"status": "unavailable", "error_type": "InvalidWorkerEvidence"}
    finally:
        for reader in readers:
            if not reader.done():
                reader.cancel()
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        # Pipe lifecycle failures must not turn an8s read into an indefinite
        # observer task. A failed retirement is explicit and yields no proof.
        try:
            await asyncio.wait_for(process.wait(), 2)
            await asyncio.wait_for(asyncio.gather(*readers, return_exceptions=True), 1)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Owned OCR worker retirement is unconfirmed") from exc


def worker(request):
    """All weights are preinstalled and checked; automatic download is disabled."""
    import importlib.metadata
    import io
    import socket
    import time

    current = identity(request["observation"])
    region = region_bounds(request["region"], current)
    image_path = Path(request["image_path"])
    if image_path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Screenshot file exceeds its bound")
    png = image_path.read_bytes()
    if hashlib.sha256(png).hexdigest() != current["sha256"]:
        raise ValueError("Screenshot bytes do not match observation")
    recipe_path = Path(request["recipe_path"])
    if recipe_path.stat().st_size > 65536:
        raise ValueError("OCR recipe exceeds its bound")
    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    versions = {key: importlib.metadata.version(key) for key in ("rapidocr", "onnxruntime")}
    if versions != recipe["packages"]:
        raise ValueError("OCR engine version differs from the recipe")
    weights = {}
    if not isinstance(recipe["models"], list) or len(recipe["models"]) != 2:
        raise ValueError("Only two preinstalled OCR weights are allowed")
    for row in recipe["models"]:
        path = Path(row["path"])
        if not path.is_absolute() or path.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("OCR weight path or size is invalid")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("OCR weight checksum differs from recipe")
        weights[row["task"]] = str(path)
    if set(weights) != {"det", "rec"}:
        raise ValueError("Only the explicit detector and recognizer are allowed")

    def no_network(*_args, **_kwargs):
        raise RuntimeError("OCR worker does not allow network access")

    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    from PIL import Image
    import numpy as np
    import cv2
    import onnxruntime
    cv2.setNumThreads(1)
    onnxruntime.disable_telemetry_events()
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import EngineType, LangDet, LangRec, ModelType, OCRVersion

    with Image.open(io.BytesIO(png)) as image:
        if image.format != "PNG" or image.size != (current["width"], current["height"]):
            raise ValueError("Image format or dimensions differ from observation")
        x, y, width, height = region
        pixels = np.array(image.convert("RGB").crop((x, y, x + width, y + height)))[:, :, ::-1].copy()
    params = {"Global.use_cls": False, "Global.log_level": "error", "Global.text_score": 0.5,
              "EngineConfig.onnxruntime.intra_op_num_threads": 2,
              "EngineConfig.onnxruntime.inter_op_num_threads": 1}
    for task in ("Det", "Rec"):
        params.update({task + ".model_path": weights[task.lower()],
                       task + ".engine_type": EngineType.ONNXRUNTIME,
                       task + ".ocr_version": OCRVersion.PPOCRV4,
                       task + ".model_type": ModelType.MOBILE,
                       task + ".lang_type": LangDet.CH if task == "Det" else LangRec.EN})
    for flag in ("use_cuda", "use_dml", "use_cann", "use_coreml"):
        params["EngineConfig.onnxruntime." + flag] = False
    started = time.monotonic()
    output = RapidOCR(params=params)(pixels)
    lines = []
    if output.txts is not None:
        for text, confidence, polygon in zip(output.txts, output.scores, output.boxes):
            bounds = [float(polygon[:, 0].min()) + x, float(polygon[:, 1].min()) + y,
                      float(polygon[:, 0].max()) + x, float(polygon[:, 1].max()) + y]
            line = {"text": str(text), "confidence": float(confidence), "bounds": bounds}
            valid_lines([line], region)
            lines.append(line)
    valid_lines(lines, region)
    return {"status": "ok", "basis": "positive_visible_cpu_ocr", "identity": current,
            "region": region, "lines": lines, "seconds": time.monotonic() - started,
            "absence_supported": False, "engine": {"packages": versions, "gpu_used": False,
            "telemetry_enabled": False, "opencv_threads": 1,
            "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
            "weights_sha256": {row["task"]: row["sha256"] for row in recipe["models"]}}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.parse_args()
    try:
        encoded = sys.stdin.buffer.read(65537)
        if len(encoded) > 65536:
            raise ValueError("OCR request exceeds its bound")
        result = worker(json.loads(encoded))
    except Exception as exc:
        result = {"status": "unavailable", "error_type": type(exc).__name__}
    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    sys.stdout.flush()
