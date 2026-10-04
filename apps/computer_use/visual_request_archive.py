"""Preserve the exact converted source before selecting an image working set."""
import hashlib
import json
from pathlib import Path
import time

from .request_archive import encode, decode
from .visual_working_context import working_context


def present_request(body, pins, trace):
    if trace is None:
        raise ValueError("Image history selection requires a private full-source archive")
    started = time.monotonic()
    trace = Path(trace)
    blobs = trace.parent / "full-history-blobs"
    blobs.mkdir(exist_ok=True, mode=0o700)
    source = json.dumps(body, ensure_ascii=False).encode()
    tree = encode(body, blobs)
    restored = json.dumps(decode(tree, blobs), ensure_ascii=False).encode()
    if restored != source:
        raise ValueError("Full source history did not roundtrip; do not dispatch inference")
    archive = {"version": 1, "kind": "converted-full-source-before-image-selection",
               "bytes": len(source), "sha256": hashlib.sha256(source).hexdigest(), "tree": tree}
    target = trace / "request.full.archive.json"
    with target.open("x", encoding="utf-8") as file:
        file.write(json.dumps(archive, ensure_ascii=False))
    target.chmod(0o600)
    archived_at = time.monotonic()
    view, report = working_context(body, pins)
    selected_at = time.monotonic()
    presented = json.dumps(view, ensure_ascii=False).encode()
    report.update(source_sha256=archive["sha256"], source_bytes=len(source),
                  presented_sha256=hashlib.sha256(presented).hexdigest(), presented_bytes=len(presented),
                  full_source_archive_seconds=archived_at-started,
                  image_selection_seconds=selected_at-archived_at,
                  full_source_archive=target.name,
                  raw_responses_request_archived=False,
                  model_latency_improvement_measured=False,
                  business_success_improvement_measured=False)
    (trace / "image-context.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return view, report
