"""Summarize CUDA memory evidence without conflating counters and capacities.

Accepts retained non-instrumented logs but marks missing traces explicitly.
Use --require-traces for a separate allocation diagnostic, never to relabel
ordinary resource samples as a complete allocator measurement.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


def summarize(engine, samples=None):
    manifest = json.loads((engine / "manifest.json").read_text())
    raw = (engine / "engine.log").read_bytes()
    lines = raw.decode(errors="replace").splitlines()
    result = {"kind": "scoped_cuda_memory_evidence", "engine_manifest": manifest,
              "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "engine_log_sha256": hashlib.sha256(raw).hexdigest(),
              "complete_allocator_census": False,
              "scopes": {
                  "tensor_trace": "Owned CudaTensor allocations >=1 MiB only; excludes views, smaller allocations and allocations bypassing this wrapper.",
                  "workspace": "Named backend capacities, reported at backend end(); not every allocation or transient peak.",
                  "device_at_trace_events": "cudaMemGetInfo snapshots at traced tensor allocation/free events; not a continuously measured maximum.",
                  "resources": "Whole-device nvidia-smi samples and process RSS counters; distinct from capacities and allocator events.",
                  "components": "Reported weights, KV/index/table capacities and usage. Overlap exists; do not add component maxima or add them to whole-device samples."}}
    allocations = re.compile(r"\[cuda-tensor-allocation\] event=(allocate|free) label=(.*?) bytes=(\d+) device_used_bytes=(\d+) status=(\d+)")
    live = Counter()
    labels = defaultdict(lambda: {"allocation_events": 0, "free_events": 0, "current_bytes": 0, "peak_live_bytes": 0})
    traced_live = traced_peak = device_peak = trace_events = 0
    errors, workspaces, attention, components = [], [], [], []

    def fields(line, unit=False):
        pattern = r"(\w+)=([\d.]+) MiB" if unit else r"(\w+)=(\d+)(?:\s|$)"
        return {k: float(v) if unit else int(v) for k, v in re.findall(pattern, line)}

    for number, line in enumerate(lines, 1):
        match = allocations.search(line)
        if match:
            event, label, size, device, status = match.groups()
            size, device, status = int(size), int(device), int(status)
            trace_events += 1
            key = (label, size)
            if event == "free" and live[key] == 0:
                errors.append({"line": number, "error": "Free without a matching traced allocation", "label": label, "bytes": size})
                continue
            delta = size if event == "allocate" else -size
            live[key] += 1 if event == "allocate" else -1
            traced_live += delta
            traced_peak = max(traced_peak, traced_live)
            group = labels[label]
            group["allocation_events" if event == "allocate" else "free_events"] += 1
            group["current_bytes"] += delta
            group["peak_live_bytes"] = max(group["peak_live_bytes"], group["current_bytes"])
            if status == 0:
                device_peak = max(device_peak, device)
            else:
                errors.append({"line": number, "error": "cudaMemGetInfo failed", "status": status})
        elif "CUDA workspace:" in line:
            values = fields(line, True)
            if not {"total", "linear", "attention"}.issubset(values):
                errors.append({"line": number, "error": "Incomplete workspace snapshot"})
            workspaces.append({"line": number, "capacity_mib": values})
        elif "CUDA attention workspace:" in line:
            values = fields(line, True)
            if not values:
                errors.append({"line": number, "error": "Empty attention workspace snapshot"})
            attention.append({"line": number, "capacity_mib": values})
        elif "[cuda-tensor-allocation]" in line:
            errors.append({"line": number, "error": "Malformed tensor allocation record"})
        elif "native load:" in line:
            components.append({"line": number, "kind": "target_weights", "reported_mib": fields(line, True)})
        elif "native DFlash 2:" in line:
            components.append({"line": number, "kind": "draft_weights", "reported": fields(line)})
        elif "native CUDA vision:" in line:
            components.append({"line": number, "kind": "vision_weights", "reported_mib": fields(line, True)})
        elif any(marker in line for marker in ("[kvmem-tier-usage]", "[bs-kbar-index]", "[kvmem-assembly]", "[bs-qcap-host]")):
            marker = next(m for m in ("kvmem-tier-usage", "bs-kbar-index", "kvmem-assembly", "bs-qcap-host") if "[" + m + "]" in line)
            components.append({"line": number, "kind": marker, "evidence": line})
        elif "[kvmem-io]" in line and "pinned_mib=" in line:
            components.append({"line": number, "kind": "pinned_io_slabs", "evidence": line})
    result["tensor_trace"] = {"present": bool(trace_events), "events": trace_events,
        "peak_traced_live_bytes": traced_peak if trace_events else None,
        "end_traced_live_bytes": traced_live if trace_events else None,
        "device_used_peak_at_trace_events_bytes": device_peak if trace_events else None,
        "labels": dict(sorted(labels.items())), "errors": errors}
    result["workspace_trace"] = {"present": bool(workspaces and attention),
        "workspace_snapshots": workspaces, "attention_snapshots": attention,
        "max_reported_total_mib": max((r["capacity_mib"].get("total", 0) for r in workspaces), default=None)}
    result["reported_components"] = components
    if samples:
        records = [json.loads(line) for line in samples.read_text().splitlines()]
        result["resource_samples"] = {"count": len(records),
            "sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
            "sampled_device_gpu_peak_bytes": max((r.get("gpu_memory_bytes", 0) for r in records), default=None),
            "process_rss_high_water_bytes": max((r.get("peak_rss_bytes", 0) for r in records), default=None)}
    env = manifest["qw3_env"]
    result["allocation_diagnostic_available"] = (bool(trace_events) and bool(workspaces and attention) and not errors
        and env.get("QW3_CUDA_TENSOR_ALLOC_TRACE") == "1" and env.get("QW3_CUDA_WORKSPACE_TRACE") == "1")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine_dir", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-traces", action="store_true")
    args = parser.parse_args()
    report = summarize(args.engine_dir, args.samples)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"allocation_diagnostic_available": report["allocation_diagnostic_available"],
                      "reported_components": len(report["reported_components"]),
                      "trace_events": report["tensor_trace"]["events"]}))
    raise SystemExit(1 if args.require_traces and not report["allocation_diagnostic_available"] else 0)
