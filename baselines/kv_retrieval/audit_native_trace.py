"""Audit native fixed-trace state reuse; shared timings stay non-comparative."""
import argparse
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = json.loads((a.run / "result.json").read_text())
    log = (a.run / "server.log").read_text()
    rows = [json.loads(line) for line in (a.run / "measurements.jsonl").read_text().splitlines()]
    reuses = []
    captures = []
    for line in log.splitlines():
        hit = re.search(r'prefix-cache HIT \(mtp\): reuse=(\d+) ckpt=([PM]) D=(\d+)', line)
        if hit:
            origin, kind, lcp = int(hit[1]), hit[2], int(hit[3])
            reuses.append({"origin": origin, "kind": kind, "lcp": lcp})
            if origin > lcp or (kind == "P" and origin % 32):
                raise AssertionError("unsafe replay origin: " + line)
        capture = re.search(r'prefix-cache CAPTURE \(mtp\): .*?pos=(\d+) P=(\d+) P_resumable=(\d+)', line)
        if capture:
            end, origin, ready = map(int, capture.groups())
            captures.append({"end": end, "P": origin, "ready": bool(ready)})
            if ready and (origin > end or origin % 32):
                raise AssertionError("unsafe prompt checkpoint: " + line)
    guided = [line for line in log.splitlines() if 'native kvmem guided query: mode=' in line]
    report = {"run": str(a.run.resolve()), "status": result["status"],
        "completed_requests": sum(row["status"] == "completed" for row in rows),
        "failed_requests": sum(row["status"] != "completed" for row in rows),
        "reuse_hits": len(reuses), "reuse_origins": reuses, "checkpoint_captures": captures,
        "unexpected_checkpoint_misses": log.count('reason=no_reusable_checkpoint') + log.count('reason=checkpoint_after_query_boundary'),
        "guided_query_attempts": len(guided), "guided_fallback_original": sum('mode=fallback-original' in line for line in guided),
        "capacity_recovery_events": [line for line in log.splitlines() if '-capacity-recovery]' in line],
        "native_binary_sha256": result.get("native_binary_sha256"),
        "peak_process_gpu_gib": result.get("peak_process_gpu_gib"),
        "performance_claim": False, "shared_gpu": result["shared_gpu"], "task_result": False}
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"reuse_origins", "checkpoint_captures"}}, indent=2))


if __name__ == "__main__":
    main()
