"""Export per-request replay metrics; keep first-pass and warm-image results separate."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text())
    rows = []
    groups = []
    pattern = re.compile(r"native generate: prompt_tokens=(\d+) prefill=([\d.]+)s .*? decoded=(\d+) decode=([\d.]+)s \(([\d.]+) tok/s\)")
    for profile in summary["profiles"]:
        label = f"{profile['method']}-kvmem-{profile['kvmem']}"
        directory = args.run_dir / label
        if profile["status"] != "completed":
            groups.append({"profile": label, "status": profile["status"], "error": profile.get("error")})
            continue
        generated = pattern.findall((directory / "engine/engine.log").read_text())
        if len(generated) != len(profile["runs"]):
            raise RuntimeError(f"{label}: engine/client request count mismatch")
        resources = [json.loads(line) for line in (directory / "resources.jsonl").read_text().splitlines()]
        for run, native in zip(profile["runs"], generated):
            usage = run.get("usage", {})
            if int(native[0]) != usage.get("prompt_tokens"):
                raise RuntimeError("Engine/client token counts differ")
            rows.append({"profile": label, "case": run["case"], "repetition": run["repetition"],
                         "status": run["status"], "prompt_tokens": usage.get("prompt_tokens"),
                         "completion_tokens": usage.get("completion_tokens"),
                         "first_output_seconds": run.get("first_output_seconds"),
                         "elapsed_seconds": run["elapsed_seconds"], "prefill_seconds": float(native[1]),
                         "decode_seconds": float(native[3]), "decode_tokens_per_second": float(native[4]),
                         "cpu_seconds": run.get("cpu_seconds"), "rss_bytes": run.get("process", {}).get("rss_bytes")})
        for case in summary["cases"]:
            trials = [row for row in rows if row["profile"] == label and row["case"] == case["name"]]
            warm = [row for row in trials if row["repetition"] > 0]
            groups.append({"profile": label, "case": case["name"], "status": "completed",
                           "completed": sum(t["status"] == "completed" for t in trials), "trials": len(trials),
                           "cold_first_output_seconds": trials[0]["first_output_seconds"],
                           "warm_first_output_seconds_median": statistics.median(t["first_output_seconds"] for t in warm) if warm else None,
                           "warm_decode_tokens_per_second_median": statistics.median(t["decode_tokens_per_second"] for t in warm) if warm else None,
                           "warm_elapsed_seconds_median": statistics.median(t["elapsed_seconds"] for t in warm) if warm else None,
                           "sampled_profile_gpu_peak_bytes": max(t.get("gpu_memory_bytes", 0) for t in resources),
                           "profile_rss_high_water_bytes": max(t.get("peak_rss_bytes", 0) for t in resources)})
    with (args.run_dir / "requests.csv").open("w", newline="") as output:
        if rows:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    (args.run_dir / "aggregate.json").write_text(json.dumps(groups, indent=2))
    for group in groups:
        if group["status"] == "completed":
            print(group["profile"], group["case"], f"{group['completed']}/{group['trials']}",
                  "warm_TTFT_s", round(group["warm_first_output_seconds_median"], 3) if group["warm_first_output_seconds_median"] else None,
                  "warm_decode_tps", group["warm_decode_tokens_per_second_median"],
                  "GPU_peak_GiB", round(group["sampled_profile_gpu_peak_bytes"] / 1024**3, 2))
        else:
            print(group)


if __name__ == "__main__":
    main()
