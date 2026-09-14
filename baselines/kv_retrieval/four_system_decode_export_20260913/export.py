"""Export existing measurements without rerunning inference or mixing versions."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
SOURCES = {
    "native_q8_mtp": BASE / "decode_comparison_20260912/q8_mtp_20260913/export/summary.csv",
    "external_initial": BASE / "mtp_qwen_20260913/export/summary.csv",
    "external_confirmation": BASE / "bottleneck_q8_mtp_20260913/thread_confirmation_summary.csv",
    "external_infllm_recheck": BASE / "bottleneck_q8_mtp_20260913/throughput_summary.csv",
}
NAMES = {"qw3": "qw3", "kvmem": "KVMem", "infllm": "InfLLM", "infinigen": "InfiniGen"}
records = []


def write_csv(name, rows, fields=None):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


for campaign, source in SOURCES.items():
    with source.open(newline="") as f:
        for line, row in enumerate(csv.DictReader(f), 2):
            if campaign == "external_confirmation" and not (
                row["mode"] == "optimized_both" and row["threads"] == "8"
            ):
                continue
            if campaign == "external_infllm_recheck" and row["method"] != "infllm":
                continue
            native = campaign == "native_q8_mtp"
            method = row["method"].split("_q8_")[0]
            mtp = int(row["method"].split("mtp")[1]) if native else int(row["mtp"])
            length = int(row["history_tokens"] if native else row["prompt_tokens"])
            suffix = "tokens_per_s" if campaign in ("native_q8_mtp", "external_initial") else "tps"
            record = {
                "system": NAMES[method], "history_tokens": length,
                "context_label": f"{length // 1024}K", "mtp_chain": mtp,
                "implementation": "native_qw3" if native else (
                    "qwen_optimized_both" if campaign == "external_confirmation" else "qwen_adapter"
                ),
                "torch_cpu_threads": row.get("threads", "" if native else "2"),
                "campaign": campaign, "repeats": int(row["n"]),
                "median_tokens_per_s": float(row[f"median_{suffix}"]),
                "min_tokens_per_s": float(row[f"min_{suffix}"]),
                "max_tokens_per_s": float(row[f"max_{suffix}"]),
                "generated_tokens": 128, "timed_output_tokens": 127,
                "source_csv": str(source), "source_line": line,
            }
            assert record["repeats"] == 3
            assert 0 < record["min_tokens_per_s"] <= record["median_tokens_per_s"] <= record["max_tokens_per_s"]
            records.append(record)

assert len(records) == 33
write_csv("measurements.csv", records)


def get(system, length, mtp, campaign):
    rows = [r for r in records if (r["system"], r["history_tokens"], r["mtp_chain"], r["campaign"])
            == (system, length, mtp, campaign)]
    assert len(rows) == 1, (system, length, mtp, campaign)
    return rows[0]["median_tokens_per_s"]


for name, lengths, latest, external_mtp in [
    ("latest_confirmed.csv", [32768, 245760], True, 0),
    ("latest_confirmed_mtp4.csv", [32768, 245760], True, 4),
    ("full_length_earlier.csv", [8192, 32768, 122880, 245760], False, 0),
    ("full_length_earlier_mtp4.csv", [8192, 32768, 122880, 245760], False, 4),
]:
    rows = []
    for length in lengths:
        rows.append({
            "history_tokens": length, "context_label": f"{length // 1024}K",
            "qw3_mtp4_tps": get("qw3", length, 4, "native_q8_mtp"),
            "kvmem_mtp4_tps": get("KVMem", length, 4, "native_q8_mtp"),
            f"infllm_mtp{external_mtp}_tps": get("InfLLM", length, external_mtp,
                "external_infllm_recheck" if latest else "external_initial"),
            f"infinigen_mtp{external_mtp}_tps": get("InfiniGen", length, external_mtp,
                "external_confirmation" if latest else "external_initial"),
        })
    write_csv(name, rows)

provenance = {}
(OUT / "sources").mkdir(exist_ok=True)
for campaign, source in SOURCES.items():
    provenance[campaign] = {
        "source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()
    }
    shutil.copyfile(source, OUT / "sources" / f"{campaign}.csv")
(OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.suffix != ".zip")
with zipfile.ZipFile(OUT / "four_system_decode_data.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(OUT))
print(json.dumps({"measurement_rows": len(records), "files": [p.name for p in files], "output": str(OUT)}))
