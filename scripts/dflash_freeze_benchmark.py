#!/usr/bin/env python3
"""Freeze the pinned official DFlash serving workload before performance tuning.

Run in an environment with datasets, transformers and huggingface_hub installed.
This downloads public datasets; it does not run inference or change the environment.
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import random
import subprocess

REFERENCE_REVISION = "07ebd93db9f472af339b644bb70221ad8428328a"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-source", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    source = args.reference_source.resolve()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != REFERENCE_REVISION:
        raise RuntimeError(f"unexpected reference revision: {head}")
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--", "dflash/benchmark.py"], check=True)
    benchmark_file = source / "dflash/benchmark.py"
    spec = importlib.util.spec_from_file_location("pinned_dflash_benchmark", benchmark_file)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError("completed manifest already exists; choose a new output directory")
    # Save revisions before downloads so a retry cannot silently advance them.
    pins_path = args.output / "dataset_revisions.json"
    if pins_path.exists():
        pins = json.loads(pins_path.read_text())
    else:
        api = HfApi()
        pins = {name: {"repository": cfg["load_args"][0],
                       "revision": api.dataset_info(cfg["load_args"][0]).sha}
                for name, cfg in official.DATASETS.items()}
        write_json(pins_path, pins)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    manifest = {
        "status": "frozen_inputs_not_benchmarked", "reference_revision": head,
        "benchmark_source_sha256": digest(benchmark_file),
        "num_prompts": 128, "concurrency": [1, 8, 32],
        "sample_order_seed": 42, "temperature": 1.0, "top_p": 0.95,
        "top_k": 20, "max_new_tokens": 4096, "reasoning_effort": "xhigh",
        "warmup": "next concurrency rows after the 128 measured rows; at most 64 output tokens",
        "mt_bench": "first user turn only, matching the pinned official OpenAI serving benchmark",
        "rendering": "official apply_chat_template; raw prompt/token IDs must be identical across engines",
        "tokenizer_directory": str(args.tokenizer.resolve()),
        "tokenizer_files": {}, "datasets": {},
    }
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        path = args.tokenizer / name
        if path.exists():
            manifest["tokenizer_files"][name] = digest(path)
    for name, cfg in official.DATASETS.items():
        pin = pins[name]
        if pin["repository"] != cfg["load_args"][0]:
            raise RuntimeError(f"repository pin mismatch: {name}")
        dataset = load_dataset(*cfg["load_args"], revision=pin["revision"], **cfg["load_kwargs"])
        order = list(range(len(dataset)))
        random.Random(42).shuffle(order)
        path = args.output / (name + ".jsonl")
        with path.open("w") as output:
            for ordinal in range(128 + 32):
                source_index = order[ordinal % len(order)]
                formatted = cfg["format"](dataset[source_index])
                turns = formatted if cfg.get("multi_turn") else [formatted]
                messages = [{"role": "user", "content": turns[0]}]
                prompt = official.apply_chat_template(tokenizer, messages, "xhigh")
                row = {"ordinal": ordinal, "source_index": source_index,
                       "role": "measured" if ordinal < 128 else "warmup",
                       "messages": messages, "prompt": prompt,
                       "input_ids": tokenizer.encode(prompt, add_special_tokens=False)}
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["datasets"][name] = {**pin, "load_args": cfg["load_args"],
            "load_kwargs": cfg["load_kwargs"], "source_rows": len(dataset),
            "file": path.name, "sha256": digest(path), "frozen_rows": 160}
        print(f"{name}: source_rows={len(dataset)} frozen_rows=160 sha256={digest(path)}", flush=True)
    write_json(manifest_path, manifest)
    print(f"Frozen workload: {manifest_path}")


if __name__ == "__main__":
    main()
