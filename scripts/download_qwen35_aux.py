#!/usr/bin/env python3
"""Extract official MTP and BF16 vision tensors using verified HTTP ranges.

No proxy variables, no full language checkpoint download. Original tensor bytes
are retained in a local safetensors file; the manifest records pinned provenance.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("size", choices=("9B", "4B"))
    parser.add_argument("--directory", type=Path, default=Path("models"))
    args = parser.parse_args()
    directory = args.directory / f"Qwen3.5-{args.size}-BF16"
    directory.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
    base = f"https://hf-mirror.com/Qwen/Qwen3.5-{args.size}/resolve"

    def curl(url, target, byte_range=None):
        headers = target.with_suffix(target.suffix + ".headers")
        command = ["curl", "-fsSL", "--retry", "5", "--connect-timeout", "30",
                   "--max-time", "600", "-D", str(headers), "-o", str(target)]
        if byte_range:
            start, end = byte_range
            # Resolver redirects have a small body larger than an 8-byte probe.
            command += ["--range", f"{start}-{end}", "--max-filesize", str(max(1 << 20, end - start + 1))]
        subprocess.run(command + [url], env=env, check=True)
        # Do not retain signed CDN redirect URLs in the evidence.
        parsed = {}
        for line in headers.read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                parsed[key.lower()] = value.strip()
        headers.unlink()
        if byte_range:
            prefix = f"bytes {start}-{end}/"
            if not parsed.get("content-range", "").startswith(prefix) or target.stat().st_size != end - start + 1:
                raise RuntimeError("server did not return the requested exact byte range")
        return parsed

    meta = curl(f"{base}/main/config.json", directory / "config.json")
    revision = meta["x-repo-commit"]
    base += "/" + revision
    for name in ("config.json", "model.safetensors.index.json", "preprocessor_config.json",
                 "tokenizer.json", "tokenizer_config.json"):
        curl(f"{base}/{name}", directory / name)
    original_index = json.loads((directory / "model.safetensors.index.json").read_text())
    (directory / "upstream.index.json").write_text(json.dumps(original_index, indent=2) + "\n")
    selected = {k: v for k, v in original_index["weight_map"].items()
                if k.startswith(("mtp.", "model.visual."))}
    by_shard = defaultdict(list)
    for name, shard in selected.items():
        by_shard[shard].append(name)
    new_header = {}
    records = []
    cursor = 0
    with tempfile.TemporaryDirectory(prefix="qw3-aux-", dir=directory) as temporary:
        temporary = Path(temporary)
        payload = temporary / "payload"
        with payload.open("wb") as output:
            for shard, names in sorted(by_shard.items()):
                url = f"{base}/{shard}"
                first = temporary / "first"
                curl(url, first, (0, 7))
                header_bytes = struct.unpack("<Q", first.read_bytes())[0]
                if not 0 < header_bytes < 1 << 28:
                    raise RuntimeError("invalid safetensors header size")
                header_file = temporary / "header"
                curl(url, header_file, (8, 7 + header_bytes))
                header = json.loads(header_file.read_bytes())
                ordered = sorted(names, key=lambda n: header[n]["data_offsets"][0])
                groups = []
                for name in ordered:
                    lo, hi = header[name]["data_offsets"]
                    if groups and lo == groups[-1][1]:
                        groups[-1][1] = hi
                        groups[-1][2].append(name)
                    else:
                        groups.append([lo, hi, [name]])
                for lo, hi, group in groups:
                    chunk = temporary / "chunk"
                    curl(url, chunk, (8 + header_bytes + lo, 7 + header_bytes + hi))
                    with chunk.open("rb") as data:
                        for name in group:
                            entry = header[name]
                            begin, end = entry["data_offsets"]
                            size = end - begin
                            new_header[name] = dict(entry, data_offsets=[cursor, cursor + size])
                            digest = hashlib.sha256()
                            remaining = size
                            while remaining:
                                part = data.read(min(8 << 20, remaining))
                                if not part:
                                    raise RuntimeError("truncated tensor range")
                                output.write(part)
                                digest.update(part)
                                remaining -= len(part)
                            cursor += size
                            records.append(dict(name=name, shard=shard, source_offsets=[begin, end],
                                                size=size, sha256=digest.hexdigest()))
                    print(f"{args.size}: extracted {len(group)} tensors, {hi - lo} bytes", flush=True)
        encoded = json.dumps(new_header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        target = directory / "model.aux.safetensors"
        staging = target.with_suffix(".partial")
        with staging.open("wb") as out, payload.open("rb") as data:
            out.write(struct.pack("<Q", len(encoded)))
            out.write(encoded)
            for part in iter(lambda: data.read(8 << 20), b""):
                out.write(part)
        staging.replace(target)
    index = dict(metadata=dict(total_size=cursor), weight_map={n: "model.aux.safetensors" for n in selected})
    (directory / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    manifest = dict(repository=f"Qwen/Qwen3.5-{args.size}", revision=revision,
                    resolver="https://hf-mirror.com", proxy_variables_removed=True,
                    extracted_tensor_bytes=cursor, tensors=records)
    (directory / "aux.provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.size}: complete, {len(records)} tensors, revision={revision}", flush=True)


if __name__ == "__main__":
    main()
