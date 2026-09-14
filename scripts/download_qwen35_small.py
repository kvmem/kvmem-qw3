#!/usr/bin/env python3
"""Download pinned Unsloth Q8 GGUFs through hf-mirror, without proxy variables."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("size", choices=("9B", "4B"))
    parser.add_argument("--directory", type=Path, default=Path("models"))
    args = parser.parse_args()
    env = {k: v for k, v in os.environ.items()
           if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
    args.directory.mkdir(parents=True, exist_ok=True)
    name = f"Qwen3.5-{args.size}-Q8_0.gguf"
    repo = f"unsloth/Qwen3.5-{args.size}-GGUF"
    base = f"https://hf-mirror.com/{repo}/resolve"
    # Inspect the resolver response, not the signed CDN URL, for provenance.
    result = subprocess.run(
        ["curl", "--fail", "--silent", "--show-error", "--head",
         "--retry", "4", "--max-time", "60", f"{base}/main/{name}"],
        env=env, check=True, capture_output=True, text=True)
    headers = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.lower()] = value.strip().strip('"')
    revision = headers["x-repo-commit"]
    expected_sha = headers["x-linked-etag"]
    expected_size = int(headers["x-linked-size"])
    target = args.directory / name
    temporary = target.with_suffix(".gguf.partial")
    if not target.exists():
        subprocess.run(
            ["curl", "--fail", "--location", "--silent", "--show-error",
             "--retry", "8", "--retry-delay", "3", "--connect-timeout", "30",
             "--speed-time", "120", "--speed-limit", "1024",
             "--continue-at", "-", "--output", str(temporary),
             f"{base}/{revision}/{name}"], env=env, check=True)
        check_path = temporary
    else:
        check_path = target
    digest = hashlib.sha256()
    with check_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    if check_path.stat().st_size != expected_size or digest.hexdigest() != expected_sha:
        raise RuntimeError(f"size or SHA-256 mismatch: {check_path}")
    if check_path == temporary:
        temporary.replace(target)
    manifest = dict(repository=repo, revision=revision, file=name,
                    size_bytes=expected_size, sha256=expected_sha,
                    resolver="https://hf-mirror.com", proxy_variables_removed=True,
                    verified_at=time.time())
    target.with_suffix(".provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
