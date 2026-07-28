#!/usr/bin/env python3
"""Measure actual peak GPU memory during over-budget prefill on the kvmem server.

Isolates the transient prefill scratch (the thing the flat 15 GiB reserve guards)
by POSTing prompts of increasing token length (all > --kvmem-budget 32768) with a
tiny max_tokens, while sampling nvidia-smi in a background thread to catch the peak.

Inference is done ONLY over the OpenAI HTTP API. nvidia-smi is used solely for
measurement. Reports scratch_delta = peak - idle per prompt size, which reveals
whether prefill scratch is window-bound (flat) or ctx-bound (grows with prompt).
"""
import json
import subprocess
import threading
import time
import urllib.request

BASE = "http://localhost:8080"
IDLE_MIB = None  # measured at start


def gpu_used_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(out.strip().splitlines()[0])


class PeakSampler(threading.Thread):
    def __init__(self, interval=0.12):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._stopev = threading.Event()

    def run(self):
        while not self._stopev.is_set():
            try:
                u = gpu_used_mib()
                if u > self.peak:
                    self.peak = u
            except Exception:
                pass
            self._stopev.wait(self.interval)

    def stop(self):
        self._stopev.set()
        self.join(timeout=2)


def build_prompt(approx_tokens):
    # ~1 token per word-ish for this filler; we read back the real count from usage.
    # Use varied text so it isn't a single repeated token (avoids degenerate paths).
    chunk = (
        "The quick brown fox jumps over the lazy dog while counting numbers "
        "one two three four five six seven eight nine ten in a long meandering "
        "sentence about logistics, weather patterns, and the history of trade. "
    )
    # chunk is ~40 tokens; scale up.
    reps = max(1, approx_tokens // 40)
    return chunk * reps


def run_one(approx_tokens, max_tokens=8):
    prompt = build_prompt(approx_tokens)
    body = json.dumps(
        {
            "model": "Qwen3.6-27B-Q8_0.gguf",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    sampler = PeakSampler()
    sampler.start()
    t0 = time.time()
    err = None
    prompt_tokens = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            data = json.loads(resp.read())
        prompt_tokens = data.get("usage", {}).get("prompt_tokens")
    except Exception as e:
        err = str(e)
    dt = time.time() - t0
    time.sleep(0.3)  # let sampler catch a trailing peak
    sampler.stop()
    return {
        "approx": approx_tokens,
        "prompt_tokens": prompt_tokens,
        "peak_mib": sampler.peak,
        "delta_mib": (sampler.peak - IDLE_MIB) if IDLE_MIB else None,
        "secs": round(dt, 1),
        "err": err,
    }


def main():
    global IDLE_MIB
    time.sleep(1.0)
    IDLE_MIB = gpu_used_mib()
    print(f"idle_resident_mib = {IDLE_MIB}", flush=True)
    # All targets are > budget 32768 so we exercise the over-budget/tiered regime.
    targets = [30000, 48000, 64000, 96000, 128000, 150000]
    results = []
    for t in targets:
        r = run_one(t)
        results.append(r)
        print(
            f"approx={r['approx']:>7}  prompt_tokens={str(r['prompt_tokens']):>7}  "
            f"peak={r['peak_mib']:>6} MiB  scratch_delta={str(r['delta_mib']):>6} MiB  "
            f"{r['secs']:>6}s  err={r['err']}",
            flush=True,
        )
    print("\nJSON:")
    print(json.dumps({"idle_mib": IDLE_MIB, "results": results}, indent=2))


if __name__ == "__main__":
    main()
