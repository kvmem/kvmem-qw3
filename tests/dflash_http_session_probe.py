#!/usr/bin/env python3
"""Replay two real, growing sessions against an already running KVMem server.

The first four frozen long-concurrency inputs must be A0, A1, B0, B1.
Compare rotating (cold) keys with retained keys under identical server options.
Wire chunks are not token timestamps; their rates are deliberately not reported
as native decode throughput. Server logs must separately prove the route, cache
hits, K/G configuration and actual joint verification.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--resident-only", action="store_true",
                        help="Validate resident HTTP routing without repeating a cold-cache performance comparison")
    parser.add_argument("--serial-session-reference", action="store_true",
                        help="Run A0,A1,B0,B1 without cache keys against the legacy serial backend")
    parser.add_argument("--serial-controller-reference", action="store_true",
                        help="Run A0,A1,B0,B1 with resident keys against the same continuous controller")
    parser.add_argument("--reference-responses", type=Path,
                        help="Compare with resident responses saved by a previous C2 run")
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 4096:
        parser.error("max tokens must be in 1..4096")
    serial_reference = args.serial_session_reference or args.serial_controller_reference
    if args.resident_only and (serial_reference or args.reference_responses):
        parser.error("resident-only cannot be combined with serial/reference options")
    if args.serial_session_reference and args.serial_controller_reference:
        parser.error("choose one reference execution route")
    if serial_reference != bool(args.reference_responses):
        parser.error("serial session reference requires --reference-responses (and vice versa)")
    raw = args.config.read_bytes()
    config = json.loads(raw)
    assert config["policy"] == {"k_tokens": 131072, "g_tokens": 65536}
    rows = config["requests"][:4]
    assert len(rows) == 4 and len({row["id"] for row in rows}) == 4
    for first, second in ((0, 1), (2, 3)):
        prefix = rows[first]["input_ids"]
        assert len(prefix) > 131072, "requires above-budget histories"
        assert rows[second]["input_ids"][:len(prefix)] == prefix, "history is not an append"
        assert len(rows[second]["input_ids"]) > len(prefix)
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.base_url.rstrip("/")
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    summary = {"status": "running", "model": model,
               "config_sha256": hashlib.sha256(raw).hexdigest(),
               "input_ids": [row["id"] for row in rows],
               "concurrency": 1 if serial_reference else 2,
               "serial_session_reference": args.serial_session_reference,
               "serial_controller_reference": args.serial_controller_reference,
               "resident_only": args.resident_only,
               "max_tokens": args.max_tokens,
               "natural_eos": True, "seed": config["seed"], "groups": [],
               "server_route_and_cache_evidence_required": True,
               "performance_qualified": False}

    def save():
        path = args.output / "summary.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(path)

    def request(index, key, barrier):
        row = rows[index]
        body = dict(model=model, prompt=row["input_ids"],
                    max_tokens=args.max_tokens, temperature=1., top_k=20, top_p=.95,
                    min_p=0., repetition_penalty=1., presence_penalty=0.,
                    sampling_filter_order="top-k-first", seed=config["seed"],
                    ignore_eos=False, enable_thinking=True, thinking_budget=28672,
                    kvmem_reselect="force",
                    kvmem_query_token_span=dict(begin=row["query_begin"], end=row["query_end"]),
                    kvmem_trace_tag=key or ("serial-" + row["id"]),
                    stream=True, stream_options=dict(include_usage=True))
        if key is not None:
            body["kvmem_prefix_cache_key"] = key
        req = urllib.request.Request(base + "/v1/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        barrier.wait()
        started = time.monotonic_ns()
        chunks, done, usage, first_text = [], False, None, None
        try:
            with urllib.request.urlopen(req, timeout=1800) as response:
                for wire in response:
                    if not wire.startswith(b"data: "):
                        continue
                    at = time.monotonic_ns()
                    data = wire[6:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    payload = json.loads(data)
                    if "error" in payload:
                        raise RuntimeError(str(payload["error"]))
                    delta = "".join(c.get("text", "") for c in payload.get("choices", []))
                    if delta and first_text is None:
                        first_text = at
                    if payload.get("usage"):
                        usage = payload["usage"]
                    chunks.append(dict(at_ns=at, payload=payload))
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')}") from error
        ended = time.monotonic_ns()
        assert done and usage and first_text is not None, "incomplete SSE response"
        assert usage["prompt_tokens"] == len(row["input_ids"]), "prompt changed"
        assert 0 < usage["completion_tokens"] <= args.max_tokens
        text = "".join(c.get("text", "") for item in chunks
                       for c in item["payload"].get("choices", []))
        return dict(id=row["id"], index=index, key=key, text=text, usage=usage,
                    started_ns=started, ended_ns=ended, first_text_ns=first_text,
                    wall_s=(ended-started)/1e9, ttft_s=(first_text-started)/1e9,
                    chunks=chunks)

    outputs = {}
    reference = None
    if args.reference_responses:
        reference_raw = args.reference_responses.read_bytes()
        reference_rows = [json.loads(line) for line in reference_raw.splitlines()]
        reference = {row["id"]: (row["text"], row["usage"])
                     for row in reference_rows if row["mode"] == "resident"}
        assert set(reference) == {row["id"] for row in rows}, "reference request set changed"
        summary["reference_responses_sha256"] = hashlib.sha256(reference_raw).hexdigest()
    save()
    try:
        with (args.output / "responses.jsonl").open("x") as log:
            for mode in (("resident",) if serial_reference or args.resident_only else ("rotating", "resident")):
                outputs[mode] = {}
                groups = ((0,), (1,), (2,), (3,)) if serial_reference else ((0, 2), (1, 3))
                for group_index, indices in enumerate(groups):
                    turn = indices[0] % 2 if serial_reference else group_index
                    barrier = threading.Barrier(len(indices) + 1)
                    with ThreadPoolExecutor(max_workers=len(indices)) as pool:
                        futures = []
                        for session, index in enumerate(indices):
                            key = f"http-{mode}-session{session}"
                            if mode == "rotating":
                                key += f"-turn{turn}"
                            if args.serial_session_reference:
                                key = None
                            elif args.serial_controller_reference:
                                key = f"http-resident-session{index // 2}"
                            futures.append(pool.submit(request, index, key, barrier))
                        began = time.monotonic_ns()
                        barrier.wait()
                        records = [future.result() for future in futures]
                    ended = max(record["ended_ns"] for record in records)
                    count = sum(record["usage"]["completion_tokens"] for record in records)
                    wall = (ended-began)/1e9
                    group = dict(mode=mode, turn=turn, wall_s=wall,
                                 completion_tokens=count, service_tok_s=count/wall,
                                 ttft_s=[record["ttft_s"] for record in records])
                    for record in records:
                        outputs[mode][record["id"]] = (record["text"], record["usage"])
                        log.write(json.dumps(dict(mode=mode, turn=turn, **record)) + "\n")
                    log.flush()
                    summary["groups"].append(group)
                    save()
                    print(json.dumps(group), flush=True)
        if serial_reference:
            same = outputs["resident"] == reference
            summary["exact_serial_resident_reference_match"] = same
            summary["per_request_reference_match"] = {
                name: value == reference[name] for name, value in outputs["resident"].items()}
            assert same, "serial reference and C2 resident outputs differ; inspect saved responses"
        elif not args.resident_only:
            same = outputs["rotating"] == outputs["resident"]
            summary["exact_text_and_usage_match"] = same
            assert same, "resident history reuse changed text or usage; inspect saved responses"
        summary["status"] = "client_checks_passed_server_evidence_pending"
        save()
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
