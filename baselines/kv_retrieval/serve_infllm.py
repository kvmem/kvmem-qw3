"""Text-only Claude Messages endpoint backed by the pinned original InfLLM.

One serialized model session; exact native prompt/token/tool frontend. Timings
from --shared-gpu runs are explicitly unsuitable for performance comparisons.
"""
import argparse
import hashlib
import importlib.metadata
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import subprocess
import time
import traceback
import uuid
from urllib.parse import urlsplit

import torch

from kv_baselines.frontend import NativeBridge, validate_text_request, stream_events, meaningful_content, action_content
from kv_baselines.generation import ThinkingBudget, sample
from kv_baselines.infllm_adapter import InfLLMConfig
from kv_baselines.infllm_hybrid import install
from kv_baselines.session import InfLLMSession
from kv_baselines.memory import session_memory
from kv_baselines.upstream import verify_source
from run_model_smoke import load_text_model, save


class Application:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.trace_lock = threading.Lock()
        self.reset_reason = None
        self.check_gpu_ownership()
        self.bridge = NativeBridge(args.model)
        free, total = torch.cuda.mem_get_info()
        if free < (args.max_gpu_gib + 2) * 2**30:
            raise RuntimeError(f"insufficient free GPU memory: {free / 2**30:.1f} GiB")
        torch.cuda.set_per_process_memory_fraction(args.max_gpu_gib * 2**30 / total)
        self.model, model_manifest = load_text_model(args.model)
        install(self.model)
        self.config = InfLLMConfig(**json.loads(args.infllm_config.read_text())) if args.infllm_config else InfLLMConfig()
        self.session = InfLLMSession(self.model, self.config, prefill_chunk=args.prefill_chunk)
        eos = self.bridge.call("tokenize", text="<|im_end|>")["input_ids"]
        if len(eos) != 1:
            raise ValueError("native end tag must be one token")
        self.eos_id = eos[0]
        self.generator = torch.Generator(device="cuda").manual_seed(args.seed)
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
        self.manifest = {"status": "ready", "method": "InfLLM (adapted to Qwen3.8)",
            "model": model_manifest, "InfLLM": verify_source("InfLLM"), "config": asdict(self.config),
            "shared_gpu": args.shared_gpu, "performance_eligible": not args.shared_gpu,
            "max_gpu_gib": args.max_gpu_gib, "max_context": args.max_context,
            "seed": args.seed, "sampler_rng": "torch.cuda.Generator; differs from native mt19937_64",
            "recurrence_backend": modeling.chunk_gated_delta_rule.__module__ if modeling.chunk_gated_delta_rule else "torch_fallback",
            "prefill_chunk": args.prefill_chunk, "checkpoint_capacity": 3,
            "sampling_defaults": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k},
            "thinking_budget_default": args.thinking_budget, "max_tokens": args.max_tokens,
            "pid": os.getpid(), "started": time.time()}
        root = Path(__file__).resolve().parent
        sources = [Path(__file__), root / "native_bridge.cpp", root / "run_model_smoke.py",
            root / "native_build/baseline_native_bridge", *sorted((root / "kv_baselines").glob("*.py"))]
        self.manifest["source_sha256"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        self.manifest["package_versions"] = {name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "triton", "fla-core", "flash-linear-attention")}
        self.manifest["causal_convolution_backend"] = "torch fallback" if modeling.causal_conv1d_fn is None else modeling.causal_conv1d_fn.__module__
        if self.manifest["recurrence_backend"] == "torch_fallback":
            raise RuntimeError("optimized recurrence is required for the HTTP baseline")
        save(args.out / "manifest.json", self.manifest)

    def check_gpu_ownership(self):
        if self.args.shared_gpu:
            return
        pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
            "--format=csv,noheader,nounits"], text=True).split()
        other = [int(pid) for pid in pids if int(pid) != os.getpid()]
        if other:
            raise RuntimeError(f"formal measurements require exclusive GPU; other PIDs: {other}")

    def record(self, row):
        with self.trace_lock, (self.args.out / "requests.jsonl").open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @torch.inference_mode()
    def generate(self, request, emit, received_at, clock_start, request_id):
        trace = {"request_id": request_id, "received_at": received_at,
            "shared_gpu": self.args.shared_gpu, "performance_eligible": not self.args.shared_gpu,
            "status": "running", "api_first_content_s": None, "api_first_action_s": None,
            "first_model_token_s": None}
        generated = []
        state_touched = False
        try:
            self.check_gpu_ownership()
            validate_text_request(request)
            rendering_started = time.perf_counter()
            rendered = self.bridge.call("render", request=request)
            generation = rendered["openai_request"]
            prompt = rendered["input_ids"]
            maximum = min(int(generation["max_tokens"]), self.args.max_tokens)
            if maximum < 0 or len(prompt) + maximum > self.args.max_context:
                raise ValueError("prompt plus output limit exceeds configured context")
            enabled = generation.get("enable_thinking", True)
            trace.update(prompt_tokens=len(prompt), render_s=time.perf_counter() - rendering_started,
                         queue_s=rendering_started - clock_start, max_tokens=maximum)
            controls = {"temperature": generation.get("temperature", self.args.temperature),
                "top_p": generation.get("top_p", self.args.top_p),
                "top_k": generation.get("top_k", self.args.top_k),
                "min_p": generation.get("min_p", 0),
                "presence_penalty": generation.get("presence_penalty", 0),
                "repetition_penalty": generation.get("repetition_penalty", 1)}
            if controls["repetition_penalty"] <= 0 or not 0 <= controls["top_p"] <= 1:
                raise ValueError("invalid sampling controls")
            trace["sampling"] = controls
            thinking = ThinkingBudget(self.bridge, enabled,
                int(generation.get("thinking_budget", self.args.thinking_budget)),
                generation.get("recover_thinking_eos", enabled), self.eos_id)
            self.bridge.call("stream_start", input_tokens=len(prompt), enable_thinking=enabled,
                tools=generation.get("tools", []), model=request.get("model", "baseline"),
                message_id="chatcmpl-" + request_id)
            def output(wire):
                if not wire:
                    return
                events = list(stream_events(wire))
                if trace["api_first_content_s"] is None and any(meaningful_content(x) for x in events):
                    trace["api_first_content_s"] = time.perf_counter() - clock_start
                if trace["api_first_action_s"] is None and any(action_content(x) for x in events):
                    trace["api_first_action_s"] = time.perf_counter() - clock_start
                emit(wire)
            prefill_start = time.perf_counter()
            state_touched = True
            logits, reuse = self.session.begin(prompt)
            torch.cuda.synchronize()
            trace.update(reuse, prefill_s=time.perf_counter() - prefill_start)
            if self.reset_reason:
                trace.update(reuse_reason=self.reset_reason, unexpected_cold_prefill=True)
                self.reset_reason = None
            decode_start = time.perf_counter()
            seen = set(prompt)
            finish = "length"
            frontend_s = 0.0
            for _ in range(maximum):
                token = thinking.apply(sample(logits, self.generator, seen=seen, **controls))
                if trace["first_model_token_s"] is None:
                    trace["first_model_token_s"] = time.perf_counter() - clock_start
                if token == self.eos_id:
                    finish = "stop"
                    break
                generated.append(token)
                seen.add(token)
                front_start = time.perf_counter()
                chunk = self.bridge.call("stream_token", token_id=token)
                output(chunk["wire"])
                frontend_s += time.perf_counter() - front_start
                if len(generated) < maximum:
                    logits = self.session.consume_generated_token(token)
            final = self.bridge.call("stream_finish", reason=finish)
            finish = final["finish_reason"]
            output(final["wire"])
            trace.update(decode_s=time.perf_counter() - decode_start,
                output_tokens=len(generated), output_token_ids=generated,
                completion=final["raw_completion"], finish_reason=finish,
                forced_thinking_tokens=thinking.forced_tokens,
                token_frontend_s=frontend_s, status="completed")
            managers = [layer.manager for layer in self.session.cache.layers if hasattr(layer, "manager")]
            trace["global_blocks_per_layer"] = [m.num_global_block for m in managers]
            trace["cached_blocks_per_layer"] = [sum(map(len, m.cached_blocks)) for m in managers]
            trace["lru_clock_per_layer"] = [m.load_count for m in managers]
            trace["torch_allocated_gib"] = torch.cuda.memory_allocated() / 2**30
            trace["torch_reserved_gib"] = torch.cuda.memory_reserved() / 2**30
            trace["process_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            if not request.get("stream", False):
                parsed = self.bridge.call("parse", text=final["raw_completion"],
                    tools=generation.get("tools", []))
                trace["response"] = self.bridge.call("anthropic_response", model=request.get("model", "baseline"),
                    response={"id": "chatcmpl-" + request_id, "choices": [{"message": {"role": "assistant", **parsed},
                        "finish_reason": finish}], "usage": {
                        "prompt_tokens": len(prompt), "completion_tokens": len(generated)}})
            return trace
        except BaseException as error:
            trace.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                         output_tokens=len(generated))
            # An interrupted forward may leave hybrid state partially updated.
            # Never reuse it after cancellation/errors; report next cold start.
            if state_touched:
                self.session = InfLLMSession(self.model, self.config, prefill_chunk=self.args.prefill_chunk)
                self.reset_reason = "previous_request_error_state_discarded"
            raise
        finally:
            trace["request_wall_s"] = time.perf_counter() - clock_start
            if self.args.memory_inventory:
                # Expensive CPU traversal is calibration-only and is explicitly
                # outside the request timing. It must be off for agent timing.
                trace["cache_storage_inventory"] = session_memory(self.session)
            self.record(trace)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def json_reply(self, code, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            app = getattr(self.server, "app", None)
            try:
                self.json_reply(200 if app else 503,
                    {"status": "ok", "method": app.manifest["method"]} if app else {"status": "loading"})
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
        else:
            self.json_reply(404, {"error": "unknown endpoint"})

    def do_POST(self):
        started, received_at = time.perf_counter(), time.time()
        streaming = False
        request_id = uuid.uuid4().hex
        route = urlsplit(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 128 * 1024 * 1024:
                raise ValueError("invalid or excessive content length")
            request = json.loads(self.rfile.read(length))
            validate_text_request(request)
            app = getattr(self.server, "app", None)
            if app is None:
                self.json_reply(503, {"error": "model is loading"})
                return
            if route == "/v1/messages/count_tokens":
                rendered = app.bridge.call("render", request=request, require_max_tokens=False)
                self.json_reply(200, {"input_tokens": len(rendered["input_ids"])})
                return
            if route != "/v1/messages":
                self.json_reply(404, {"error": "unknown endpoint"})
                return
            (app.args.out / "inputs" / f"{request_id}.json").write_text(json.dumps(request, ensure_ascii=False) + "\n")
            with app.lock:
                if request.get("stream", False):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    streaming = True
                    def emit(wire):
                        self.wfile.write(wire.encode())
                        self.wfile.flush()
                else:
                    emit = lambda wire: None
                result = app.generate(request, emit, received_at, started, request_id)
                if not streaming:
                    self.json_reply(200, result["response"])
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as error:
            traceback.print_exc()
            body = {"type": "error", "error": {"type": "invalid_request_error" if isinstance(error, ValueError) else "api_error", "message": str(error)}}
            if streaming:
                self.wfile.write(("event: error\ndata: " + json.dumps(body) + "\n\n").encode())
                self.wfile.flush()
            else:
                self.json_reply(400 if isinstance(error, ValueError) else 500, body)

    def do_HEAD(self):
        self.send_response(200 if getattr(self.server, "app", None) else 503)
        self.send_header("Content-Length", "0")
        self.end_headers()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18223)
    p.add_argument("--max-gpu-gib", type=float, default=55)
    p.add_argument("--shared-gpu", action="store_true")
    p.add_argument("--max-context", type=int, default=1048576)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--thinking-budget", type=int, default=28672)
    p.add_argument("--temperature", type=float, default=1)
    p.add_argument("--top-p", type=float, default=.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--prefill-chunk", type=int, default=512)
    p.add_argument("--infllm-config", type=Path)
    p.add_argument("--memory-inventory", action="store_true")
    args = p.parse_args()
    args.model = args.model.resolve()
    (args.out / "inputs").mkdir(parents=True, exist_ok=True)
    # Bind before allocating 50 GiB of model weights.
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        server.app = Application(args)
        print(json.dumps({"ready": True, "port": args.port}), flush=True)
        serving.join()
    finally:
        server.shutdown()
        server.server_close()
        if hasattr(server, "app"):
            server.app.bridge.close()


if __name__ == "__main__":
    main()
