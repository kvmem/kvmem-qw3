#!/usr/bin/env python3
"""Write a non-secret, reproducible manifest for one OSWorld-V2 run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import hashlib
import os
import platform
import subprocess
from pathlib import Path


def command_output(*args: str) -> str | None:
    try:
        return subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def file_metadata(path: Path) -> dict:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qw3-root", type=Path, required=True)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument("--test-meta", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vision-model", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-profile", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--prefix-cache", type=int, choices=(0, 1), required=True)
    parser.add_argument("--prefix-cache-guard-pages", type=int, required=True)
    parser.add_argument("--continuous-batching", type=int, choices=(0, 1), required=True)
    parser.add_argument("--max-active", type=int, required=True)
    parser.add_argument("--kvmem-prefix-cache", type=int, choices=(0, 1), required=True)
    parser.add_argument("--kvmem-block-tokens", type=int, required=True)
    parser.add_argument("--kvmem-budget", type=int, required=True)
    parser.add_argument("--kvmem-gen-budget", type=int, required=True)
    parser.add_argument("--kvmem-cpu-gb", type=float, required=True)
    parser.add_argument("--kvmem-immutable-k", type=int, choices=(0, 1), required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    task_meta = json.loads(args.test_meta.read_text(encoding="utf-8"))
    selected_tasks = [item for values in task_meta.values() for item in values]
    vision_index = args.vision_model / "model.safetensors.index.json"
    manifest = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "run_label": args.run_label,
        "run_id": args.run_id,
        "evaluation_profile": args.evaluation_profile,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu": command_output(
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ),
        },
        "source": {
            "qw3_commit": command_output("git", "-C", str(args.qw3_root), "rev-parse", "HEAD"),
            "osworld_commit": command_output(
                "git", "-C", str(args.osworld_root), "rev-parse", "HEAD"
            ),
            "osworld_tag": command_output(
                "git", "-C", str(args.osworld_root), "describe", "--tags", "--exact-match"
            ),
            "worktree_status": file_metadata(args.output.parent / "source_status.txt"),
            "dirty_patch": file_metadata(args.output.parent / "source_dirty.patch"),
            "osworld_dirty_patch": file_metadata(args.output.parent / "osworld_source_dirty.patch"),
            "osworld_untracked_snapshot": file_metadata(
                args.output.parent / "osworld_source_untracked_snapshot.tar.gz"
            ),
            "untracked_snapshot": file_metadata(
                args.output.parent / "source_untracked_snapshot.tar.gz"
            ),
        },
        "inputs": {
            "test_meta": file_metadata(args.test_meta),
            "selected_tasks": selected_tasks,
            "task_count": len(selected_tasks),
            "model": file_metadata(args.model),
            "vision_model_index": file_metadata(vision_index),
            "qw3_binary": {
                **file_metadata(Path(os.environ.get("QW3_OSWORLD_BINARY", str(args.qw3_root / "build" / "qw3")))),
                "sha256": hashlib.sha256(Path(os.environ.get("QW3_OSWORLD_BINARY", str(args.qw3_root / "build" / "qw3"))).read_bytes()).hexdigest(),
            },
        },
        "run": {
            "provider": "docker",
            "observation_type": "screenshot",
            "action_space": "pyautogui",
            "max_steps": args.max_steps,
            "max_output_tokens": args.max_output_tokens,
            "num_envs": 1,
            "port": args.port,
            "recording_enabled": os.environ.get("QW3_OSWORLD_ENABLE_RECORDING", "0") == "1",
            "vnc_enabled": os.environ.get("QW3_OSWORLD_ENABLE_VNC", "0") == "1",
            "guest_resource_trace_enabled": os.environ.get("QW3_OSWORLD_TRACE_GUEST", "1") == "1",
            "sampling": {
                "seed": int(os.environ.get("QW3_OSWORLD_SEED", "0")),
                "temperature": float(os.environ.get("QW3_OSWORLD_TEMPERATURE", "1.0")),
                "top_p": float(os.environ.get("QW3_OSWORLD_TOP_P", "0.95")),
                "top_k": int(os.environ.get("QW3_OSWORLD_TOP_K", "20")),
                "min_p": float(os.environ.get("QW3_OSWORLD_MIN_P", "0.0")),
                "presence_penalty": float(
                    os.environ.get("QW3_OSWORLD_PRESENCE_PENALTY", "0.0")
                ),
                "repetition_penalty": float(
                    os.environ.get("QW3_OSWORLD_REPETITION_PENALTY", "1.0")
                ),
                "enable_thinking": os.environ.get(
                    "QW3_OSWORLD_ENABLE_THINKING", "1"
                ).strip().lower()
                in {"1", "true", "yes", "on"},
                "preserve_thinking": os.environ.get(
                    "QW3_OSWORLD_PRESERVE_THINKING", "1"
                ).strip().lower()
                in {"1", "true", "yes", "on"},
                "thinking_budget": int(
                    os.environ.get("QW3_OSWORLD_THINKING_BUDGET", "3072")
                ),
            },
        },
        "agent_compaction": {
            "history_mode": os.environ.get("QW3_OSWORLD_HISTORY_MODE", "sliding"),
            "image_folding_enabled": os.environ.get("QW3_OSWORLD_HISTORY_MODE", "sliding") == "sliding",
            "sliding_window_enabled": os.environ.get("QW3_OSWORLD_HISTORY_MODE", "sliding") == "sliding",
            "summary_trigger_tokens": int(os.environ.get("QW3_OSWORLD_SUMMARY_TRIGGER_TOKENS", "196608")),
            "summary_max_tokens": int(os.environ.get("QW3_OSWORLD_SUMMARY_MAX_TOKENS", "4096")),
            "summary_source": "all_previous_images_and_text" if os.environ.get("QW3_OSWORLD_HISTORY_MODE") == "summary" else None,
            "history_n": int(os.environ.get("QW3_OSWORLD_HISTORY_N", "100")),
            "image_max": int(os.environ.get("QW3_OSWORLD_IMAGE_MAX", "20")),
            "fold_size": int(os.environ.get("QW3_OSWORLD_FOLD_SIZE", "10")),
        },
        "runtime": {
            "message_prefix_transport": os.environ.get("QW3_OSWORLD_MESSAGE_PREFIX_CACHE", "0") == "1",
            "message_prefix_cache": os.environ.get("QW3_MESSAGE_PREFIX_CACHE", "0") == "1",
            "prepared_prefix_cache": os.environ.get("QW3_MESSAGE_PREFIX_CACHE", "0") == "1" and os.environ.get("QW3_PREPARED_PREFIX_CACHE", "1").lower() not in {"0", "off", "false", "none", "n", "no"},
            "server_token_cache": os.environ.get("QW3_SERVER_TOKEN_CACHE", "0").lower() in {"1", "true", "yes", "on"},
            "server_enable_thinking_default": os.environ.get(
                "QW3_OSWORLD_SERVER_ENABLE_THINKING",
                os.environ.get("QW3_OSWORLD_ENABLE_THINKING", "1"),
            ).strip().lower() in {"1", "true", "yes", "on"},
            "openai_timeout_seconds": float(os.environ.get("OSWORLD_OPENAI_TIMEOUT", "600")),
            "request_cancel_ack": os.environ.get("QW3_OSWORLD_REQUEST_CANCEL", "0") == "1",
            "route_expected": (
                "continuous_batching" if args.continuous_batching else "plain"
            ),
            "prefix_cache": bool(args.prefix_cache),
            "prefix_cache_commit_guard_pages": (
                args.prefix_cache_guard_pages if args.prefix_cache else None
            ),
            "kvmem_enabled": bool(args.kvmem_prefix_cache),
            "kvmem_prefix_cache": bool(args.kvmem_prefix_cache),
            "kvmem_block_tokens": args.kvmem_block_tokens,
            "kvmem_budget": args.kvmem_budget if args.kvmem_prefix_cache else None,
            "kvmem_gen_budget": (
                args.kvmem_gen_budget if args.kvmem_prefix_cache else None
            ),
            "kvmem_cpu_gb": args.kvmem_cpu_gb if args.kvmem_prefix_cache else None,
            "kvmem_method": os.environ.get("QW3_OSWORLD_KVMEM_METHOD", "recency") if args.kvmem_prefix_cache else None,
            "kvmem_retrieval_method": os.environ.get("QW3_OSWORLD_KVMEM_RETRIEVAL_METHOD", "mean-k") if args.kvmem_prefix_cache else None,
            "kvmem_query_conditioned": args.kvmem_prefix_cache == 1 and os.environ.get("QW3_OSWORLD_KVMEM_QUERY_CONDITIONED", "0") == "1",
            "kvmem_index_placement": os.environ.get("QW3_OSWORLD_KVMEM_INDEX_PLACEMENT", "gpu") if args.kvmem_prefix_cache else None,
            "kvmem_retrievable_visual_history": args.kvmem_prefix_cache == 1 and os.environ.get("QW3_KVMEM_RETRIEVABLE_VISUAL_HISTORY", "0").lower() in {"1", "true", "yes", "on"},
            "kvmem_update_mode": os.environ.get("QW3_OSWORLD_KVMEM_UPDATE_MODE", "interval") if args.kvmem_prefix_cache else None,
            "kvmem_guided_reselect": os.environ.get("QW3_OSWORLD_KVMEM_GUIDED_RESELECT", "off") if args.kvmem_prefix_cache else None,
            "kvmem_guided_thinking_tokens": int(os.environ.get("QW3_OSWORLD_KVMEM_GUIDED_THINKING_TOKENS", "0")),
            "kvmem_guided_query_tokens": int(os.environ.get("QW3_OSWORLD_KVMEM_GUIDED_QUERY_TOKENS", "256")),
            "kvmem_refresh_tokens": int(os.environ.get("QW3_OSWORLD_KVMEM_REFRESH_TOKENS", "28672")),
            "kvmem_nvme_gb": 0,
            "kvmem_opt_stage_out": os.environ.get("QW3_OSWORLD_KVMEM_OPT_STAGE_OUT", "on") == "on",
            "kvmem_raw_k_nvme": False,
            "kvmem_immutable_k": (
                bool(args.kvmem_immutable_k) if args.kvmem_prefix_cache else None
            ),
            "kvmem_harness_semantic_pins": os.environ.get(
                "QW3_OSWORLD_KVMEM_HARNESS_PINS", "0"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            "continuous_batching": bool(args.continuous_batching),
            "max_active": args.max_active,
            "mtp_chain": int(os.environ.get("QW3_OSWORLD_MTP_CHAIN", "0")),
            "context_tokens": args.context_tokens,
            "kv_dtype": os.environ.get("QW3_OSWORLD_KV_DTYPE", "fp16"),
            "vision_device": "cuda",
            "vision_embedding_cache_mib": int(
                os.environ.get("QW3_VISION_GPU_CACHE_MIB", "512")
            ),
            "vision_embedding_cpu_cache_mib": int(
                os.environ.get("QW3_VISION_CPU_CACHE_MIB", "0")
            ),
            "kvmem_decode_checkpoint_tokens": int(
                os.environ.get("QW3_KVMEM_DECODE_CHECKPOINT_TOKENS", "0")
            ),
        },
        "endpoints": {
            "website_host_suffix": os.environ.get("WEBSITE_HOST_SUFFIX"),
            "hf_endpoint": os.environ.get("HF_ENDPOINT"),
            "openai_base_url": os.environ.get("OPENAI_BASE_URL"),
        },
        "proxy_environment_present": {
            name: bool(os.environ.get(name))
            for name in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
