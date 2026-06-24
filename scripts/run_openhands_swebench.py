#!/usr/bin/env python3
"""Run headless OpenHands SWE-bench rollout and official SWE-bench scoring.

This wrapper assumes the model server already exposes an OpenAI-compatible API
at --base-url (qw3 serve defaults to http://127.0.0.1:8080/v1). It delegates the
agent rollout to OpenHands/benchmarks, then delegates patch scoring to the
official SWE-bench harness through OpenHands' swebench-eval entry point.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENHANDS_DIR = REPO_ROOT / "third_party" / "openhands-benchmarks"
DEFAULT_OPENHANDS_REPO = "git@github.com:OpenHands/benchmarks.git"
OPENHANDS_WORKSPACE_MEMBERS = [
    "vendor/software-agent-sdk/openhands-sdk",
    "vendor/software-agent-sdk/openhands-tools",
    "vendor/software-agent-sdk/openhands-workspace",
    "vendor/software-agent-sdk/openhands-agent-server",
]


def git_sha_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "nogit"


def default_run_id(memory_policy: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"openhands_qw3_{memory_policy}_{git_sha_short()}_{stamp}"


def ensure_openhands_dir(
    path: Path, clone: bool, clone_retries: int, repo_url: str
) -> None:
    submodule_cmd = [
        "git",
        "-C",
        str(path),
        "-c",
        "url.git@github.com:.insteadOf=https://github.com/",
        "-c",
        "url.git@github.com:.insteadOf=https://github.com:",
        "submodule",
        "update",
        "--init",
        "--recursive",
        "--depth",
        "1",
    ]
    if path.exists():
        if not (path / "pyproject.toml").is_file():
            raise SystemExit(
                f"{path} exists but does not look like OpenHands/benchmarks "
                "(missing pyproject.toml)"
            )
        missing_members = [
            member
            for member in OPENHANDS_WORKSPACE_MEMBERS
            if not (path / member / "pyproject.toml").is_file()
        ]
        if missing_members:
            print(
                "[clone] repairing missing OpenHands submodules: "
                + ", ".join(missing_members),
                flush=True,
            )
            print("+ " + " ".join(submodule_cmd), flush=True)
            subprocess.run(submodule_cmd, check=True)
        return
    if not clone:
        raise SystemExit(
            f"OpenHands benchmarks directory not found: {path}\n"
            "Clone it first or pass --clone-openhands."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    clone_cmd = ["git", "clone", "--depth", "1", repo_url, str(path)]
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, clone_retries + 1):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        print(f"[clone] attempt {attempt}/{clone_retries}", flush=True)
        print("+ " + " ".join(clone_cmd), flush=True)
        try:
            subprocess.run(clone_cmd, check=True)
            print("+ " + " ".join(submodule_cmd), flush=True)
            subprocess.run(submodule_cmd, check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if path.exists() and not (path / "pyproject.toml").is_file():
                shutil.rmtree(path, ignore_errors=True)
            if attempt < clone_retries:
                time.sleep(min(30, 5 * attempt))
    assert last_error is not None
    raise last_error


def check_qw3_health(base_url: str, timeout: float) -> None:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    health_url = root + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:
            if response.status >= 400:
                raise SystemExit(f"qw3 health check failed: HTTP {response.status}")
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"qw3 health check failed for {health_url}: {exc}\n"
            "Start qw3 serve first, or pass --skip-health-check."
        ) from exc


def write_llm_config(path: Path, args: argparse.Namespace) -> None:
    cfg: dict[str, Any] = {
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "api_key": args.api_key,
        "temperature": args.temperature,
    }
    if args.max_output_tokens:
        cfg["max_output_tokens"] = args.max_output_tokens
    if args.custom_tokenizer:
        cfg["custom_tokenizer"] = args.custom_tokenizer
    if args.disable_native_tool_calling:
        cfg["native_tool_calling"] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def strip_dockerfile_syntax(openhands_dir: Path) -> None:
    dockerfiles = [
        openhands_dir
        / "vendor"
        / "software-agent-sdk"
        / "openhands-agent-server"
        / "openhands"
        / "agent_server"
        / "docker"
        / "Dockerfile"
    ]
    for dockerfile in dockerfiles:
        if not dockerfile.is_file():
            continue
        text = dockerfile.read_text(encoding="utf-8")
        lines = text.splitlines()
        if lines and lines[0].startswith("# syntax=docker/dockerfile:"):
            dockerfile.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
            print(f"[docker] stripped BuildKit syntax directive from {dockerfile}")


def run_logged(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+ " + " ".join(cmd), flush=True)
    json_lines: list[dict[str, Any]] = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write("+ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    json_lines.append(parsed)
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)
    return json_lines


def newest_output_jsonl(output_dir: Path) -> Path:
    candidates = sorted(
        output_dir.rglob("output.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"Could not find output.jsonl under {output_dir}")
    return candidates[0]


def count_jsonl_entries(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def copy_if_present(src: Path | None, dst_dir: Path) -> Path | None:
    if src is None:
        return None
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def build_infer_cmd(
    args: argparse.Namespace,
    llm_config: Path,
    openhands_output_dir: Path,
    select_file: Path | None,
) -> list[str]:
    cmd = [
        args.uv_bin,
        "run",
        "swebench-infer",
        str(llm_config),
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--workspace",
        args.workspace,
        "--max-iterations",
        str(args.max_iterations),
        "--num-workers",
        str(args.num_workers),
        "--output-dir",
        str(openhands_output_dir),
        "--n-critic-runs",
        str(args.n_critic_runs),
        "--max-retries",
        str(args.max_retries),
        "--tool-preset",
        args.tool_preset,
        "--note",
        args.note,
    ]
    if args.n_limit:
        cmd.extend(["--n-limit", str(args.n_limit)])
    if select_file is not None:
        cmd.extend(["--select", str(select_file)])
    if args.enable_delegation:
        cmd.append("--enable-delegation")
    if args.memory_policy == "condenser":
        cmd.append("--enable-condenser")
        if args.condenser_max_size is not None:
            cmd.extend(["--condenser-max-size", str(args.condenser_max_size)])
        if args.condenser_max_tokens is not None:
            cmd.extend(["--condenser-max-tokens", str(args.condenser_max_tokens)])
        if args.condenser_max_output_tokens is not None:
            cmd.extend(
                ["--condenser-max-output-tokens", str(args.condenser_max_output_tokens)]
            )
        if args.condenser_keep_first is not None:
            cmd.extend(["--condenser-keep-first", str(args.condenser_keep_first)])
    else:
        cmd.append("--disable-condenser")
    return cmd


def build_eval_cmd(
    args: argparse.Namespace,
    output_json: Path,
    predictions_path: Path,
) -> list[str]:
    cmd = [
        args.uv_bin,
        "run",
        "swebench-eval",
        str(output_json),
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--run-id",
        args.run_id,
        "--workers",
        str(args.eval_workers),
        "--timeout",
        str(args.eval_timeout),
        "--output-file",
        str(predictions_path),
    ]
    if args.skip_evaluation:
        cmd.append("--skip-evaluation")
    if args.modal:
        cmd.append("--modal")
    else:
        cmd.append("--no-modal")
    if args.apptainer_eval:
        cmd.append("--apptainer")
        if args.apptainer_sandbox_root:
            cmd.extend(["--apptainer-sandbox-root", args.apptainer_sandbox_root])
        if args.apptainer_cache:
            cmd.extend(["--apptainer-cache", args.apptainer_cache])
    return cmd


def build_images_cmd(args: argparse.Namespace, select_file: Path | None) -> list[str]:
    cmd = [
        args.uv_bin,
        "run",
        "python",
        "-m",
        "benchmarks.swebench.build_images",
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--image",
        args.agent_server_image,
        "--target",
        args.image_target,
        "--max-workers",
        str(args.image_workers),
    ]
    if args.n_limit:
        cmd.extend(["--n-limit", str(args.n_limit)])
    if select_file is not None:
        cmd.extend(["--select", str(select_file)])
    return cmd


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    output_json: Path | None,
    predictions_path: Path,
    report_json: Path | None,
) -> None:
    lines = [
        "# OpenHands SWE-bench Run",
        "",
        f"- run_id: `{args.run_id}`",
        f"- dataset: `{args.dataset}`",
        f"- split: `{args.split}`",
        f"- workspace: `{args.workspace}`",
        f"- memory_policy: `{args.memory_policy}`",
        f"- model: `{args.model}`",
        f"- base_url: `{args.base_url}`",
        f"- output_json: `{output_json or ''}`",
        f"- predictions: `{predictions_path}`",
        f"- report_json: `{report_json or ''}`",
        "",
        "## Next Files",
        "",
        "- `llm_config.json` is the OpenHands LLM config generated for qw3.",
        "- `infer.log` contains the OpenHands rollout log.",
        "- `eval.log` contains conversion and official SWE-bench scoring output.",
        "- `output.jsonl` is the OpenHands trajectory/output file when rollout ran.",
        "- `predictions.swebench.jsonl` is the official SWE-bench predictions file.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenHands SWE-bench rollout against an existing qw3 server."
    )
    parser.add_argument(
        "--openhands-dir",
        type=Path,
        default=Path(os.getenv("OPENHANDS_BENCHMARKS_DIR", DEFAULT_OPENHANDS_DIR)),
        help="Path to a clone of OpenHands/benchmarks.",
    )
    parser.add_argument(
        "--clone-openhands",
        action="store_true",
        help="Clone OpenHands benchmarks into --openhands-dir if it is missing.",
    )
    parser.add_argument(
        "--openhands-repo",
        default=DEFAULT_OPENHANDS_REPO,
        help="Git URL for OpenHands/benchmarks. Defaults to SSH.",
    )
    parser.add_argument(
        "--clone-retries",
        type=int,
        default=3,
        help="Number of git clone attempts when --clone-openhands is used.",
    )
    parser.add_argument("--uv-bin", default="uv", help="uv executable to use.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="openai/qw3")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--custom-tokenizer")
    parser.add_argument(
        "--disable-native-tool-calling",
        action="store_true",
        help="Set native_tool_calling=false in the OpenHands LLM config.",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Do not check http://host:port/health before rollout.",
    )
    parser.add_argument("--health-timeout", type=float, default=10.0)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--workspace",
        default="docker",
        choices=["docker", "remote", "apptainer"],
        help="OpenHands workspace type for SWE-bench rollout.",
    )
    parser.add_argument("--n-limit", type=int, default=0)
    parser.add_argument(
        "--select",
        type=Path,
        help="Optional text file with one SWE-bench instance_id per line.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--n-critic-runs", type=int, default=1)
    parser.add_argument(
        "--tool-preset",
        default="default",
        choices=["default", "gemini", "gpt5", "planning"],
    )
    parser.add_argument("--enable-delegation", action="store_true")
    parser.add_argument(
        "--memory-policy",
        default="condenser",
        choices=["condenser", "no_condenser"],
        help="OpenHands memory policy for this run.",
    )
    parser.add_argument("--condenser-max-size", type=int)
    parser.add_argument("--condenser-max-tokens", type=int)
    parser.add_argument("--condenser-max-output-tokens", type=int)
    parser.add_argument("--condenser-keep-first", type=int)
    parser.add_argument(
        "--run-id",
        help="SWE-bench run id. Defaults to a timestamped OpenHands/qw3 id.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=REPO_ROOT / "benchmark" / "results",
    )
    parser.add_argument(
        "--existing-output-json",
        type=Path,
        help="Skip rollout and evaluate an existing OpenHands output.jsonl.",
    )
    parser.add_argument(
        "--skip-infer",
        action="store_true",
        help="Skip OpenHands rollout. Requires --existing-output-json.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only convert OpenHands output to SWE-bench predictions.",
    )
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument("--modal", action="store_true")
    parser.add_argument("--apptainer-eval", action="store_true")
    parser.add_argument("--apptainer-sandbox-root")
    parser.add_argument("--apptainer-cache")
    parser.add_argument(
        "--build-images",
        action="store_true",
        help="Build OpenHands SWE-bench agent-server Docker images before rollout.",
    )
    parser.add_argument(
        "--build-retries",
        type=int,
        default=3,
        help="Number of attempts for OpenHands build_images.",
    )
    parser.add_argument(
        "--agent-server-image",
        default="ghcr.io/openhands/eval-agent-server",
    )
    parser.add_argument("--image-target", default="source-minimal")
    parser.add_argument("--image-workers", type=int, default=4)
    parser.add_argument(
        "--image-tag-prefix",
        help=(
            "Override OpenHands IMAGE_TAG_PREFIX so build_images and "
            "swebench-infer agree on local image tags. For local builds, use "
            "the SDK short SHA, e.g. 43376f1."
        ),
    )
    parser.add_argument(
        "--swebench-image-template",
        help=(
            "Override SWE-bench base images, passed as "
            "OPENHANDS_SWEBENCH_IMAGE_TEMPLATE. Template variables include "
            "{instance_id}, {repo}, {name}, and {arch}."
        ),
    )
    parser.add_argument(
        "--strip-dockerfile-syntax",
        action="store_true",
        help=(
            "Remove OpenHands agent-server Dockerfile '# syntax=' directive to "
            "avoid pulling docker/dockerfile from DockerHub."
        ),
    )
    parser.add_argument(
        "--note",
        default="qw3-openhands-swebench",
        help="OpenHands eval note included in its output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_id is None:
        args.run_id = default_run_id(args.memory_policy)
    if args.skip_infer and args.existing_output_json is None:
        raise SystemExit("--skip-infer requires --existing-output-json")

    ensure_openhands_dir(
        args.openhands_dir,
        args.clone_openhands,
        args.clone_retries,
        args.openhands_repo,
    )
    if args.strip_dockerfile_syntax:
        strip_dockerfile_syntax(args.openhands_dir)
    if not args.skip_health_check and not args.skip_infer:
        check_qw3_health(args.base_url, args.health_timeout)

    run_root = (
        args.result_root
        / f"{git_sha_short()}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        / "openhands_swebench"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    llm_config = run_root / "llm_config.json"
    write_llm_config(llm_config, args)
    select_file = copy_if_present(args.select, run_root)
    openhands_output_dir = run_root / "openhands_outputs"
    predictions_path = run_root / "predictions.swebench.jsonl"

    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", args.api_key)
    # Force git dependencies pulled by uv/pip (for example git+https GitHub
    # packages in OpenHands' pyproject) through SSH as well.
    env.setdefault("GIT_CONFIG_COUNT", "2")
    env.setdefault("GIT_CONFIG_KEY_0", "url.git@github.com:.insteadOf")
    env.setdefault("GIT_CONFIG_VALUE_0", "https://github.com/")
    env.setdefault("GIT_CONFIG_KEY_1", "url.git@github.com:.insteadOf")
    env.setdefault("GIT_CONFIG_VALUE_1", "https://github.com:")
    if args.swebench_image_template:
        env["OPENHANDS_SWEBENCH_IMAGE_TEMPLATE"] = args.swebench_image_template
    if args.image_tag_prefix:
        env["IMAGE_TAG_PREFIX"] = args.image_tag_prefix

    if args.build_images:
        last_exc: subprocess.CalledProcessError | None = None
        for attempt in range(1, args.build_retries + 1):
            try:
                run_logged(
                    build_images_cmd(args, select_file),
                    cwd=args.openhands_dir,
                    log_path=run_root / f"build_images_attempt{attempt}.log",
                    env=env,
                )
                last_exc = None
                break
            except subprocess.CalledProcessError as exc:
                last_exc = exc
                if attempt < args.build_retries:
                    time.sleep(min(60, 10 * attempt))
        if last_exc is not None:
            raise last_exc

    output_json: Path | None
    if args.skip_infer:
        output_json = args.existing_output_json.resolve()
    else:
        infer_json = run_logged(
            build_infer_cmd(args, llm_config, openhands_output_dir, select_file),
            cwd=args.openhands_dir,
            log_path=run_root / "infer.log",
            env=env,
        )
        output_paths = [item.get("output_json") for item in infer_json if "output_json" in item]
        if output_paths:
            output_json = Path(str(output_paths[-1]))
            if not output_json.is_absolute():
                output_json = (args.openhands_dir / output_json).resolve()
        else:
            output_json = newest_output_jsonl(openhands_output_dir).resolve()
        shutil.copy2(output_json, run_root / "output.jsonl")

    output_count = count_jsonl_entries(output_json)
    if output_count == 0:
        raise SystemExit(
            f"OpenHands rollout produced no results: {output_json}\n"
            f"See inference logs under {openhands_output_dir}."
        )

    report_json: Path | None = None
    eval_json = run_logged(
        build_eval_cmd(args, output_json, predictions_path),
        cwd=args.openhands_dir,
        log_path=run_root / "eval.log",
        env=env,
    )
    report_paths = [item.get("report_json") for item in eval_json if "report_json" in item]
    if report_paths and report_paths[-1]:
        report_json = Path(str(report_paths[-1])).resolve()
        if report_json.exists() and report_json.parent != run_root:
            shutil.copy2(report_json, run_root / report_json.name)

    write_summary(
        run_root / "summary_openhands_swebench.md",
        args=args,
        output_json=output_json,
        predictions_path=predictions_path,
        report_json=report_json,
    )
    print(json.dumps({"result_dir": str(run_root), "output_json": str(output_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
