"""Shared subprocess helpers for qw3 benchmark and profiling scripts."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union


PathLike = Union[str, Path]


def _cwd_arg(cwd: Optional[PathLike]) -> Optional[str]:
    return str(cwd) if cwd is not None else None


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_captured(
    cmd: Sequence[str],
    *,
    timeout: float,
    env: Optional[Mapping[str, str]] = None,
    stdin=None,
    cwd: Optional[PathLike] = None,
) -> subprocess.CompletedProcess:
    """Run a child process and kill its whole process group on timeout."""
    proc = subprocess.Popen(
        list(cmd),
        cwd=_cwd_arg(cwd),
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        proc.communicate()
        raise
    return subprocess.CompletedProcess(list(cmd), proc.returncode, out, err)


def run_logged(
    cmd: Sequence[str],
    *,
    cwd: PathLike,
    env: Optional[Mapping[str, str]] = None,
    log_path: Optional[PathLike] = None,
    timeout: Optional[float] = None,
    status_prefix: str = "[auto]",
) -> int:
    """Run a command, stream captured output, and optionally persist a log."""
    cmd_list = list(cmd)
    print("+ " + " ".join(cmd_list), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd_list,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(proc)
        out, _ = proc.communicate()
    elapsed = time.time() - t0
    out = out or ""
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    code = 124 if timed_out else proc.returncode
    if timed_out:
        print(
            f"{status_prefix} timeout after {timeout:.1f}s; killed process group",
            flush=True,
        )
    print(f"{status_prefix} exit={code} elapsed={elapsed:.1f}s", flush=True)
    if log_path:
        Path(log_path).write_text(out)
    return code
