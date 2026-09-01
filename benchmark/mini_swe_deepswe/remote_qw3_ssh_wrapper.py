#!/usr/bin/env python3
"""Run QW3 on a remote GPU while exposing it on the local Docker bridge.

``run_requestplan10.py`` normally launches a local QW3 binary.  This file can
be supplied through ``QW3_DEEPSWE_BINARY`` instead.  It preserves the runner's
per-task server lifecycle, but replaces the local server with an SSH local
forward and a foreground QW3 process on the remote host.  Pier and every task
container continue to run locally.

The wrapper intentionally contains no credentials.  Authentication is handled
by the user's SSH configuration/key agent.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteConfig:
    host: str
    ssh_port: int
    binary: str
    workdir: str
    remote_listen_host: str
    remote_listen_port: int


def config_from_environment() -> RemoteConfig:
    host = os.environ.get("QW3_REMOTE_SSH_HOST", "").strip()
    if not host:
        raise RuntimeError("QW3_REMOTE_SSH_HOST is required")
    binary = os.environ.get("QW3_REMOTE_BINARY", "").strip()
    if not binary:
        raise RuntimeError("QW3_REMOTE_BINARY is required")
    return RemoteConfig(
        host=host,
        ssh_port=int(os.environ.get("QW3_REMOTE_SSH_PORT", "22")),
        binary=binary,
        workdir=os.environ.get("QW3_REMOTE_WORKDIR", "/home/chaidi/qw3"),
        remote_listen_host=os.environ.get(
            "QW3_REMOTE_LISTEN_HOST", "127.0.0.1"
        ),
        remote_listen_port=int(os.environ.get("QW3_REMOTE_LISTEN_PORT", "8000")),
    )


def replace_option(arguments: list[str], option: str, value: str) -> list[str]:
    result = list(arguments)
    try:
        index = result.index(option)
    except ValueError:
        result.extend([option, value])
        return result
    if index + 1 >= len(result):
        raise RuntimeError(f"{option} is missing its value")
    result[index + 1] = value
    return result


def forwarded_qw3_environment() -> dict[str, str]:
    """Forward only QW3 knobs; never leak arbitrary local credentials."""

    return {
        key: value
        for key, value in os.environ.items()
        if key.startswith("QW3_")
        and not key.startswith("QW3_REMOTE_")
        and key
        not in {
            "QW3_DEEPSWE_BINARY",
            "QW3_DEEPSWE_REMOTE_BINARY_SHA256",
            "QW3_REMOTE_WRAPPER_PRINT_COMMAND",
        }
    }


def build_ssh_command(
    arguments: list[str], config: RemoteConfig
) -> tuple[list[str], str, int]:
    if not arguments or arguments[0] != "serve":
        raise RuntimeError("remote wrapper currently supports only `qw3 serve`")

    local_host = "127.0.0.1"
    local_port = 8000
    if "--host" in arguments:
        host_index = arguments.index("--host")
        if host_index + 1 >= len(arguments):
            raise RuntimeError("--host is missing its value")
        local_host = arguments[host_index + 1]
    if "--port" in arguments:
        port_index = arguments.index("--port")
        if port_index + 1 >= len(arguments):
            raise RuntimeError("--port is missing its value")
        local_port = int(arguments[port_index + 1])

    remote_arguments = replace_option(
        arguments, "--host", config.remote_listen_host
    )
    remote_arguments = replace_option(
        remote_arguments, "--port", str(config.remote_listen_port)
    )
    remote_command = [config.binary, *remote_arguments]
    environment = forwarded_qw3_environment()
    environment_assignments = [
        f"{key}={value}" for key, value in sorted(environment.items())
    ]
    shell_command = (
        f"cd {shlex.quote(config.workdir)} && exec "
        + shlex.join(["env", *environment_assignments, *remote_command])
    )
    forward = (
        f"{local_host}:{local_port}:"
        f"{config.remote_listen_host}:{config.remote_listen_port}"
    )
    ssh_command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-L",
        forward,
        "-p",
        str(config.ssh_port),
        config.host,
        shell_command,
    ]
    return ssh_command, local_host, local_port


def main() -> int:
    config = config_from_environment()
    command, local_host, local_port = build_ssh_command(sys.argv[1:], config)
    if os.environ.get("QW3_REMOTE_WRAPPER_PRINT_COMMAND") == "1":
        print(
            json.dumps(
                {
                    "local_endpoint": f"{local_host}:{local_port}",
                    "remote_host": config.host,
                    "remote_binary": config.binary,
                    "ssh_command": command,
                },
                indent=2,
            )
        )
        return 0
    os.execvp(command[0], command)
    raise AssertionError("os.execvp returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
