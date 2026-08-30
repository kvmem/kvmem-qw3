"""Official Pier MiniSweAgent with local-run reliability hardening.

The agent prompt, tools, execution loop, trajectory conversion, version, and
network policy are inherited unchanged.  The adapter hardens the curl transport
used to fetch the pinned uv installer and removes local-machine limits that are
shorter than the official DeepSWE task commands themselves.
"""

import base64
import os
from pathlib import Path
from typing import Any

import yaml

from pier.agents.installed.mini_swe_agent import MiniSweAgent
from pier.models.agent.install import AgentInstallSpec


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a small harness policy into an upstream config."""

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_config(base[key], value)
        else:
            base[key] = value
    return base


class ReliableMiniSweAgent(MiniSweAgent):
    """Keep official solving semantics while hardening slow local execution."""

    # MiniSweAgent's stock 30-second shell timeout is shorter than the 300-second
    # test commands embedded in several locked DeepSWE tasks.  A timeout only
    # returns an observation to the model, but repeatedly killing cargo/go/npm
    # builds can prevent an otherwise valid trajectory from ever completing.
    # The task-level Pier timeout remains the outer safety boundary.
    _LOCAL_RUN_POLICY: dict[str, Any] = {
        "agent": {
            "step_limit": 0,
            "cost_limit": 0,
            "wall_time_limit_seconds": 0,
            "max_consecutive_format_errors": 10,
        },
        "environment": {"timeout": 1800},
        "model": {"model_kwargs": {"timeout": 21600}},
    }

    _UPSTREAM_UV_INSTALL = (
        "curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh"
    )
    _UV_CACHE_SHA256 = (
        "04e7399b45054f5ae4239ed60cd579311daafdd8d43e0e6ac01003436f19eaac"
    )
    _UV_CACHE_URL = os.environ.get(
        "QW3_PIER_UV_CACHE_URL",
        "http://172.17.0.1:8766/uv-0.7.13",
    )
    _RELIABLE_UV_INSTALL = rf'''mkdir -p "$HOME/.local/bin"
uv_cache_url={_UV_CACHE_URL!r}
if curl --noproxy '*' --http1.1 --connect-timeout 2 --max-time 10 \
        --retry 1 --retry-all-errors -fsS "$uv_cache_url" -o /tmp/qw3-uv; then
    printf '%s  %s\n' {_UV_CACHE_SHA256!r} /tmp/qw3-uv | sha256sum -c -
    install -m 0755 /tmp/qw3-uv "$HOME/.local/bin/uv"
    ln -sf uv "$HOME/.local/bin/uvx"
    cat > "$HOME/.local/bin/env" <<'ENV'
#!/bin/sh
case ":${{PATH}}:" in
    *:"${{HOME}}/.local/bin":*) ;;
    *) export PATH="${{HOME}}/.local/bin:${{PATH}}" ;;
esac
ENV
else
cat > "$HOME/.curlrc" <<'CURLRC'
http1.1
retry = 5
retry-all-errors
retry-delay = 2
connect-timeout = 20
CURLRC
curl --http1.1 --retry 5 --retry-all-errors -LsSf https://astral.sh/uv/0.7.13/install.sh | sh
rm -f "$HOME/.curlrc"
fi
rm -f /tmp/qw3-uv'''

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        configured = yaml.safe_load(self._config_yaml or "{}") or {}
        if not isinstance(configured, dict):
            raise TypeError("MiniSweAgent custom config must be a YAML mapping")
        _merge_config(configured, self._LOCAL_RUN_POLICY)
        self._config_yaml = yaml.safe_dump(configured, sort_keys=False)

    def install_spec(self) -> AgentInstallSpec:
        spec = super().install_spec()
        matches = sum(
            step.run.count(self._UPSTREAM_UV_INSTALL) for step in spec.steps
        )
        if matches != 1:
            raise RuntimeError(
                "Pinned Pier MiniSweAgent uv installer changed: "
                f"expected one exact command, found {matches}"
            )
        for step in spec.steps:
            step.run = step.run.replace(
                self._UPSTREAM_UV_INSTALL,
                self._RELIABLE_UV_INSTALL,
            )
        spec.metadata = {
            **spec.metadata,
            "upstream_agent": "pier.agents.installed.mini_swe_agent:MiniSweAgent",
            "transport_only_patch": (
                "prefer a SHA-256-verified bridge-local uv 0.7.13 cache; "
                "otherwise force curl HTTP/1.1 and retry the pinned download"
            ),
            "uv_cache_url": self._UV_CACHE_URL,
            "uv_cache_sha256": self._UV_CACHE_SHA256,
            "local_run_policy": self._LOCAL_RUN_POLICY,
        }
        return spec


class CompactingMiniSweAgent(ReliableMiniSweAgent):
    """Reliable official agent with a dense-256K compaction runtime."""

    _RUNTIME_MODULE = Path(__file__).with_name(
        "compacting_mini_swe_agent_runtime.py"
    )
    _COMPACTION_POLICY: dict[str, Any] = {
        "agent": {
            "agent_class": (
                "qw3_compacting_agent.CompactingInteractiveAgent"
            ),
            "compaction_trigger_tokens": 220_000,
            "compaction_summary_max_tokens": 16_384,
            "compaction_retry_attempts": 3,
        }
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        configured = yaml.safe_load(self._config_yaml or "{}") or {}
        if not isinstance(configured, dict):
            raise TypeError("MiniSweAgent custom config must be a YAML mapping")
        _merge_config(configured, self._COMPACTION_POLICY)
        self._config_yaml = yaml.safe_dump(configured, sort_keys=False)

    def install_spec(self) -> AgentInstallSpec:
        spec = super().install_spec()
        runtime = base64.b64encode(self._RUNTIME_MODULE.read_bytes()).decode(
            "ascii"
        )
        injection = f'''\
site_dir="$($python_bin -c 'import site; print(site.getsitepackages()[0])')"
printf '%s' '{runtime}' | base64 -d > "$site_dir/qw3_compacting_agent.py"
"$python_bin" -c 'from qw3_compacting_agent import CompactingInteractiveAgent; print(CompactingInteractiveAgent.__name__)'
'''
        targets = [
            step for step in spec.steps if "mini-swe-agent --help" in step.run
        ]
        if len(targets) != 1:
            raise RuntimeError(
                "Pinned MiniSweAgent install layout changed: expected one "
                f"runtime injection target, found {len(targets)}"
            )
        targets[0].run = targets[0].run.replace(
            "mini-swe-agent --help", injection + "\nmini-swe-agent --help"
        )
        spec.metadata = {
            **spec.metadata,
            "compaction_runtime": str(self._RUNTIME_MODULE),
            "compaction_policy": self._COMPACTION_POLICY,
        }
        return spec
