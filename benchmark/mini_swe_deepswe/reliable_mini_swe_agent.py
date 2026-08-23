"""Official Pier MiniSweAgent with a transport-only installer hardening.

The agent prompt, tools, execution loop, trajectory conversion, version, and
network policy are inherited unchanged.  Only the curl transport used to fetch
the pinned uv installer is made resilient to flaky HTTP/2 transfers.
"""

from pier.agents.installed.mini_swe_agent import MiniSweAgent
from pier.models.agent.install import AgentInstallSpec


class ReliableMiniSweAgent(MiniSweAgent):
    """Keep official MiniSweAgent semantics while hardening uv installation."""

    _UPSTREAM_UV_INSTALL = (
        "curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh"
    )
    _RELIABLE_UV_INSTALL = r'''cat > "$HOME/.curlrc" <<'CURLRC'
http1.1
retry = 5
retry-all-errors
retry-delay = 2
connect-timeout = 20
CURLRC
curl --http1.1 --retry 5 --retry-all-errors -LsSf https://astral.sh/uv/0.7.13/install.sh | sh
rm -f "$HOME/.curlrc"'''

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
                "force curl HTTP/1.1 and retry the pinned uv 0.7.13 download"
            ),
        }
        return spec
