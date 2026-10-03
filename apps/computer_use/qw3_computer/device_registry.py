"""Owner-supplied device catalogue; models cannot add endpoints or permissions."""
import json
from pathlib import Path
import re
from urllib.parse import urlparse

from .desktop import Desktop, DesktopError


def load_registry(path):
    path = Path(path).resolve()
    body = json.loads(path.read_text())
    if not isinstance(body, dict) or set(body) != {"devices"} or not isinstance(body["devices"], list) or not body["devices"]:
        raise ValueError("Device registry must contain a nonempty devices list")
    devices = {}
    endpoints = set()
    for item in body["devices"]:
        if not isinstance(item, dict) or set(item) != {"id", "label", "platform", "endpoint", "token_file"}:
            raise ValueError("Device entries require id, label, platform, endpoint and token_file")
        id = item["id"]
        if not isinstance(id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", id) or id in devices:
            raise ValueError("Device identity is invalid or repeated")
        if item["platform"] not in {"windows", "linux", "android"}:
            raise ValueError("This release supports Windows, Linux and Android adapters")
        if not isinstance(item["label"], str) or not 1 <= len(item["label"]) <= 100:
            raise ValueError("Device label must contain 1..100 characters")
        url = urlparse(item["endpoint"])
        if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1"} or not url.port or url.path not in {"", "/"} or url.username or url.query or url.fragment:
            raise ValueError("First-release device endpoints must be explicit local HTTP ports")
        endpoint = f"http://127.0.0.1:{url.port}"
        if endpoint in endpoints:
            raise ValueError("Two entries cannot refer to the same input endpoint")
        token = Path(item["token_file"])
        if not token.is_absolute(): token = path.parent / token
        devices[id] = item | {"endpoint": endpoint, "token_file": token.resolve()}
        endpoints.add(endpoint)
    return devices


class RegisteredDesktop:
    def __init__(self, path, selected, runtime):
        self.devices = load_registry(path)
        self.base_runtime = Path(runtime)
        self.selected = selected or next(iter(self.devices))
        if self.selected not in self.devices: raise ValueError("Unknown selected device")
        self.active = self.make(self.selected)

    def make(self, id):
        item = self.devices[id]
        token = item["token_file"].read_text().strip()
        if not token: raise ValueError("Selected device token is empty")
        return Desktop(item["endpoint"], token, self.base_runtime / id)

    @property
    def platform(self): return self.devices[self.selected]["platform"]

    def catalogue(self):
        return {"selected": self.selected, "devices": [{k: item[k] for k in ("id", "label", "platform")} for item in self.devices.values()]}

    def __getattr__(self, name): return getattr(self.active, name)

    @property
    def task_id(self): return self.active.task_id

    @task_id.setter
    def task_id(self, value): self.active.task_id = value

    @property
    def execution(self): return self.active.execution

    @execution.setter
    def execution(self, value): self.active.execution = value

    async def start(self): return await self.active.start()

    async def select(self, id):
        if id not in self.devices: raise DesktopError("Unknown device")
        if id == self.selected: return await self.active.observe()
        if self.active.unresolved(): raise DesktopError("Resolve the current device's uncertain input before transferring ownership")
        candidate = self.make(id)
        execution = self.active.execution
        try:
            observation = await candidate.start()
        except BaseException:
            await candidate.close();raise
        await self.active.control("pause")
        await self.active.close()
        self.active = candidate
        self.active.execution = execution
        self.selected = id
        return observation

    async def close(self): await self.active.close()
