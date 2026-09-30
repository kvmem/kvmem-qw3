"""Durable desktop boundary: typed actions, observations and uncertain delivery."""
import asyncio
import base64
import hashlib
import json
import re
from pathlib import Path
import sqlite3
import time

import aiohttp


class DesktopError(RuntimeError):
    pass


class DesktopRejected(DesktopError):
    """The executor explicitly rejected a request before dispatching it."""


class Desktop:
    def __init__(self, endpoint, token, runtime):
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.runtime = Path(runtime)
        self.runtime.mkdir(parents=True, exist_ok=True)
        (self.runtime / "screenshots").mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.runtime / "desktop.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS actions (id TEXT PRIMARY KEY, task TEXT, request TEXT, status TEXT, result TEXT, created REAL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS observations (id TEXT PRIMARY KEY, task TEXT, metadata TEXT, created REAL)")
        self.db.commit()
        self.session = None
        self.latest = None
        self.paused = True
        self.task_id = "manual"
        self.lock = asyncio.Lock()

    async def start(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20), trust_env=False,
                                             headers={"Authorization": "Bearer " + self.token})
        await self.control("pause")
        return await self.observe()

    async def request(self, method, route, body=None):
        try:
            async with self.session.request(method, self.endpoint + route, json=body) as response:
                data = await response.json(content_type=None)
                if response.status != 200:
                    if data.get("delivery") == "rejected":
                        raise DesktopRejected(data.get("error", "Desktop rejected the request"))
                    raise DesktopError(data.get("error", f"Desktop HTTP {response.status}"))
                return data
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise DesktopError(f"Desktop unavailable: {type(exc).__name__}") from exc

    def remember(self, observation):
        image = base64.b64decode(observation["image"], validate=True)
        digest = hashlib.sha256(image).hexdigest()
        if digest != observation["sha256"] or not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise DesktopError("Desktop returned an invalid screenshot")
        path = self.runtime / "screenshots" / (digest + ".png")
        if not path.exists():
            path.write_bytes(image)
        metadata = {k: v for k, v in observation.items() if k != "image"}
        self.db.execute("INSERT OR IGNORE INTO observations VALUES (?, ?, ?, ?)",
                        (metadata["observation_id"], self.task_id, json.dumps(metadata), time.time()))
        self.db.commit()
        self.latest = metadata
        return metadata

    def with_image(self, metadata):
        path = self.runtime / "screenshots" / (metadata["sha256"] + ".png")
        return metadata | {"image": base64.b64encode(path.read_bytes()).decode()}

    async def observe(self):
        async with self.lock:
            observation = await self.request("GET", "/observe")
            self.remember(observation)
            return observation

    def unresolved(self):
        return [{"action_id": row[0], "request": json.loads(row[1]), "status": row[2]}
                for row in self.db.execute("SELECT id, request, status FROM actions WHERE status IN ('dispatching','uncertain')")]

    async def control(self, command, acknowledge_uncertain=False):
        if command != "resume":
            self.paused = True
        if command == "resume":
            # Query the durable remote receipt without reissuing an action.
            for action in self.unresolved():
                receipt = await self.request("GET", "/actions/" + action["action_id"])
                if receipt.get("status") == "completed":
                    result = receipt["result"]
                    metadata = self.remember(result["observation"])
                    result = result | {"observation": metadata}
                    self.db.execute("UPDATE actions SET status='completed', result=? WHERE id=?",
                                    (json.dumps(result), action["action_id"]))
            self.db.commit()
            if self.unresolved():
                await self.observe()
                if not acknowledge_uncertain:
                    raise DesktopError("上次操作结果尚未确认。请核对当前画面，再确认继续。")
                self.db.execute("UPDATE actions SET status='user_reconciled' WHERE status IN ('dispatching','uncertain')")
                self.db.commit()
        result = await self.request("POST", "/control", {"command": command})
        self.paused = result["paused"]
        return result

    async def act(self, arguments):
        async with self.lock:
            if self.paused:
                raise DesktopError("Desktop is paused; wait for the user to resume")
            action_id = arguments.get("action_id")
            if not isinstance(action_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{8,80}", action_id):
                raise DesktopError("action_id must contain 8..80 letters, digits, underscores or hyphens")
            serialized = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
            prior = self.db.execute("SELECT request,status,result FROM actions WHERE id=?", (action_id,)).fetchone()
            if prior:
                if prior[0] != serialized:
                    raise DesktopError("action_id reused with different arguments")
                if prior[1] == "completed":
                    result = json.loads(prior[2])
                    return result | {"observation": self.with_image(result["observation"]), "duplicate": True}
                raise DesktopError(f"Previous action is {prior[1]}; observe/reconcile, do not retry it")
            if self.latest is None or arguments.get("desktop_id") != self.latest["desktop_id"]:
                raise DesktopError("Unknown desktop; observe first")
            if arguments.get("observation_id") != self.latest["observation_id"]:
                raise DesktopError("Stale observation; observe again")
            self.db.execute("INSERT INTO actions VALUES (?, ?, ?, 'dispatching', NULL, ?)",
                            (action_id, self.task_id, serialized, time.time()))
            self.db.commit()
            try:
                result = await self.request("POST", "/act", arguments)
                metadata = self.remember(result["observation"])
                self.db.execute("UPDATE actions SET status='completed', result=? WHERE id=?",
                                (json.dumps(result | {"observation": metadata}), action_id))
                self.db.commit()
                if result.get("no_progress"):
                    self.paused = True
                return result
            except DesktopRejected as exc:
                self.db.execute("UPDATE actions SET status='rejected', result=? WHERE id=?",
                                (json.dumps({"error": str(exc)}), action_id))
                self.db.commit()
                raise
            except BaseException:
                # Even a client cancellation can arrive after the guest clicked.
                # Keep the action ID permanently reserved and pause dispatch.
                self.paused = True
                self.db.execute("UPDATE actions SET status='uncertain' WHERE id=?", (action_id,))
                self.db.commit()
                try:
                    await asyncio.shield(self.request("POST", "/control", {"command": "pause"}))
                except (DesktopError, asyncio.CancelledError):
                    pass
                raise

    async def close(self):
        if self.session:
            await self.session.close()
        self.db.close()


def mcp_observation(observation, result=None):
    metadata = {k: v for k, v in observation.items() if k != "image"}
    text = {"observation": metadata}
    if result:
        text.update({k: v for k, v in result.items() if k != "observation"})
    return {"content": [{"type": "text", "text": json.dumps(text, ensure_ascii=False)},
                        {"type": "image", "data": observation["image"], "mimeType": "image/png"}]}
