"""Durable desktop boundary: typed actions, observations and uncertain delivery."""
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import aiohttp
from .coordinates import dispatch_arguments
from .action_validation import ActionValidationError, validate_action
from .targeting import TargetError, identify_controls


class DesktopError(RuntimeError):
    pass


class DesktopRejected(DesktopError):
    """The executor explicitly rejected a request before dispatching it."""
    def __init__(self,message,*,observation=None):
        super().__init__(message)
        self.observation=observation


class DesktopValidationError(DesktopError):
    """A public tool field is invalid; no desktop request was dispatched."""
    def __init__(self, problem):
        self.problem = problem
        super().__init__(f"{problem['path']}: {problem['message']} {problem.get('hint', '')}")


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
        self.db.execute("CREATE TABLE IF NOT EXISTS coordinate_dispatches (id TEXT PRIMARY KEY, mapping TEXT)")
        self.db.execute('CREATE TABLE IF NOT EXISTS revalidations(id TEXT,source TEXT,evidence TEXT,PRIMARY KEY(id,source))')
        self.db.commit()
        self.session = None
        self.latest = None
        self.paused = True
        self.task_id = "manual"
        self.lock = asyncio.Lock()
        self.execution = None
        self.lease = None

    async def start(self):
        if self.session and not self.session.closed:
            return await self.observe()
        from .device_lease import DeviceLease
        self.lease = DeviceLease(self.endpoint).acquire()
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
        metadata = identify_controls({k: v for k, v in observation.items() if k != "image"})
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
            metadata = self.remember(observation)
            result = metadata | {"image": observation["image"]}
            if self.execution:
                result["step_check"] = self.execution.after_delivery(metadata)
            return result

    def unresolved(self):
        return [{"action_id": json.loads(row[1])["action_id"], "receipt_id": row[0],
                 "request": json.loads(row[1]), "status": row[2]}
                for row in self.db.execute("SELECT id, request, status FROM actions WHERE status IN ('dispatching','uncertain')")]

    def action_receipt_id(self, action_id):
        """The executor's durable namespace is shared by all application tasks.

        Preserve legacy IDs only when the saved row belongs to this task.
        New rows use a stable task-scoped wire ID, including after a restart.
        """
        if not isinstance(self.task_id, str) or not self.task_id:
            raise DesktopError("A saved task identity is required for desktop actions")
        key = "q3_" + hashlib.sha256(json.dumps([self.task_id, action_id],
            ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
        row = self.db.execute("SELECT task FROM actions WHERE id=?", (key,)).fetchone()
        if row:
            if row[0] != self.task_id:
                raise DesktopError("Saved desktop action belongs to another task")
            return key
        legacy = self.db.execute("SELECT task FROM actions WHERE id=?", (action_id,)).fetchone()
        return action_id if legacy and legacy[0] == self.task_id else key

    def checked_receipt(self, result, receipt_id, request):
        if (result.get("action_id") != receipt_id or result.get("status") != "completed"
                or result.get("observation", {}).get("desktop_id") != request["desktop_id"]):
            raise DesktopError("Executor receipt does not match the saved action and desktop; reconcile before continuing")
        return result | {"action_id": request["action_id"], "receipt_id": receipt_id}

    async def control(self, command, acknowledge_uncertain=False):
        if command != "resume":
            self.paused = True
        if command == "resume":
            # Query the durable remote receipt without reissuing an action.
            for action in self.unresolved():
                receipt = await self.request("GET", "/actions/" + action["receipt_id"])
                if receipt.get("status") == "completed":
                    result = self.checked_receipt(receipt["result"], action["receipt_id"], action["request"])
                    mapping = self.db.execute("SELECT mapping FROM coordinate_dispatches WHERE id=?", (action["receipt_id"],)).fetchone()
                    if mapping:
                        result = result | {"coordinate_mapping": json.loads(mapping[0])}
                    metadata = self.remember(result["observation"])
                    result = result | {"observation": metadata}
                    self.db.execute("UPDATE actions SET status='completed', result=? WHERE id=?",
                                    (json.dumps(result), action["receipt_id"]))
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
            try:
                validate_action(arguments)
            except ActionValidationError as exc:
                raise DesktopValidationError(exc.problem) from exc
            action_id = arguments["action_id"]
            receipt_id = self.action_receipt_id(action_id)
            serialized = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
            prior = self.db.execute("SELECT request,status,result FROM actions WHERE id=?", (receipt_id,)).fetchone()
            if prior:
                if prior[0] != serialized:
                    raise DesktopError("action_id reused with different arguments")
                if prior[1] == "completed":
                    result = json.loads(prior[2])
                    return result | {"observation": self.with_image(result["observation"]), "duplicate": True}
                raise DesktopError(f"Previous action is {prior[1]}; observe/reconcile, do not retry it")
            if self.latest is None or arguments.get("desktop_id") != self.latest["desktop_id"]:
                raise DesktopValidationError({"code": "unknown_desktop", "path": "$.desktop_id",
                    "message": "Unknown desktop; observe first", "hint": "Copy desktop_id from a fresh observation."})
            if arguments.get("observation_id") != self.latest["observation_id"]:
                raise DesktopValidationError({"code": "stale_observation", "path": "$.observation_id",
                    "message": "Stale observation; observe again", "hint": "Use the latest observation_id and reassess the current screen."})
            if (self.execution and arguments["action"]["kind"] in {"click", "drag", "scroll"}
                    and arguments.get("coordinate_space") != "normalized_1000"):
                raise DesktopValidationError({"code": "normalized_coordinates_required", "path": "$.coordinate_space",
                    "message": "Verified visual input requires explicit normalized_1000 coordinates.",
                    "hint": "For screenshot estimates use0..1000 on both axes with coordinate_space=normalized_1000. "
                            "For an exposed system control use click_control and its copied control_id. No input was sent; do not relabel pixel values as normalized."})
            try:
                dispatched, mapping = dispatch_arguments(arguments, self.latest)
            except TargetError as exc:
                raise DesktopValidationError(exc.problem) from exc
            except (ValueError, TypeError, KeyError) as exc:
                raise DesktopValidationError({"code": "invalid_coordinates", "path": "$.coordinate_space",
                    "message": str(exc), "hint": "normalized_1000 is only for click, drag or scroll, with integers from 0 to 1000. Use pixels or omit this field for key, type and open_app."}) from exc
            revalidation=None
            captured=self.latest.get('captured_at')
            if self.execution and type(captured) in (int,float) and time.time()*1000-captured>=50000:
                old=self.latest
                fresh=self.remember(await self.request('GET','/observe'))
                identity=('desktop_id','device_id','session_id','display_id','width','height','rotation')
                same_device=all(old.get(k)==fresh.get(k) for k in identity)
                def controls(observation):
                    a11y=observation.get('accessibility',{})
                    return a11y | {'controls':[{k:v for k,v in c.items() if k!='control_id'}
                        for c in a11y.get('controls',[])]}
                same_view=(old.get('sha256')==fresh.get('sha256') and old.get('foreground')==fresh.get('foreground')
                           and controls(old)==controls(fresh))
                independent_app_intent=arguments['action']['kind']=='open_app'
                revalidation={'source_observation_id':old['observation_id'],'fresh_observation_id':fresh['observation_id'],
                    'age_milliseconds':time.time()*1000-captured,'same_device':same_device,'same_view':same_view,
                    'scope':'Explicit allowed app intent' if independent_app_intent else 'Exact screenshot, foreground and control evidence',
                    'input_sent':False}
                self.db.execute('INSERT INTO revalidations VALUES (?,?,?)',(receipt_id,old['observation_id'],json.dumps(revalidation)))
                self.db.commit()
                if not same_device or not (same_view or independent_app_intent):
                    raise DesktopRejected('The observation aged while reasoning and the current view changed. '
                        'No input was sent. Retain the existing plan; reassess its next target against this current observation.',
                        observation=self.with_image(fresh))
                dispatched=dispatched | {'desktop_id':fresh['desktop_id'],'observation_id':fresh['observation_id']}
            if self.execution:
                self.execution.before_dispatch(arguments, dispatched, receipt_id)
            elif "step" in arguments:
                raise DesktopValidationError({"code":"unsupported_step","path":"$.step",
                    "message":"Expected-result steps require verified execution mode","hint":"Use the selected mode's advertised tool schema."})
            dispatched = {k: v for k, v in dispatched.items() if k != "step"}
            self.db.execute("INSERT INTO actions VALUES (?, ?, ?, 'dispatching', NULL, ?)",
                            (receipt_id, self.task_id, serialized, time.time()))
            if mapping:
                self.db.execute("INSERT INTO coordinate_dispatches VALUES (?, ?)", (receipt_id, json.dumps(mapping)))
            self.db.commit()
            try:
                result = await self.request("POST", "/act", dispatched | {"action_id": receipt_id})
                result = self.checked_receipt(result, receipt_id, arguments)
                if mapping:
                    result = result | {"coordinate_mapping": mapping}
                if revalidation:
                    result=result | {'observation_revalidation':revalidation | {'input_sent':True}}
                metadata = self.remember(result["observation"])
                result = result | {"observation": metadata | {"image": result["observation"]["image"]}}
                self.db.execute("UPDATE actions SET status='completed', result=? WHERE id=?",
                                (json.dumps(result | {"observation": metadata}), receipt_id))
                self.db.commit()
                if result.get("no_progress"):
                    self.paused = True
                if self.execution:
                    result = result | {"step_check": self.execution.after_delivery(result["observation"])}
                return result
            except DesktopRejected as exc:
                self.db.execute("UPDATE actions SET status='rejected', result=? WHERE id=?",
                                (json.dumps({"error": str(exc)}), receipt_id))
                self.db.commit()
                raise
            except BaseException:
                # Even a client cancellation can arrive after the guest clicked.
                # Keep the action ID permanently reserved and pause dispatch.
                self.paused = True
                self.db.execute("UPDATE actions SET status='uncertain' WHERE id=?", (receipt_id,))
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
        if self.lease:
            self.lease.close()


def model_observation(observation):
    """Compact presentation only; the executor retains full evidence unchanged."""
    metadata = {k: v for k, v in observation.items() if k != "image"}
    accessibility = observation.get("accessibility")
    if isinstance(accessibility, dict):
        controls = []
        for control in accessibility.get("controls", []):
            visible = {k: v for k, v in control.items() if k not in {
                "x", "y", "width", "height", "resource_id"}}
            if visible.get("value") in (None, "", visible.get("name")):
                visible.pop("value", None)
            states = visible.get("states")
            if isinstance(states, dict):
                # Explicit unchecked/unselected is useful for toggles; false
                # scroll/click flags are redundant when positive flags exist.
                visible["states"] = {k: v for k, v in states.items()
                                     if v or k in {"checked", "selected", "enabled", "focused"}}
            if all(k in control for k in ("x", "y", "width", "height")):
                visible["bounds"] = [control[k] for k in ("x", "y", "width", "height")]
            controls.append(visible)
        metadata["accessibility"] = accessibility | {"controls": controls}
    return metadata


def mcp_observation(observation, result=None):
    metadata = model_observation(observation)
    text = {"observation": metadata}
    if result:
        text.update({k: v for k, v in result.items() if k != "observation"})
    return {"content": [{"type": "text", "text": json.dumps(text, ensure_ascii=False)},
                        {"type": "image", "data": observation["image"], "mimeType": "image/png"}]}
