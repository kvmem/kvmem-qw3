"""Authenticated local gateway around mature device drivers with durable receipts."""
import argparse
import asyncio
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import secrets
import sqlite3
import time
import uuid

from aiohttp import web
from PIL import Image

from .action_validation import validate_action
from .device_protocol import capabilities


class DeviceRejected(RuntimeError):
    """The adapter confirms that it did not send input."""


class Gateway:
    def __init__(self, driver, runtime):
        self.driver = driver
        self.runtime = Path(runtime)
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.runtime / "receipts.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS actions(id TEXT PRIMARY KEY,request TEXT,status TEXT,result TEXT)")
        self.db.commit()
        self.paused = True
        self.latest = None
        self.lock = asyncio.Lock()
        # A gateway/driver restart invalidates all old captures. Pending IDs stay reserved.
        self.session_id = uuid.uuid4().hex
        self.desktop_id = driver.device_id + ":" + self.session_id

    async def observe(self):
        raw = await self.driver.observe()
        image = raw.pop("png")
        with Image.open(io.BytesIO(image)) as decoded:
            if decoded.format != "PNG":
                raise RuntimeError("Driver did not return a PNG")
            width, height = decoded.size
            decoded.load()
        digest = hashlib.sha256(image).hexdigest()
        self.latest = raw | {"desktop_id": self.desktop_id, "device_id": self.driver.device_id,
            "session_id": self.session_id, "display_id": "primary", "rotation": raw.get("rotation", 0),
            "observation_id": uuid.uuid4().hex, "captured_at": int(time.time()*1000),
            "width": width, "height": height, "sha256": digest, "image": base64.b64encode(image).decode(),
            "mime_type": "image/png", "paused": self.paused, "capabilities": self.driver.capabilities}
        return self.latest

    async def act(self, arguments):
        validate_action(arguments)
        request = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        action_id = arguments["action_id"]
        row = self.db.execute("SELECT request,status,result FROM actions WHERE id=?", (action_id,)).fetchone()
        if row:
            if row[0] != request:
                raise DeviceRejected("action_id was reused with different content")
            if row[1] == "completed":
                return json.loads(row[2]) | {"duplicate": True}
            raise DeviceRejected("Action identity remains reserved; reconcile its receipt, never resend")
        if self.paused:
            raise DeviceRejected("Device is paused")
        latest = self.latest
        if not latest or arguments["desktop_id"] != self.desktop_id or arguments["observation_id"] != latest["observation_id"]:
            raise DeviceRejected("Desktop or observation identity is stale")
        if int(time.time()*1000) - latest["captured_at"] > 60000:
            raise DeviceRejected("Observation expired; observe again")
        action = arguments["action"]
        if action["kind"] not in self.driver.capabilities["actions"]:
            raise DeviceRejected("Action is unsupported on this device")
        # Gateway accepts only resolved pixels. Agent coordinates are converted by Desktop.
        if arguments.get("coordinate_space", "pixels") != "pixels":
            raise DeviceRejected("Gateway accepts resolved pixel coordinates only")
        for x,y in (("x","y"),("from_x","from_y"),("to_x","to_y")):
            if x in action and not (0 <= action[x] < latest["width"] and 0 <= action[y] < latest["height"]):
                raise DeviceRejected("Coordinates are outside the captured display")
        await self.driver.preflight(action, latest)
        self.db.execute("INSERT INTO actions VALUES (?,?,'dispatching',NULL)", (action_id, request))
        self.db.commit()
        try:
            # Exactly one driver call. Network timeouts are not automatically retried.
            delivery = await self.driver.dispatch(action)
            await asyncio.sleep(0.25)
            observation = await self.observe()
            captures = 1
            if observation.get('accessibility', {}).get('status') == 'foreground_changed' and not self.paused:
                # A transition invalidated the source read. Recapture once,
                # without input, before making the model spend another turn.
                await asyncio.sleep(0.2)
                observation = await self.observe()
                captures += 1
            result = {"action_id": action_id, "status": "completed", "delivery": "delivered",
                      "verification": {"passed": None, "scope": "Input delivery only; check step conditions."},
                      "driver_receipt": delivery, "observation": observation,
                      "post_input_captures": captures}
            self.db.execute("UPDATE actions SET status='completed',result=? WHERE id=?", (json.dumps(result),action_id))
            self.db.commit()
            return result
        except DeviceRejected as exc:
            self.db.execute("UPDATE actions SET status='rejected',result=? WHERE id=?", (json.dumps({"error":str(exc)}),action_id))
            self.db.commit()
            raise
        except BaseException as exc:
            self.paused = True
            self.db.execute("UPDATE actions SET status='uncertain' WHERE id=?", (action_id,))
            self.db.commit()
            if isinstance(exc,(ValueError,KeyError,TypeError)):
                raise RuntimeError("Driver input or observation failed after reservation; delivery is uncertain") from exc
            raise

    def receipt(self, action_id):
        row = self.db.execute("SELECT status,result FROM actions WHERE id=?", (action_id,)).fetchone()
        return {"status": row[0], "result": json.loads(row[1]) if row[1] else None} if row else {"status": "unknown"}

    async def close(self):
        await self.driver.close()
        self.db.close()


def create_app(gateway, token):
    @web.middleware
    async def boundary(request, handler):
        if request.headers.get("Authorization") != "Bearer " + token:
            return web.json_response({"error": "Unauthorized"}, status=401)
        try:
            return await handler(request)
        except DeviceRejected as exc:
            return web.json_response({"delivery": "rejected", "error": str(exc)}, status=409)
        except (ValueError, KeyError, TypeError) as exc:
            return web.json_response({"delivery": "rejected", "error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": type(exc).__name__ + ": " + str(exc)}, status=503)

    app = web.Application(middlewares=[boundary], client_max_size=128*1024)

    async def observe(request):
        async with gateway.lock:
            return web.json_response(await gateway.observe())

    async def act(request):
        arguments = await request.json()
        async with gateway.lock:
            return web.json_response(await gateway.act(arguments))

    async def control(request):
        command = (await request.json()).get("command")
        if command not in {"resume", "pause", "stop"}:
            raise DeviceRejected("Unsupported control command")
        # Pause is immediate, even while a capture or input receipt is in flight.
        gateway.paused = command != "resume"
        return web.json_response({"paused": gateway.paused})

    async def receipt(request):
        return web.json_response(gateway.receipt(request.match_info["action_id"]))

    async def files(request):
        # Business artifacts are collected by independent, task-specific runners.
        return web.json_response({"files": []})

    app.router.add_get("/observe", observe)
    app.router.add_post("/act", act)
    app.router.add_post("/control", control)
    app.router.add_get("/actions/{action_id}", receipt)
    app.router.add_get("/files", files)
    return app


async def run(args, driver=None):
    runtime = args.runtime.resolve()
    runtime.mkdir(parents=True,exist_ok=True)
    token_path = runtime / "token.txt"
    if not token_path.exists():
        token_path.write_text(secrets.token_hex(32))
        token_path.chmod(0o600)
    if driver is not None:
        if driver.device_id != args.device_id or driver.endpoint.rstrip('/') != args.driver_endpoint.rstrip('/'):
            raise ValueError('Injected driver identity differs from the declared device')
    elif args.platform == "linux":
        from .linux_driver import LinuxDriver
        driver = LinuxDriver(args.driver_endpoint, args.device_id)
    else:
        from .android_driver import AndroidDriver
        driver = AndroidDriver(args.driver_endpoint, args.device_id, args.android_serial)
    from .device_lease import DeviceLease
    handle = DeviceLease(driver.endpoint + ":" + getattr(driver,"serial", "desktop")).acquire()
    await driver.start()
    gateway = Gateway(driver, runtime)
    runner = web.AppRunner(create_app(gateway, token_path.read_text().strip()), handler_cancellation=True)
    try:
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", args.port).start()
        print(f"{args.platform} gateway ready on 127.0.0.1:{args.port}",flush=True)
        stop = asyncio.Event()
        import signal
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,stop.set)
        await stop.wait()
    finally:
        await runner.cleanup()
        await gateway.close()
        handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("linux","android"),required=True)
    parser.add_argument("--driver-endpoint",required=True)
    parser.add_argument("--device-id",required=True)
    parser.add_argument("--android-serial",default="emulator-5556")
    parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("--port",type=int,required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":main()
