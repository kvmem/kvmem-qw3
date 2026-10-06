"""Separate opt-in Mac gateway, retaining canonical receipt exception identities."""
import argparse
import asyncio
from pathlib import Path
import secrets
import signal

from aiohttp import web

from macos_agent_driver import MacDriver
from qw3_computer.device_gateway import Gateway, create_app
from qw3_computer.device_lease import DeviceLease


def create_mac_app(gateway, token):
    app = create_app(gateway, token)

    @web.middleware
    async def native_control(request, handler):
        if request.method == 'POST' and request.path == '/control':
            body = await request.json()
            command = body.get('command')
            if command in {'pause', 'stop'}:
                # Host pause takes effect before waiting for remote confirmation.
                gateway.paused = True
                await gateway.driver.control(command)
            elif command == 'resume':
                # Remote failure leaves host input paused.
                gateway.paused = True
                await gateway.driver.control(command)
        return await handler(request)

    # Canonical auth/error middleware runs first, before remote controls.
    app.middlewares.append(native_control)
    return app


async def run(args):
    args.runtime.mkdir(parents=True, exist_ok=True)
    token_path = args.runtime / 'token.txt'
    if not token_path.exists():
        token_path.write_text(secrets.token_hex(32))
        token_path.chmod(0o600)
    driver = MacDriver(args.driver_endpoint, args.device_id, args.native_token, args.runtime / 'native-receipts')
    # Tunnel ports can change; lease identity follows the concrete Mac.
    lease = DeviceLease('macos:' + args.device_id).acquire()
    gateway = runner = None
    try:
        await driver.start()
        gateway = Gateway(driver, args.runtime)
        runner = web.AppRunner(create_mac_app(gateway, token_path.read_text().strip()))
        await runner.setup()
        await web.TCPSite(runner, '127.0.0.1', args.port).start()
        print('Mac gateway ready, initially paused', flush=True)
        done = asyncio.Event()
        loop = asyncio.get_running_loop()
        for item in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(item, done.set)
        await done.wait()
    finally:
        if runner:
            await runner.cleanup()
        if gateway:
            await gateway.close()
        else:
            await driver.close()
        lease.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--driver-endpoint', required=True)
    result.add_argument('--device-id', required=True)
    result.add_argument('--native-token', type=Path, required=True)
    result.add_argument('--runtime', type=Path, required=True)
    result.add_argument('--port', type=int, required=True)
    return result


if __name__ == '__main__':
    asyncio.run(run(parser().parse_args()))
