"""Canonical module entry: driver and gateway must share exception identities.

Launching device_gateway itself with -m creates a second __main__ class while
drivers import qw3_computer.device_gateway.DeviceRejected. Import its runner
normally here so a confirmed preflight rejection stays a non-delivery response.
"""
import argparse
import asyncio
from pathlib import Path

from qw3_computer.device_gateway import run


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--platform',choices=['linux','android'],required=True)
    result.add_argument('--driver-endpoint',required=True)
    result.add_argument('--device-id',required=True)
    result.add_argument('--runtime',type=Path,required=True)
    result.add_argument('--port',type=int,required=True)
    result.add_argument('--android-serial',default='emulator-5556')
    return result


if __name__=='__main__':asyncio.run(run(parser().parse_args()))
