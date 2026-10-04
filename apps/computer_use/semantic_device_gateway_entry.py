"""Opt-in Linux application observations through the ordinary device gateway."""
import asyncio

from device_gateway_entry import parser
from linux_app_semantics import SemanticLinuxDriver
from qw3_computer.device_gateway import run


def main():
    args=parser().parse_args()
    if args.platform!='linux':raise ValueError('This qualified behavior adapter is Linux-only')
    driver=SemanticLinuxDriver(args.driver_endpoint,args.device_id)
    asyncio.run(run(args,driver=driver))


if __name__=='__main__':main()
