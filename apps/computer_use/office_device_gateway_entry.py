"""Opt-in bounded Office observations on the ordinary Linux device gateway."""
import asyncio

from bounded_linux_observation import BoundedOfficeLinuxDriver
from device_gateway_entry import parser
from qw3_computer.device_gateway import run


def main():
    args=parser().parse_args()
    if args.platform!='linux':raise ValueError('The Office observation adapter is Linux-only')
    driver=BoundedOfficeLinuxDriver(args.driver_endpoint,args.device_id)
    asyncio.run(run(args,driver=driver))


if __name__=='__main__':main()
