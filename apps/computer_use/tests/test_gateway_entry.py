"""A driver preflight rejection must never become ambiguous delivery."""
import tempfile
from pathlib import Path
import types
import unittest

from aiohttp.test_utils import TestClient,TestServer

from qw3_computer import device_gateway
from apps.computer_use.tests.test_device_gateway import FakeDriver
from device_gateway_entry import run


class GatewayEntryTests(unittest.IsolatedAsyncioTestCase):
    async def boundary(self, module):
        with tempfile.TemporaryDirectory() as root:
            driver=FakeDriver();driver.mode='preflight_reject'
            gateway=module.Gateway(driver,Path(root))
            await gateway.observe();gateway.paused=False
            request=dict(action_id='preflight-one',desktop_id=gateway.desktop_id,
                         observation_id=gateway.latest['observation_id'],action={'kind':'key','key':'ESC'})
            client=TestClient(TestServer(module.create_app(gateway,'entry-test')))
            await client.start_server()
            try:
                response=await client.post('/act',json=request,headers={'Authorization':'Bearer entry-test'})
                body=await response.json()
                return response.status,body,driver.effects,gateway.receipt('preflight-one')['status']
            finally:
                await client.close();await gateway.close()

    async def test_duplicate_module_class_reproduces_false_uncertainty_before_any_input(self):
        duplicate=types.ModuleType('duplicate_gateway_main')
        duplicate.__package__='qw3_computer'
        exec(compile(Path(device_gateway.__file__).read_text(),'<duplicate -m namespace>','exec'),duplicate.__dict__)
        status,body,effects,receipt=await self.boundary(duplicate)
        self.assertEqual(status,503)
        self.assertNotEqual(body.get('delivery'),'rejected')
        self.assertEqual(effects,0)
        self.assertEqual(receipt,'unknown')

    async def test_canonical_runner_keeps_preflight_rejection_as_zero_input(self):
        self.assertIs(run,device_gateway.run)
        status,body,effects,receipt=await self.boundary(device_gateway)
        self.assertEqual(status,409)
        self.assertEqual(body['delivery'],'rejected')
        self.assertEqual(effects,0)
        self.assertEqual(receipt,'unknown')


if __name__=='__main__':unittest.main()
