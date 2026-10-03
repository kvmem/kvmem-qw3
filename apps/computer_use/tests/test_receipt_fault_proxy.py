"""Use a real HTTP boundary and SQLite receipts to reject replay after loss."""
import tempfile
import unittest
from pathlib import Path

import aiohttp
from aiohttp import web

from device_receipt_qualification import exercise
from receipt_fault_proxy import ReceiptFaultProxy
from qw3_computer.desktop import Desktop
from qw3_computer.device_gateway import Gateway,create_app
from apps.computer_use.tests.test_device_gateway import FakeDriver


class ReceiptFaultTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root=tempfile.TemporaryDirectory()
        self.driver=FakeDriver()
        self.gateway=Gateway(self.driver,Path(self.root.name)/'gateway')
        self.backend=web.AppRunner(create_app(self.gateway,'test-token'),handler_cancellation=True)
        await self.backend.setup()
        site=web.TCPSite(self.backend,'127.0.0.1',0)
        await site.start()
        self.proxy=ReceiptFaultProxy('http://127.0.0.1:'+str(site._server.sockets[0].getsockname()[1]),'test-token')
        await self.proxy.start()
        self.desktop=None

    async def asyncTearDown(self):
        if self.desktop:
            await self.desktop.control('pause')
            await self.desktop.close()
        await self.proxy.close()
        await self.backend.cleanup()
        await self.gateway.close()
        self.root.cleanup()

    async def test_restart_reconciles_one_lost_receipt_without_second_dispatch(self):
        self.desktop=Desktop(self.proxy.origin,'test-token',Path(self.root.name)/'desktop')
        self.desktop,result=await exercise(self.desktop,self.proxy)
        self.assertTrue(result['passed'])
        self.assertEqual(self.driver.effects,1)
        self.assertEqual(len(self.proxy.receipt_queries),1)
        self.assertNotIn('test-token',str(self.proxy.evidence()))

    async def test_unauthorized_request_does_not_reach_device(self):
        async with aiohttp.ClientSession(trust_env=False) as session:
            async with session.post(self.proxy.origin+'/act',json={'action_id':'fake'}) as response:
                self.assertEqual(response.status,401)
        self.assertEqual(self.driver.effects,0)
        self.assertFalse(self.proxy.dispatches)

    async def test_repeated_qualification_preserves_unique_durable_task_namespaces(self):
        self.desktop=Desktop(self.proxy.origin,'test-token',Path(self.root.name)/'first-client')
        self.desktop,first=await exercise(self.desktop,self.proxy)
        await self.desktop.control('pause');await self.desktop.close();self.desktop=None
        endpoint=self.proxy.endpoint
        await self.proxy.close()
        self.proxy=ReceiptFaultProxy(endpoint,'test-token');await self.proxy.start()
        self.desktop=Desktop(self.proxy.origin,'test-token',Path(self.root.name)/'second-client')
        self.desktop,second=await exercise(self.desktop,self.proxy)
        self.assertTrue(first['passed'] and second['passed'])
        self.assertNotEqual(first['task_id'],second['task_id'])
        self.assertEqual(self.driver.effects,2)
        self.assertEqual(self.gateway.db.execute('select count(*) from actions').fetchone()[0],2)

    async def test_malformed_and_nondevice_routes_do_not_reach_backend(self):
        async with aiohttp.ClientSession(trust_env=False,headers={'Authorization':'Bearer test-token'}) as session:
            async with session.post(self.proxy.origin+'/act',json=[]) as response:
                self.assertEqual(response.status,400)
            async with session.post(self.proxy.origin+'/execute',json={'command':'not authorized'}) as response:
                self.assertEqual(response.status,404)
        self.assertEqual(self.driver.effects,0)
        self.assertFalse(self.proxy.dispatches)

    def test_external_or_credentialed_upstream_cannot_be_configured(self):
        for endpoint in ('http://example.invalid:1234','http://user@127.0.0.1:1234','http://127.0.0.1:1234/execute'):
            with self.assertRaises(ValueError):ReceiptFaultProxy(endpoint,'test-token')


if __name__=='__main__':unittest.main()
