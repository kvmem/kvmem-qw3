"""Observe stays bounded and returns honest evidence if AT-SPI is unavailable."""
import io
import json
import unittest

from aiohttp import web
from PIL import Image

from qw3_computer.linux_driver import LinuxDriver, FOREGROUND_SCRIPT
from test_bridge import serve


class LinuxCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests=[]
        self.fail_tree=False
        image=io.BytesIO()
        Image.new('RGB',(128,128),'blue').save(image,format='PNG')
        self.png=image.getvalue()
        self.foreground={'handle':'0x1','process':'editor','title':'Note'}
        async def execute(request):
            body=await request.json()
            self.requests.append(body)
            script=body['command'][-1]
            if script==FOREGROUND_SCRIPT:
                output=json.dumps(self.foreground)
            elif self.fail_tree:
                return web.json_response({'status':'error','message':'Capture deadline exceeded'},status=500)
            else:
                xml='''<frame name="Note" xmlns:s="state" xmlns:c="component"><button name="Save" s:visible="true" s:showing="true" s:enabled="true" c:screencoord="(2, 4)" c:size="(10, 8)"/></frame>'''
                output=json.dumps({'status':'ok','AT':xml})
            return web.json_response({'status':'success','returncode':0,'output':output})
        async def screenshot(_):
            return web.Response(body=self.png,content_type='image/png')
        app=web.Application()
        app.router.add_post('/execute',execute)
        app.router.add_get('/screenshot',screenshot)
        self.runner,origin=await serve(app)
        self.driver=LinuxDriver(origin,'capture-fixture')
        await self.driver.start()

    async def asyncTearDown(self):
        await self.driver.close()
        await self.runner.cleanup()

    async def test_foreground_read_has_guest_deadline_and_uses_official_traversal(self):
        observation=await self.driver.observe()
        self.assertEqual(observation['png'],self.png)
        self.assertEqual([c['name'] for c in observation['accessibility']['controls']],['Save'])
        self.assertEqual(observation['accessibility']['scope'],'foreground_window')
        tree_requests=[r for r in self.requests if r['command'][-1]!=FOREGROUND_SCRIPT]
        self.assertEqual(len(tree_requests),1)
        self.assertEqual(tree_requests[0]['timeout'],4)
        self.assertIn('a.create_atspi_node',tree_requests[0]['command'][-1])
        self.assertEqual(len(self.requests),3)

    async def test_unavailable_tree_keeps_image_and_marks_controls_unavailable_without_retry(self):
        self.fail_tree=True
        observation=await self.driver.observe()
        self.assertEqual(observation['png'],self.png)
        self.assertEqual(observation['foreground'],self.foreground)
        self.assertEqual(observation['accessibility']['status'],'unavailable')
        self.assertEqual(observation['accessibility']['controls'],[])
        self.assertEqual(observation['accessibility']['error_kind'],'RuntimeError')
        self.assertEqual(len(self.requests),3)


if __name__=='__main__':
    unittest.main()
