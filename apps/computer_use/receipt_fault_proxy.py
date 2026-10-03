"""Owner-only transport fault: hide one completed receipt, never replay input."""
import asyncio
import json
import time
from urllib.parse import urlparse

import aiohttp
from aiohttp import web


class ReceiptFaultProxy:
    def __init__(self, endpoint, token, kind='key'):
        url=urlparse(endpoint)
        if (url.scheme!='http' or url.hostname!='127.0.0.1' or not url.port
                or url.path not in {'','/'} or url.query or url.fragment or url.username):
            raise ValueError('Fault transport requires an explicit local device endpoint')
        if not token or kind not in {'key','type','click'}:
            raise ValueError('Fault transport requires authentication and a supported input kind')
        self.endpoint=endpoint.rstrip('/')
        self.token=token
        self.kind=kind
        self.delivered=asyncio.Event()
        self.release=asyncio.Event()
        self.selected=None
        self.dispatches={}
        self.receipt_queries=[]
        self.events=[]
        self.runner=None
        self.session=None

    def record(self, event, **fields):
        self.events.append(dict(event=event,at=time.time(),**fields))

    async def forward(self, request):
        if request.headers.get('Authorization')!='Bearer '+self.token:
            return web.json_response({'error':'Unauthorized'},status=401)
        route=request.path
        permitted=(request.method=='GET' and (route in {'/observe','/files','/health'}
                   or route.startswith('/actions/'))
                   or request.method=='POST' and route in {'/act','/control'})
        if not permitted:
            return web.json_response({'error':'Unsupported device route'},status=404)
        raw=await request.read()
        selected=False
        if route=='/act':
            try:body=json.loads(raw)
            except ValueError:return web.json_response({'error':'Malformed action'},status=400)
            if not isinstance(body,dict):
                return web.json_response({'error':'Malformed action'},status=400)
            action_id=body.get('action_id')
            if not isinstance(action_id,str):
                return web.json_response({'error':'Missing action identity'},status=400)
            self.dispatches[action_id]=self.dispatches.get(action_id,0)+1
            if self.selected is None and body.get('action',{}).get('kind')==self.kind:
                self.selected=action_id
                selected=True
                self.record('selected_input',action_id=action_id)
        if route.startswith('/actions/'):
            self.receipt_queries.append(route.split('/')[-1])
        async with self.session.request(request.method,self.endpoint+route,data=raw,
                                        headers={'Content-Type':'application/json'}) as response:
            data=await response.read()
            status=response.status
        if selected:
            try:receipt=json.loads(data)
            except ValueError:receipt={}
            if (status==200 and receipt.get('status')=='completed'
                    and receipt.get('action_id')==self.selected):
                self.record('completed_receipt_withheld',action_id=self.selected)
                self.delivered.set()
                await self.release.wait()
            else:
                self.record('fault_not_injected',action_id=self.selected,http_status=status)
        return web.Response(status=status,body=data,content_type='application/json')

    async def start(self):
        self.session=aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=30),
                         headers={'Authorization':'Bearer '+self.token})
        app=web.Application(client_max_size=128*1024)
        app.router.add_route('*','/{path:.*}',self.forward)
        self.runner=web.AppRunner(app,handler_cancellation=True)
        await self.runner.setup()
        site=web.TCPSite(self.runner,'127.0.0.1',0)
        await site.start()
        self.origin='http://127.0.0.1:'+str(site._server.sockets[0].getsockname()[1])
        return self.origin

    def evidence(self):
        return dict(kind='one_completed_receipt_withheld',selected_action_id=self.selected,
                    dispatch_counts=dict(self.dispatches),receipt_queries=list(self.receipt_queries),
                    events=list(self.events),input_replay=False,credential_exported=False)

    async def close(self):
        self.release.set()
        if self.runner:await self.runner.cleanup()
        if self.session:await self.session.close()
