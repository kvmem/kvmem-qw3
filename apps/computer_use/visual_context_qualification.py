"""Real pinned Codex/MCP transport qualification with synthetic desktop/backend."""
import argparse
import asyncio
import base64
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import time

from aiohttp import web
from PIL import Image


async def serve(app):
    runner=web.AppRunner(app,handler_cancellation=True);await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
    return runner,'http://127.0.0.1:'+str(site._server.sockets[0].getsockname()[1])


async def qualify(candidate, output):
    candidate,output=Path(candidate).resolve(),Path(output).resolve()
    build=json.loads((candidate/'build.json').read_text())
    package=candidate/'framework/qw3_computer'
    if build.get('kind')!='isolated_visual_working_context_candidate':raise ValueError('Declare the image-context candidate')
    for name,digest in build['candidate_files_sha256'].items():
        if hashlib.sha256((package/name).read_bytes()).hexdigest()!=digest:raise ValueError('Candidate source changed: '+name)
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(package.parent))
    app=importlib.import_module('qw3_computer.app')
    desktop_module=importlib.import_module('qw3_computer.desktop')
    codec=importlib.import_module('qw3_computer.request_archive')
    results=[]
    for platform in ('windows','linux','android'):
        checks=[];calls=[];timeline=[]
        def require(condition,name):
            checks.append({'name':name,'passed':bool(condition)})
            if not condition:raise AssertionError(name)

        class SyntheticDesktop(desktop_module.Desktop):
            def __init__(self,runtime):
                super().__init__('http://127.0.0.1:9','synthetic-unused',runtime)
                self.sequence=0
            async def start(self):return await self.observe()
            async def observe(self):
                n=self.sequence;self.sequence+=1
                stream=io.BytesIO();Image.new('RGB',(128,96),(n % 256,70,90)).save(stream,format='PNG')
                pixels=stream.getvalue()
                obs={'observation_id':f'obs{n}','desktop_id':'synthetic-desktop-'+platform,
                     'device_id':'synthetic-'+platform,'session_id':'synthetic-session','display_id':'primary',
                     'width':128,'height':96,'sha256':hashlib.sha256(pixels).hexdigest(),
                     'image':base64.b64encode(pixels).decode(),'captured_at':n,
                     'foreground':{'handle':'window1','process':'synthetic-app','title':f'View {n}'},
                     'capabilities':{'platform':platform},'accessibility':{'status':'ok','truncated':False,'controls':[]}}
                return self.remember(obs)|{'image':obs['image']}
            async def control(self,action,acknowledge_uncertain=False):
                if action not in ('pause','resume'):raise AssertionError('Unexpected synthetic control')
                self.paused=action=='pause';return {'paused':self.paused}
            async def request(self,method,route,body=None):
                if method=='GET' and route=='/files':return {'files':[]}
                raise AssertionError('No native request is allowed')

        operations=[('observe',{}),('observe',{}),('history',{'operation':'pin','observation_id':'obs2'}),
            ('observe',{}),('observe',{}),('observe',{}),
            ('history',{'operation':'read','observation_id':'obs2'}),('history',{'operation':'list'}),
            ('history',{'operation':'unpin','observation_id':'obs2'}),('observe',{})]
        async def backend(request):
            body=await request.json();calls.append(body)
            response=web.StreamResponse(headers={'Content-Type':'text/event-stream'});await response.prepare(request)
            if len(calls)<=len(operations):
                name,args=operations[len(calls)-1]
                actual=next(t['function']['name'] for t in body['tools'] if t['function']['name']=='mcp__computer__'+name)
                delta={'tool_calls':[{'index':0,'id':'synthetic-call-'+str(len(calls)),'type':'function',
                        'function':{'name':actual,'arguments':json.dumps(args)}}]};finish='tool_calls'
            else:delta={'content':'Transport qualification finished; no business task has been completed.'};finish='stop'
            await response.write(('data: '+json.dumps({'choices':[{'index':0,'delta':delta,'finish_reason':None}]})+'\n\n').encode())
            await response.write(('data: '+json.dumps({'choices':[{'index':0,'delta':{},'finish_reason':finish}]})+'\n\n').encode())
            await response.write(b'data: [DONE]\n\n');return response

        backend_app=web.Application();backend_app.router.add_post('/v1/chat/completions',backend)
        backend_runner,url=await serve(backend_app)
        runtime=output/platform
        desktop=SyntheticDesktop(runtime/'device')
        controller=app.Controller(runtime/'app',desktop,url+'/v1',0,workflow=True,device_platform=platform,
                                  thinking=True,model_output_tokens=16384,image_history='recent-observations',tool_call_limit=12)
        original_tool=controller.tool
        async def tool(name,args):
            before=desktop.latest['observation_id']
            result=await original_tool(name,args)
            timeline.append({'name':name,'arguments':args,'latest_before':before,
                             'latest_after':desktop.latest['observation_id'],'calls_used':controller.state['tool_calls_used']})
            return result
        controller.tool=tool
        app_runner,app_url=await serve(app.web_app(controller));controller.port=int(app_url.rsplit(':',1)[1])
        try:
            await controller.start()
            await controller.start_task('Exercise own archived observations without sending native input. This is transport qualification only.')
            async with asyncio.timeout(40):
                while controller.state['status'] in ('starting','running','verifying'):await asyncio.sleep(.02)
            require(len(calls)==11,'all eleven requests traversed actual Codex and private bridge')
            require(len(timeline)==10 and controller.state['tool_calls_used']==10,'all ten real MCP calls share cumulative budget')
            require(controller.state['image_history']=='recent-observations','policy persisted in actual task checkpoint')
            require(controller.state['status']=='paused','unchecked synthetic task remains pending independent business review')
            require(all(c['enable_thinking'] and c['max_tokens']==16384 for c in calls),'thinking and output budget preserved')
            schema=next(t['function']['parameters'] for t in calls[0]['tools'] if t['function']['name']=='mcp__computer__history')
            require(schema.get('properties',{}).get('operation',{}).get('enum')==['list','read','pin','unpin'],
                    'history schema survives actual Codex normalization')
            history_events=[e for e in timeline if e['name']=='history']
            require(len(history_events)==4 and all(e['latest_before']==e['latest_after'] for e in history_events),
                    'pin read list unpin never promote latest observation')
            require(timeline[-1]['latest_after']=='obs7' and desktop.latest['observation_id']=='obs8',
                    'only live observes and final live collection advance current observation')
            require(desktop.db.execute('SELECT COUNT(*) FROM actions').fetchone()[0]==0,'no device actions entered input ledger')
            traces=[]
            for trace in sorted((runtime/'app/inference').iterdir()):
                if not trace.is_dir() or not (trace/'request.json').is_file():continue
                report=json.loads((trace/'image-context.json').read_text())
                meta=json.loads((trace/'request.full.archive.json').read_text())
                source=codec.decode(meta['tree'],trace.parent/'full-history-blobs')
                source_bytes=json.dumps(source,ensure_ascii=False).encode()
                actual_bytes=(trace/'request.json').read_bytes();actual=json.loads(actual_bytes)
                require(hashlib.sha256(source_bytes).hexdigest()==meta['sha256'] and len(source_bytes)==meta['bytes'],
                        'full converted source roundtrip '+trace.name)
                require(hashlib.sha256(actual_bytes).hexdigest()==report['presented_sha256'],
                        'dispatched trace identity '+trace.name)
                require(actual in calls,'recorded presented input actually reached backend '+trace.name)
                traces.append({'trace':trace.name,'report':report})
            require(len(traces)==11,'all eleven presented and full histories retained')
            require(max(t['report']['dropped_images'] for t in traces)>=3,'old synthetic images are replaced in actual wire requests')
            require(any(p['observation_id']=='obs2' for t in traces for p in t['report']['selected']),
                    'explicit pinned reference survives the actual request view')
            require(all(not t['report']['missing_pins'] for t in traces),'no pinned reference silently lost in this qualification')
            require(not controller.history.reference_pins(),'unpin persists through the actual MCP boundary')
            rows=desktop.db.execute('SELECT COUNT(*) FROM observations WHERE task=?',(controller.state['id'],)).fetchone()[0]
            require(rows==8,'archived read does not create a synthetic live capture')
            results.append({'platform_tool_catalogue':platform,'passed':True,'checks':checks,'timeline':timeline,
                            'traces':traces,'model_requests':len(calls),'real_qw3_model_calls':0,
                            'native_device_tested':False,'native_inputs':0,'business_success_claimed':False})
        finally:
            try:
                await controller.close()
            finally:
                try:
                    if controller.client and controller.client.proc and controller.client.proc.returncode is None:
                        await controller.client.close()
                finally:
                    await app_runner.cleanup();await backend_runner.cleanup()
    result={'kind':'actual_codex_mcp_image_context_transport_qualification','at':time.time(),'passed':True,
            'candidate_build_sha256':hashlib.sha256((candidate/'build.json').read_bytes()).hexdigest(),
            'recipe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'platforms':results,'qw3_model_calls':0,'native_device_inputs':0,'native_windows_qualified':False,
            'business_success_improvement_proven':False,'advisor_enabled':False}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:result=asyncio.run(qualify(args.candidate,args.output))
    except Exception as exc:
        if args.output.is_dir():
            (args.output/'result.json').write_text(json.dumps({'passed':False,'error_type':type(exc).__name__,
                'error':str(exc),'qw3_model_calls':0,'native_device_inputs':0,'business_success_claimed':False},indent=2)+'\n')
        raise
    print(json.dumps({'passed':result['passed'],'checks':sum(len(p['checks']) for p in result['platforms']),
                      'qw3_model_calls':0,'native_device_inputs':0}))


if __name__=='__main__':main()
