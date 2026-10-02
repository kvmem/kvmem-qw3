"""Administrative Edge render probe; block every nonlocal page request.

This is not a model trajectory. Only browser navigation, screenshots, geometry
reads and ordinary clicks are used; no application state is injected via CDP.
"""
import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import aiohttp


HOSTS = {name+'.qw3.test:18090' for name in ('formcraft', 'teamchat', 'cloudcrm', 'mailhub', 'calendar')}


async def run(plan, output):
    if os.name != 'nt' or os.environ.get('USERNAME') != 'Qw3Test':
        raise ValueError('Dedicated Windows qualification only')
    for case in plan:
        url = urlparse(case['url'])
        if url.scheme != 'http' or url.netloc not in HOSTS:
            raise ValueError('Unexpected qualification page origin')
    output.mkdir(exist_ok=False)
    report = dict(kind='windows_offline_private_site_rendering', passed=False, formal_run=False,
                  model_used=False, pages=[], requests=[], blocked_external_requests=[], console_exceptions=[],
                  interception_errors=[])
    token = Path(r'C:\Qw3Computer\executor-token.txt').read_text().strip()
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=15)) as session:
        async with session.get('http://127.0.0.1:5000/health', headers={'Authorization': 'Bearer '+token}) as response:
            response.raise_for_status()
            if not (await response.json())['paused']:
                raise ValueError('Desktop must be paused')
        async with session.get('http://127.0.0.1:19222/json/list') as response:
            response.raise_for_status()
            pages = await response.json()
        targets = [p for p in pages if p.get('type') == 'page' and p.get('url') == 'about:blank']
        if len(targets) != 1:
            raise ValueError('Need one initially blank qualification tab')
        async with session.ws_connect(targets[0]['webSocketDebuggerUrl']) as ws:
            pending, handlers, sequence = {}, set(), 0

            async def command(method, params=None):
                nonlocal sequence
                sequence += 1
                identity = sequence
                future = asyncio.get_running_loop().create_future()
                pending[identity] = future
                try:
                    await ws.send_json(dict(id=identity, method=method, params=params or {}))
                    async with asyncio.timeout(20):
                        value = await future
                    if value.get('error') or value.get('result', {}).get('exceptionDetails'):
                        raise ValueError('Browser command failed: '+method)
                    return value['result']
                finally:
                    pending.pop(identity, None)

            async def intercept(params):
                url = urlparse(params['request']['url'])
                local = (url.scheme == 'http' and url.netloc in HOSTS) or url.scheme in ('data', 'blob', 'about')
                row = dict(url=params['request']['url'], resource_type=params['resourceType'], allowed=local)
                report['requests'].append(row)
                if not local:
                    report['blocked_external_requests'].append(row)
                await command('Fetch.continueRequest' if local else 'Fetch.failRequest',
                              dict(requestId=params['requestId']) if local else
                              dict(requestId=params['requestId'], errorReason='BlockedByClient'))

            def intercepted(task):
                handlers.discard(task)
                if not task.cancelled() and task.exception() is not None:
                    report['interception_errors'].append(repr(task.exception()))

            async def read():
                async for message in ws:
                    value = json.loads(message.data)
                    if 'id' in value:
                        future = pending.get(value['id'])
                        if future is not None and not future.done():
                            future.set_result(value)
                    elif value.get('method') == 'Fetch.requestPaused':
                        task = asyncio.create_task(intercept(value['params']))
                        handlers.add(task)
                        task.add_done_callback(intercepted)
                    elif value.get('method') == 'Runtime.exceptionThrown':
                        report['console_exceptions'].append(value['params'])

            reader = asyncio.create_task(read())
            async def evaluate(expression):
                value = await command('Runtime.evaluate', dict(expression=expression, returnByValue=True, awaitPromise=True))
                return value['result'].get('value')

            async def snapshot(name):
                await evaluate('document.fonts.ready.then(() => true)')
                async with asyncio.timeout(15):
                    while True:
                        data = await evaluate('''JSON.stringify({title:document.title,bodyText:document.body.innerText,
                          fonts:document.fonts.status,images:[...document.images].map(i=>({src:i.currentSrc||i.src,
                          naturalWidth:i.naturalWidth,complete:i.complete,alt:i.alt}))})''')
                        data = json.loads(data)
                        if data['bodyText'].strip() and all(i['complete'] for i in data['images']):
                            break
                        await asyncio.sleep(.1)
                image = await command('Page.captureScreenshot', dict(format='png'))
                (output / (name+'.png')).write_bytes(base64.b64decode(image['data']))
                data['all_images_decoded'] = all(i['naturalWidth'] > 0 for i in data['images'])
                (output / (name+'.json')).write_text(json.dumps(data, indent=2))
                return data

            async def click(selector, scroll=False):
                expression = '''(() => {for(const e of document.querySelectorAll(SELECTOR)) {
                  const r=e.getBoundingClientRect();const x=r.x+r.width/2,y=r.y+r.height/2;
                  if(r.width>0&&r.height>0&&x>=0&&x<innerWidth&&y>=0&&y<innerHeight)
                    return {x,y,text:e.innerText};}return null;})()'''.replace('SELECTOR', json.dumps(selector))
                async with asyncio.timeout(12):
                    while True:
                        target = await evaluate(expression)
                        if target:
                            break
                        if scroll:
                            geometry = await evaluate('''(() => {const e=document.querySelector(SELECTOR);
                              if(!e)return null;const r=e.getBoundingClientRect();
                              return {x:Math.min(innerWidth-20,Math.max(20,r.x+r.width/2)),
                                y:innerHeight/2,deltaY:Math.max(-400,Math.min(400,r.y-innerHeight/2))};})()'''
                              .replace('SELECTOR', json.dumps(selector)))
                            if geometry:
                                await command('Input.dispatchMouseEvent', dict(type='mouseWheel', deltaX=0, **geometry))
                        await asyncio.sleep(.1)
                for kind in ('mousePressed', 'mouseReleased'):
                    await command('Input.dispatchMouseEvent', dict(type=kind, button='left', clickCount=1,
                                                                  x=target['x'], y=target['y']))
                await asyncio.sleep(.4)
                return target

            try:
                await command('Page.enable')
                await command('Runtime.enable')
                await command('Network.enable')
                await command('Network.setCacheDisabled', dict(cacheDisabled=True))
                # Installed while the only page is blank: no external request
                # from the tested sites can escape before interception begins.
                await command('Fetch.enable', dict(patterns=[dict(urlPattern='*', requestStage='Request')]))
                for case in plan:
                    begin = len(report['requests'])
                    await command('Page.navigate', dict(url=case['url']))
                    async with asyncio.timeout(20):
                        while True:
                            try:
                                ready = await evaluate('document.readyState === "complete" && location.host === '
                                    + json.dumps(urlparse(case['url']).netloc)
                                    + ' && document.cookie.split("; ").includes('
                                    + json.dumps('user_id='+case['identity']) + ')')
                                if ready:
                                    break
                            except ValueError:
                                pass
                            await asyncio.sleep(.2)
                    await asyncio.sleep(2)
                    data = await snapshot(case['name'])
                    row = dict(name=case['name'], app=case['app'], image_count=len(data['images']),
                               all_images_decoded=data['all_images_decoded'], fonts_loaded=data['fonts']=='loaded',
                               body_text_length=len(data['bodyText']),
                               expected_page_text_visible=case['expected_text'] in data['bodyText'],
                               missing_channel='Channel not found' in data['bodyText'])
                    if case['app'] == 'calendar':
                        row['event_click'] = await click('[draggable="true"]')
                        await click('button[title="Edit"]')
                        modal = await snapshot(case['name']+'-event-modal')
                        row['modal_images_decoded'] = modal['all_images_decoded']
                        row['local_dynamic_avatars'] = sum(i['src'].startswith('data:image/svg+xml') for i in modal['images'])
                        await click('input[placeholder="Add guests"]', scroll=True)
                        picker = await snapshot(case['name']+'-guest-picker')
                        row['picker_images_decoded'] = picker['all_images_decoded']
                        row['picker_image_count'] = len(picker['images'])
                    row['network_requests'] = len(report['requests'])-begin
                    row['passed'] = row['all_images_decoded'] and row['fonts_loaded'] and row['body_text_length'] > 100 and row['expected_page_text_visible'] and not row['missing_channel']
                    if case['app'] == 'calendar':
                        row['passed'] &= row['modal_images_decoded'] and row['picker_images_decoded'] and row['local_dynamic_avatars'] > 0
                    report['pages'].append(row)
                    (output / 'progress.json').write_text(json.dumps(report, indent=2))
                report['passed'] = all(p['passed'] for p in report['pages']) and not report['blocked_external_requests'] and not report['console_exceptions'] and not report['interception_errors']
            except BaseException as exc:
                report['error'] = repr(exc)
                raise
            finally:
                try:
                    # Keep interception active until the tested application is
                    # gone, so removing it cannot release a delayed remote retry.
                    await command('Page.navigate', dict(url='about:blank'))
                    async with asyncio.timeout(10):
                        while not await evaluate('location.href === "about:blank" && document.readyState === "complete"'):
                            await asyncio.sleep(.1)
                    if handlers:
                        await asyncio.gather(*handlers, return_exceptions=True)
                    await command('Fetch.disable')
                    if report['interception_errors']:
                        report['passed'] = False
                finally:
                    reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)
                    (output / 'result.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run(json.loads(args.plan.read_text(encoding='utf-8-sig')), args.output))
    raise SystemExit(0 if result['passed'] else 1)
