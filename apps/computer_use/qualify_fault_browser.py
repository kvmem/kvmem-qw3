"""Actual host-browser fixture qualification; never a VM or model result."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from playwright.async_api import async_playwright
from fault_browser_pilot import FIELDS,SERVER
from browser_pilot import grade_records


async def case(browser,directory,condition,viewport,transport='node'):
    directory.mkdir(exist_ok=False)
    source=SERVER.read_text().replace('__QW3_FAULT_CONDITION__',json.dumps(condition))
    (directory/'server.js').write_text(source)
    with socket.socket() as endpoint:
        endpoint.bind(('127.0.0.1',0));port=endpoint.getsockname()[1]
    stderr=(directory/'server.log').open('wb')
    if transport=='stdlib':
        from fault_browser_windows_guest import page
        (directory/'page.html').write_text(page(condition,SERVER.read_text()),encoding='utf-8')
        command=[sys.executable,str(Path(__file__).with_name('fault_browser_windows_guest.py')),
                 '--directory',str(directory),'--condition',condition,'--page',str(directory/'page.html'),'--port',str(port)]
    else:command=[shutil.which('node'),str(directory/'server.js'),str(directory),str(port)]
    process=subprocess.Popen(command,stdout=stderr,stderr=stderr)
    context=None
    result={'condition':condition,'viewport':viewport,'transport':transport,'model_used':False,'vm_used':False,
            'scripted_host_browser_only':True,'started_at':time.time(),
            'fixture_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'checks':{}}
    try:
        deadline=time.monotonic()+5
        while not (directory/'server-ready.json').exists():
            if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Qualification server failed')
            await asyncio.sleep(.05)
        (directory/'arm.json').write_text(json.dumps({'armed_at_ms':int(time.time()*1000)}))
        armed=time.monotonic()
        context=await browser.new_context(viewport=viewport)
        page=await context.new_page()
        # Recovery pages intentionally poll for readiness. A network-idle
        # heuristic is not their completion condition; verify visible state.
        await page.goto('http://127.0.0.1:'+str(port)+'/',wait_until='domcontentloaded')
        if condition=='slow_loading':
            await page.locator('#status').filter(has_text='Loading request form').wait_for()
            result['checks']['loading_visible']=await page.locator('#form').is_hidden()
            await page.screenshot(path=str(directory/'loading.png'))
            await page.locator('#form').wait_for(state='visible',timeout=120000)
            result['loading_seconds']=time.monotonic()-armed
            result['checks']['actual_90_second_loading']=result['loading_seconds']>=89
        await page.locator('input[name=name]').click()
        if condition=='focus_dialog':
            await page.locator('#notice').wait_for(state='visible')
            result['checks']['modal_takes_focus']=await page.evaluate("document.querySelector('#notice').contains(document.activeElement)")
            await page.screenshot(path=str(directory/'modal.png'))
            await page.locator('#continue').click()
            await page.locator('#notice').wait_for(state='hidden')
        await page.locator('input[name=name]').fill(FIELDS['name'])
        await page.locator('input[name=reference]').fill(FIELDS['reference'])
        await page.locator('#save').click()
        expected='Saving…' if condition=='no_progress' else 'Saved'
        await page.wait_for_function('(value)=>document.querySelector("#status").textContent===value',arg=expected)
        await asyncio.sleep(.5)
        state=json.loads((directory/'fault-state.json').read_text())
        names=[row['name'] for row in state['events']]
        if condition=='no_progress':
            result['checks']['pending_without_saved_record']=state['saved_count']==0 and not (directory/'submissions.jsonl').exists()
            result['checks']['save_disabled']=await page.locator('#save').is_disabled()
            result['checks']['pending_visible_event']='saving_pending_visible' in names
        else:
            result['checks']['exactly_one_correct_saved_request']=grade_records((directory/'submissions.jsonl').read_text(),FIELDS)['passed']
            result['checks']['saved_visible_event']='saved_visible' in names
        if condition=='focus_dialog':result['checks']['notice_shown_and_closed']='notice_shown' in names and 'notice_closed' in names
        if condition=='slow_loading':result['checks']['loading_then_ready']='loading_visible' in names and 'form_ready' in names
        result['checks']['protected_note_preserved']=await page.locator('#note').inner_text()=='Existing note: Protected unrelated note.'
        result['checks']['one_save_attempt']=len(state['save_attempts'])==1
        await page.screenshot(path=str(directory/'final.png'))
        result['passed']=all(result['checks'].values())
    except Exception as exc:result.update(passed=False,error=type(exc).__name__+': '+str(exc))
    finally:
        if context:await context.close()
        if process.poll() is None:process.terminate()
        await asyncio.to_thread(process.wait,timeout=5)
        stderr.close()
        result['finished_at']=time.time()
        (directory/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


async def run(args):
    if args.transport=='node' and not shutil.which('node'):raise ValueError('Use the existing Node runtime')
    args.output.mkdir(parents=True,exist_ok=False)
    async with async_playwright() as playwright:
        browser=await playwright.chromium.launch(executable_path=str(args.chrome),headless=True,
            args=['--no-sandbox','--disable-gpu','--no-proxy-server','--disable-background-networking'])
        try:
            results=await asyncio.gather(*[case(browser,args.output/(name+'-'+condition),condition,viewport,args.transport)
                for name,viewport in [('desktop',{'width':1360,'height':900}),('mobile',{'width':393,'height':852})]
                for condition in ['normal','slow_loading','focus_dialog','no_progress']])
        finally:await browser.close()
    summary={'kind':'scripted_host_browser_fault_fixture_qualification','model_used':False,'vm_used':False,
        'native_device_qualification':False,'actual_clock_slow_loading':True,'passed':all(row['passed'] for row in results),
        'checks':sum(len(row['checks']) for row in results),'cases':results,
        'transport':args.transport,'template_sha256':hashlib.sha256(SERVER.read_bytes()).hexdigest(),
        'stdlib_server_sha256':hashlib.sha256(Path(__file__).with_name('fault_browser_windows_guest.py').read_bytes()).hexdigest() if args.transport=='stdlib' else None,
        'qualification_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'finished_at':time.time()}
    (args.output/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print({key:summary[key] for key in ['passed','checks','model_used','vm_used']},flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chrome',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--transport',choices=['node','stdlib'],default='node')
    args=parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args))['passed'] else 1)
