"""Exercise the real device selector without starting model tasks or inputs."""
import argparse
from pathlib import Path
import time
from urllib.parse import urlparse

from osworld_linux.runner import request, save


def run(args):
    from playwright.sync_api import sync_playwright
    parsed=urlparse(args.app)
    if parsed.hostname!='127.0.0.1' or parsed.scheme!='http' or not parsed.port:
        raise ValueError('Use the explicit local application endpoint')
    initial=request(args.app,'/api/state')
    if initial['status'] not in {'idle','stopped','completed','failed'} or initial['uncertain_actions']:
        raise ValueError('Device selection requires a terminal task without uncertain input')
    known={item['id'] for item in initial['devices']['devices']}
    if not set(args.devices)<=known or args.restore not in known:
        raise ValueError('Qualification only selects existing registered devices')
    args.output.mkdir(parents=True,exist_ok=False)
    result={'kind':'real-browser-device-selector-qualification','model_used':False,
            'passed':False,'checks':[],'page_errors':[],'initial_device':initial['devices']['selected']}
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            try:
                page=browser.new_page(viewport={'width':1280,'height':1000})
                page.on('pageerror',lambda error:result['page_errors'].append(str(error)))
                page.goto(args.app,wait_until='networkidle')
                page.wait_for_function('(n)=>document.querySelector("#device").options.length===n',arg=len(known))
                for identifier in args.devices:
                    page.select_option('#device',identifier)
                    deadline=time.monotonic()+60
                    while time.monotonic()<deadline:
                        state=request(args.app,'/api/state')
                        if state.get('ready') and state['devices']['selected']==identifier:
                            break
                        time.sleep(.5)
                    if not state['ready'] or state['devices']['selected']!=identifier:
                        raise TimeoutError('Device transfer did not become ready: '+identifier)
                    page.wait_for_function('(id)=>document.querySelector("#device").value===id&&!document.querySelector("#device").disabled',arg=identifier)
                    page.screenshot(path=str(args.output/(identifier+'.png')),full_page=True)
                    platform=next(item['platform'] for item in state['devices']['devices'] if item['id']==identifier)
                    result['checks'].append({'device':identifier,'platform':platform,
                        'ready':state['ready'],'task_status':state['status'],'screenshot':identifier+'.png',
                        'selected_identity':state['devices']['selected']})
            finally:
                browser.close()
        result['passed']=len(result['checks'])==len(args.devices) and not result['page_errors']
    except Exception as exc:
        result['error']=type(exc).__name__+': '+str(exc)
    finally:
        try:
            final=request(args.app,'/api/device',{'device_id':args.restore})
            result['final_device']=final['devices']['selected']
        except Exception as exc:
            result.update(passed=False,restore_error=type(exc).__name__+': '+str(exc))
        save(args.output/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app',default='http://127.0.0.1:18205')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--devices',nargs='+',default=['linux-vm','windows-vm','android-emulator'])
    parser.add_argument('--restore',default='android-emulator')
    args=parser.parse_args()
    result=run(args)
    print({'passed':result['passed'],'devices':[x['device'] for x in result['checks']],
           'error':result.get('error'),'restore_error':result.get('restore_error')})
    raise SystemExit(0 if result['passed'] else 1)
