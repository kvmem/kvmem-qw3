"""Independent fixture qualification, never a real-model success result."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import time
import urllib.request

from browser_pilot import initialize, grade, stop
from platform_pilots import android_command, linux_execute, owned_container
from osworld_linux.runner import save
from osworld_windows.source import ROOT


def observe(platform, directory, label):
    gateway = 18130 if platform == 'linux' else 18204
    runtime = ROOT/'apps/computer_use/.runtime'/platform/'gateway-03'
    request = urllib.request.Request(f'http://127.0.0.1:{gateway}/observe',
                                    headers={'Authorization':'Bearer '+(runtime/'token.txt').read_text().strip()})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=20) as response:
        result=json.load(response)
    (directory/(label+'.png')).write_bytes(base64.b64decode(result.pop('image')))
    save(directory/(label+'.json'),result)
    return result


def linux_input_control(directory, name, kind, after=''):
    observation=observe('linux',directory,'before-'+name.replace(' ','-'))
    matched=[c for c in observation['accessibility']['controls'] if c['name']==name and c['kind']==kind]
    if len(matched)!=1:
        raise ValueError(f'Expected one visible {kind}: {name}')
    target=matched[0]
    code=('import pyautogui;pyautogui.FAILSAFE=False;'
          f'pyautogui.click({target["center_x"]},{target["center_y"]});'+after)
    linux_execute(code)


def qualify_linux(fixture,directory):
    # Administrative qualification uses the actual guest GUI and freshly read
    # AT-SPI targets. This transport is never included in model tool permissions.
    fields=fixture['fields']
    linux_execute('import pyautogui,time;pyautogui.hotkey("ctrl","l");'
                  f'pyautogui.write({fixture["url"]!r},interval=.01);pyautogui.press("enter");time.sleep(2)')
    for label, field in [('Full name','name'),('Email address','email'),('Details','notes')]:
        linux_input_control(directory,label,'entry',
                            'pyautogui.hotkey("ctrl","a");'+f'pyautogui.write({fields[field]!r},interval=.005)')
    linux_input_control(directory,'Topic','combo-box','pyautogui.press("home");pyautogui.press("down");pyautogui.press("enter")')
    linux_input_control(directory,'Preferred delivery','combo-box','pyautogui.press("home");pyautogui.press("down");pyautogui.press("enter")')
    linux_input_control(directory,'I confirm these are test details','check-box')
    linux_input_control(directory,'Review request','push-button')
    time.sleep(.5)
    preview=observe('linux',directory,'review')
    before_save=grade(fixture,linux_execute)
    linux_input_control(directory,'Save request','push-button')
    time.sleep(.5)
    final=observe('linux',directory,'saved')
    return {'actual_guest_browser_gui':True,'preview_did_not_save':before_save['submission_count']==0,
            'preview_visible':any(c['name']=='Review before saving' for c in preview['accessibility']['controls']),
            'saved_confirmation_visible':any(c['name']=='Request saved' for c in final['accessibility']['controls'])}


def qualify_android(fixture,directory):
    android_command(['shell','am','start','-a','android.intent.action.VIEW','-d',fixture['url'],'-p','com.android.chrome'],capture_output=True)
    time.sleep(2)
    screen=observe('android',directory,'opened')
    interventions=[]
    # Set up this isolated test browser through visible first-run controls.
    # Never create an account or submit any real user information.
    for index in range(4):
        if screen['foreground'].get('process')!='com.android.chrome':
            break
        controls=screen.get('accessibility',{}).get('controls',[])
        target_name='Use without an account'
        if any(c['name']=='Chrome notifications make things easier' for c in controls):
            target_name='No thanks'
        matches=[c for c in controls if c['name']==target_name]
        if len(matches)!=1:
            break
        control=matches[0]
        android_command(['shell','input','tap',str(control['center_x']),str(control['center_y'])],capture_output=True)
        interventions.append({'name':control['name'],'observation_id':screen['observation_id'],'initialization_only':True})
        time.sleep(2)
        screen=observe('android',directory,'initialized-'+str(index))
    deadline=time.monotonic()+20
    while not any(c['name']=='Local support request' for c in screen.get('accessibility',{}).get('controls',[])) and time.monotonic()<deadline:
        time.sleep(.5)
        screen=observe('android',directory,'browser-ready')
    # Android fixture grade qualification checks ordinary POST storage in the
    # local container; full UI interaction remains a separate model pilot.
    code=('const http=require("http");const body=new URLSearchParams('+json.dumps(fixture['fields'])+').toString();'
          'function send(path){return new Promise((resolve,reject)=>{const req=http.request({host:"127.0.0.1",port:'+str(fixture['port'])+
          ',path,method:"POST",headers:{"Content-Type":"application/x-www-form-urlencoded"}},res=>{let b="";res.on("data",c=>b+=c);res.on("end",()=>resolve({status:res.statusCode,body:b}));});req.on("error",reject);req.end(body);});}'
          '(async()=>{let review=await send("/review");if(review.status!==200||!review.body.includes("Review before saving"))throw Error("Preview failed");'
          'const fs=require("fs");if(fs.existsSync('+json.dumps(fixture['guest_directory']+'/submissions.jsonl')+'))throw Error("Preview wrote submission");'
          'let final=await send("/submit");if(final.status!==200||!final.body.includes("Request saved"))throw Error("Save failed");'
          'process.stdout.write(JSON.stringify({preview_did_not_save:true,normal_post_storage:true}));})().catch(e=>{process.stderr.write(String(e));process.exit(1);});')
    reply=subprocess.run(['docker','exec','qw3-computer-android','/node/bin/node','-e',code],capture_output=True,text=True,check=True,timeout=15)
    result=json.loads(reply.stdout)
    result.update(actual_guest_browser_gui=False,
                  initialization_interventions=interventions,
                  form_initial_screen_verified=any(c['name']=='Local support request' for c in screen.get('accessibility',{}).get('controls',[])),
                  browser_visible_process=screen['foreground'].get('process'),
                  page_visible_controls=[c['name'] for c in screen.get('accessibility',{}).get('controls',[])],
                  scope='Container fixture storage and captured Android browser initial screen; not full Android GUI completion')
    return result


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    owned_container('qw3-computer-'+args.platform,'qw3.goal','cross-device-20261003')
    result={'kind':'qualification-not-model-results','model_used':False,'platform':args.platform,'passed':False}
    fixture=None
    try:
        fixture,_=initialize(args.platform,args.output.name,args.output,linux_execute)
        result['negative']=grade(fixture,linux_execute)
        if args.platform=='linux':
            linux_execute('import subprocess,time;subprocess.Popen('+repr(['google-chrome','--new-window',fixture['url']])+
                          ',stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);time.sleep(2)')
            result.update(qualify_linux(fixture,args.output))
        else:
            result.update(qualify_android(fixture,args.output))
        result['positive']=grade(fixture,linux_execute)
        result['storage_qualified']=not result['negative']['passed'] and result['positive']['passed'] and result['preview_did_not_save']
        result['passed']=result['storage_qualified']
        if args.platform=='linux':
            result['passed']=result['passed'] and result['preview_visible'] and result['saved_confirmation_visible']
        else:
            result['passed']=result['passed'] and result['form_initial_screen_verified']
    except Exception as exc:
        result.update(passed=False,error=type(exc).__name__+': '+str(exc))
    finally:
        if fixture:
            try:result['service_stop']=stop(fixture,linux_execute)
            except Exception as exc:result.update(passed=False,service_stop_error=type(exc).__name__+': '+str(exc))
        save(args.output/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform',choices=['linux','android'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    result=run(parser.parse_args())
    print({k:result.get(k) for k in ['passed','platform','actual_guest_browser_gui','error','service_stop_error']})
    raise SystemExit(0 if result['passed'] else 1)
