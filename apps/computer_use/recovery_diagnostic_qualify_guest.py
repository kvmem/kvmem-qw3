"""Scripted P3 fixture qualification, never credited to the local model."""
import argparse
import asyncio
import json
from pathlib import Path
import time

from qw3_computer.desktop import Desktop, DesktopError, DesktopRejected


async def run(args):
    token=Path('C:/Qw3Computer/executor-token.txt').read_text().strip()
    out=args.directory/'qualification';out.mkdir()
    desktop=Desktop('http://127.0.0.1:5001',token,out/'desktop')
    result=dict(kind='scripted_real_windows_fault_qualification',model_used=False,condition=args.condition,passed=False)
    counter=0
    async def act(action):
        nonlocal counter
        counter+=1
        return await desktop.act(dict(desktop_id=desktop.latest['desktop_id'],observation_id=desktop.latest['observation_id'],
            action_id=args.directory.name+'-'+str(counter),action=action))
    async def find(name,kind):
        deadline=time.monotonic()+35
        while time.monotonic()<deadline:
            hits=[v for v in desktop.latest.get('accessibility',{}).get('controls',[]) if v['name'].strip()==name and v['kind']==kind]
            if len(hits)==1:return hits[0]
            await asyncio.sleep(.5);await desktop.observe()
        raise RuntimeError('Control missing or ambiguous: '+name)
    async def click(name,kind):
        c=await find(name,kind)
        return await act(dict(kind='click',x=c['center_x'],y=c['center_y'],button='left',clicks=1))
    try:
        await desktop.start()
        result['start']=await desktop.request('POST','/fixture/start',{})
        await desktop.control('resume');await desktop.observe()
        result['initial']=desktop.latest
        front=desktop.latest['foreground']
        if args.condition=='wrong_focus':
            assert front['process'].casefold()=='notepad' and 'p3-sentinel' in front['title']
        if args.condition=='outside_scope':
            assert front['process'].casefold()=='charmap' and 'Character Map' in front['title']
            try:await act(dict(kind='key',key='CTRL+L'))
            except DesktopRejected:result['scope_rejection_observed']=True
            else:raise AssertionError('Outside-scope operation was not rejected')
            await desktop.observe()
        if args.condition in ('wrong_focus','outside_scope','wrong_page'):
            await act(dict(kind='open_app',app='edge'))
        for name,text in (('Name','Aster'),('Reference','R-204')):
            await click(name,'Edit');await act(dict(kind='type',text=text))
        try:
            await click('Save request','Button')
            if args.condition=='save_confirmation':
                await act(dict(kind='key',key='ENTER'))
        except DesktopError:
            if args.condition!='uncertain_delivery':raise
            assert desktop.paused and len(desktop.unresolved())==1
            result['uncertain_before_reconciliation']=desktop.unresolved()
            await desktop.control('resume')
            assert not desktop.unresolved()
            result['receipt_reconciled_without_repeat']=True
        await asyncio.sleep(.5)
        state=json.loads((args.directory/'state.json').read_text(encoding='utf-8'))
        assert state['saved_count']==1 and state['saved']=={'name':'Aster','reference':'R-204'}
        events={e['event'] for e in state['events']}
        if args.condition=='save_confirmation':assert {'confirmation_shown','confirmation_accepted'}<=events
        if args.condition=='uncertain_delivery':assert state['withheld'] and result.get('receipt_reconciled_without_repeat')
        if args.condition in ('normal_slow_loading','delayed_update'):
            ready=[e for e in state['events'] if e['event']=='form_ready']
            duration=(ready[0]['monotonic_ns']-state['started_monotonic_ns'])/1e9
            assert (12 if args.condition=='normal_slow_loading' else 20)<=duration<25
            result['actual_loading_seconds']=duration
        result.update(passed=True,actual_actions=counter)
    except Exception as exc:result['error']=f'{type(exc).__name__}: {exc}'
    finally:
        await desktop.control('pause');await desktop.close()
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--condition',required=True)
    result=asyncio.run(run(parser.parse_args()))
    raise SystemExit(0 if result['passed'] else 1)
