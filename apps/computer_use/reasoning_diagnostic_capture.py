"""Capture public P2 cards in the real Windows VM, without exposing answers."""
import argparse
import asyncio
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import aiohttp
from visual_diagnostic_capture import ACTIVATE
from windows_suite import command

ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'apps/computer_use'
SHARE=APP/'.runtime/windows/shared'


async def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    lock=open('/tmp/qw3-osworld-vm.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    report={'kind':'P2_real_windows_screen_capture','model_used':False,'status':'preparing','started_at':time.time()}
    try:
        info=json.loads(await command('docker','inspect','qw3-computer-windows'))[0]
        if info['Config'].get('Labels',{}).get('com.qw3.computer-use.owner')!='codex-goal-20260930':raise ValueError('Not task-owned VM')
        token=(SHARE/'executor-token.txt').read_text().strip()
        async with aiohttp.ClientSession(trust_env=False,headers={'Authorization':'Bearer '+token}) as client:
            async with client.get('http://127.0.0.1:18082/health') as response:
                response.raise_for_status()
                if not (await response.json())['paused']:raise ValueError('A task is active')
        corpus=json.loads((args.materials/'corpus.json').read_text())
        manifest=json.loads((args.materials/'public/materials.json').read_text())
        stage=SHARE/out.name;shutil.copytree(args.materials/'public',stage)
        cases=[]
        for s in corpus['samples']:
            name=s['html'];source=stage/name
            if source.parent!=stage or hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['files'][name]:raise ValueError('Public HTML changed')
            cases.append({'id':s['id'],'title':s['title'],'file':name,'sha256':manifest['files'][name]})
        (stage/'screens.json').write_text(json.dumps(cases,ensure_ascii=False),encoding='utf-8')
        script="$ErrorActionPreference='Stop'\n"+ACTIVATE+r'''
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$result=@{kind='P2_real_windows_screen_capture';status='running';screens=@();model_used=$false}
try {
 $share='\\host.lan\Data\CAPTURE_ID'
 $root='C:\Qw3Computer\CAPTURE_ID'
 if (Test-Path $root) {throw 'Capture folder already exists'}
 Copy-Item $share $root -Recurse
 $screens=Get-Content (Join-Path $root 'screens.json') -Raw | ConvertFrom-Json
 foreach ($case in $screens) {
  $file=Join-Path $root $case.file
  if ((Get-FileHash $file -Algorithm SHA256).Hash.ToLowerInvariant() -cne $case.sha256) {throw 'Public card hash changed'}
  $url='--app=file:///'+$file.Replace('\','/')
  Start-Process "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe" -ArgumentList @('--no-first-run','--no-default-browser-check','--disable-sync','--user-data-dir=C:/Qw3Computer/ReasoningCardsBrowser',$url)
  $limit=(Get-Date).AddSeconds(30);$h=[IntPtr]::Zero
  while ($h -eq [IntPtr]::Zero -and (Get-Date) -lt $limit) {$h=[CaptureWin]::Find($case.title);Start-Sleep -Milliseconds 300}
  if ($h -eq [IntPtr]::Zero) {throw 'Card window not found'}
  [CaptureWin]::ShowWindow($h,3) | Out-Null
  if (![CaptureWin]::SetForegroundWindow($h)) {
   [CaptureWin]::keybd_event(18,0,0,[UIntPtr]::Zero)
   try {[CaptureWin]::SetForegroundWindow($h) | Out-Null} finally {[CaptureWin]::keybd_event(18,0,2,[UIntPtr]::Zero)}
  }
  Start-Sleep -Seconds 2
  if ([CaptureWin]::GetForegroundWindow() -ne $h) {throw 'Card window is not foreground'}
  $bounds=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds
  $bitmap=New-Object System.Drawing.Bitmap($bounds.Width,$bounds.Height)
  $graphics=[System.Drawing.Graphics]::FromImage($bitmap)
  try {
   $graphics.CopyFromScreen($bounds.X,$bounds.Y,0,0,$bounds.Size)
   $destination=Join-Path $share ($case.id+'.png')
   $bitmap.Save($destination,[System.Drawing.Imaging.ImageFormat]::Png)
  } finally {$graphics.Dispose();$bitmap.Dispose()}
  $result.screens+=@{id=$case.id;image=($case.id+'.png');sha256=(Get-FileHash $destination -Algorithm SHA256).Hash.ToLowerInvariant();width=$bounds.Width;height=$bounds.Height;foreground_handle=$h.ToInt64()}
  $result | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $share 'capture-result.json') -Encoding UTF8
 }
 $result.status='captured_pending_review'
} catch {$result.status='failed';$result.error=$_.Exception.ToString()}
$result | ConvertTo-Json -Depth 8 | Set-Content '\\host.lan\Data\CAPTURE_ID\capture-result.json' -Encoding UTF8
'''.replace('CAPTURE_ID',out.name)
        source=out/'capture.ps1';source.write_text(script,encoding='utf-8-sig')
        await command(sys.executable,APP/'vm/provision.py',source)
        deadline=time.monotonic()+180;receipt=stage/'capture-result.json'
        while time.monotonic()<deadline:
            if receipt.exists():
                try:
                    current=json.loads(receipt.read_text(encoding='utf-8-sig'))
                    if current['status']!='running':break
                except ValueError:pass
            await asyncio.sleep(.5)
        else:raise TimeoutError('Capture still unresolved; inspect guest before another launch')
        report.update(current)
        for s in current.get('screens',[]):
            p=stage/s['image']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=s['sha256']:raise ValueError('Captured PNG changed')
            shutil.copyfile(p,out/p.name)
    except Exception as exc:report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        report['finished_at']=time.time();(out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');lock.close()
    print(json.dumps({'status':report['status'],'screens':len(report.get('screens',[]))}));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--materials',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    asyncio.run(run(parser.parse_args()))
