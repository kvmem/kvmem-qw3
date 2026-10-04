"""Owner-only Windows transport for the shared browser recovery scenarios."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request

from fault_browser_pilot import SERVER
from fault_browser_windows_guest import CONDITIONS,page
from osworld_windows.bundle import verify_bundle
from osworld_windows.source import ROOT
from platform_pilots import owned_container


APP=ROOT/'apps/computer_use'
SHARE=APP/'.runtime/windows/shared'


def identity(value):
    if not isinstance(value,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{8,45}',value):
        raise ValueError('Use a unique short fixture identity')
    return value


def health():
    token=(SHARE/'executor-token.txt').read_text().strip()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req=urllib.request.Request('http://127.0.0.1:18082/health',headers={'Authorization':'Bearer '+token})
    with opener.open(req,timeout=5) as response:return json.load(response)


def execute(script,output,label):
    target=Path(output)/(label+'.ps1');target.write_text(script,encoding='utf-8-sig')
    subprocess.run([sys.executable,str(APP/'vm/provision.py'),str(target)],cwd=ROOT,check=True,
                   stdout=subprocess.DEVNULL)


def wait_json(path):
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        try:return json.loads(path.read_text(encoding='utf-8-sig'))
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(.2)
    raise TimeoutError('No complete fixture receipt')


def initialize(value,condition,directory,bundle):
    value=identity(value)
    if condition not in CONDITIONS:raise ValueError('Unknown recovery condition')
    vm=owned_container('qw3-computer-windows','com.qw3.computer-use.owner','codex-goal-20260930')
    if not health().get('paused'):raise ValueError('Windows input must be acknowledged paused')
    archive=verify_bundle(bundle)['archive']
    source=APP/'fault_browser_windows_guest.py'
    staged=SHARE/(value+'-fault.py');staged_page=SHARE/(value+'-fault.html')
    with staged.open('xb') as stream:stream.write(source.read_bytes())
    with staged_page.open('x',encoding='utf-8') as stream:stream.write(page(condition,SERVER.read_text()))
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    page_digest=hashlib.sha256(staged_page.read_bytes()).hexdigest()
    script=r'''$ErrorActionPreference='Stop'
$result=@{prepared=$false}
try {
 $dir='C:\Qw3Computer\P3Tasks\IDENTITY'
 if(Test-Path $dir){throw 'Fixture identity already exists'}
 if(Get-NetTCPConnection -State Listen -LocalPort 18083,8086 -ErrorAction SilentlyContinue){throw 'Native model app or fixture port is occupied'}
 $python='C:\Qw3Computer\clients\CLIENT\python\python.exe'
 if(!(Test-Path $python)){throw 'Qualified client Python is missing'}
 New-Item -ItemType Directory $dir | Out-Null
 $result.directory=$dir
 $source='\\host.lan\Data\IDENTITY-fault.py';$page='\\host.lan\Data\IDENTITY-fault.html'
 if((Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant() -cne 'HELPER'){throw 'Fixture source changed'}
 if((Get-FileHash $page -Algorithm SHA256).Hash.ToLowerInvariant() -cne 'PAGE'){throw 'Fixture page changed'}
 Copy-Item $source "$dir\server.py";Copy-Item $page "$dir\page.html"
 $entries='C:\Qw3Computer\application-entries.json'
 if(Test-Path $entries){Copy-Item $entries "$dir\previous-entries.json"}
 $profile="$dir\BrowserProfile"
 @{version=1;run='IDENTITY';task='p3';browser_profile=$profile;browser_urls=@('http://127.0.0.1:8086/');explorer_path='C:\Qw3Computer\outputs'} | ConvertTo-Json | Set-Content $entries -Encoding UTF8
 $worker=Start-Process $python -ArgumentList @('-X','utf8',"$dir\server.py",'--directory',$dir,'--condition','CONDITION','--page',"$dir\page.html",'--port','8086') -WindowStyle Hidden -PassThru -RedirectStandardOutput "$dir\stdout.log" -RedirectStandardError "$dir\stderr.log"
 $result.worker_pid=$worker.Id
 $deadline=[DateTime]::UtcNow.AddSeconds(15)
 while(!(Test-Path "$dir\server-ready.json")){if($worker.HasExited -or [DateTime]::UtcNow -gt $deadline){throw 'Fixture server failed'};Start-Sleep -Milliseconds 100}
 $edge="${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
 $browser=Start-Process $edge -ArgumentList @('--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-sync','--force-renderer-accessibility',"--user-data-dir=$profile",'http://127.0.0.1:8086/') -PassThru
 $result=@{prepared=$true;worker_pid=$worker.Id;browser_pid=$browser.Id;directory=$dir;browser_profile=$profile;condition='CONDITION';helper_sha256='HELPER';page_sha256='PAGE'}
}catch{$result.error=$_.Exception.ToString()}
$result | ConvertTo-Json -Depth 6 | Set-Content '\\host.lan\Data\IDENTITY-fault-ready.json' -Encoding UTF8
'''
    for marker,replacement in [('IDENTITY',value),('CLIENT',archive['sha256'][:12]),('CONDITION',condition),
                               ('HELPER',digest),('PAGE',page_digest)]:script=script.replace(marker,replacement)
    execute(script,directory,'prepare-windows-fault')
    result=wait_json(SHARE/(value+'-fault-ready.json'))
    result.update(identity=value,container_id=vm['Id'],bundle_sha256=archive['sha256'],
                  template_sha256=hashlib.sha256(SERVER.read_bytes()).hexdigest())
    (Path(directory)/'windows-fault-fixture.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result.get('prepared'):raise RuntimeError('Fixture preparation failed: '+str(result.get('error')))
    return result


def arm(fixture,directory):
    value=identity(fixture['identity'])
    if owned_container('qw3-computer-windows','com.qw3.computer-use.owner','codex-goal-20260930')['Id']!=fixture['container_id']:
        raise ValueError('Windows fixture device changed')
    script=r'''$ErrorActionPreference='Stop'
$dir='C:\Qw3Computer\P3Tasks\IDENTITY'
$worker=Get-CimInstance Win32_Process -Filter 'ProcessId=PID'
if(!$worker -or $worker.CommandLine -notlike "*$dir\server.py*"){throw 'Fixture server identity changed'}
$path="$dir\arm.json"
if(Test-Path $path){throw 'Fixture already armed'}
@{armed_at_ms=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()} | ConvertTo-Json | Set-Content $path -Encoding UTF8
'''.replace('IDENTITY',value).replace('PID',str(fixture['worker_pid']))
    execute(script,directory,'arm-windows-fault')


def collect(fixture,directory):
    value=identity(fixture['identity'])
    if not health().get('paused'):raise ValueError('Do not collect before acknowledged desktop pause')
    if owned_container('qw3-computer-windows','com.qw3.computer-use.owner','codex-goal-20260930')['Id']!=fixture['container_id']:
        raise ValueError('Fixture container identity changed')
    script=r'''$ErrorActionPreference='Stop'
$dir='C:\Qw3Computer\P3Tasks\IDENTITY'
$worker=Get-CimInstance Win32_Process -Filter 'ProcessId=PID'
if($worker -and $worker.CommandLine -notlike "*$dir\server.py*"){throw 'Fixture process identity changed'}
if($worker){Stop-Process -Id PID -Force}
# Stop only processes demonstrably created with this private test profile.
Get-CimInstance Win32_Process -Filter "Name='msedge.exe'" | Where-Object {$_.CommandLine -like "*$dir\BrowserProfile*"} | ForEach-Object {Stop-Process -Id $_.ProcessId -Force}
$export='\\host.lan\Data\IDENTITY-fault-final'
if(Test-Path $export){throw 'Fixture collection already exists'}
New-Item -ItemType Directory $export | Out-Null
foreach($name in @('fault-state.json','submissions.jsonl','arm.json','server-ready.json','stdout.log','stderr.log')){
 if(Test-Path "$dir\$name"){Copy-Item "$dir\$name" "$export\$name"}
}
$entries='C:\Qw3Computer\application-entries.json'
$active=Get-Content $entries -Raw | ConvertFrom-Json
if($active.run -cne 'IDENTITY'){throw 'Application entry was replaced; no restore performed'}
if(Test-Path "$dir\previous-entries.json"){Copy-Item "$dir\previous-entries.json" $entries -Force}else{Remove-Item $entries}
@{collected=$true;worker_stopped=!(Get-Process -Id PID -ErrorAction SilentlyContinue)} | ConvertTo-Json | Set-Content "$export\collection.json" -Encoding UTF8
'''.replace('IDENTITY',value).replace('PID',str(fixture['worker_pid']))
    execute(script,directory,'collect-windows-fault')
    source=SHARE/(value+'-fault-final');receipt=wait_json(source/'collection.json')
    shutil.copytree(source,Path(directory)/'windows-fault-final')
    if not receipt.get('worker_stopped'):raise ValueError('Fixture server stop unconfirmed')
    return receipt
