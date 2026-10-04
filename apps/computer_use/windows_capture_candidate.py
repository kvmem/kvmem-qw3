"""Stage a Windows UIA observation upgrade without editing the frozen driver.

Only cached visible text/value capture and read-only health provenance change.
Input validation, delivery, receipts, scopes and recovery remain byte-identical.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

from osworld_windows.source import ROOT
from platform_pilots import owned_container


APP=ROOT/'apps/computer_use'
SHARE=APP/'.runtime/windows/shared'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(base):
    candidate=base
    replacements=[
        ('    private static Task<ControlHints> pending;',
         '''    private static Task<ControlHints> pending;
    private static readonly System.Collections.Concurrent.BlockingCollection<System.Action> requests =
        new System.Collections.Concurrent.BlockingCollection<System.Action>(1);
    static DesktopAccessibility() {
        // Keep all UIA acquisition/cached reads on one lasting MTA apartment,
        // rather than moving COM element lifetimes between thread-pool tasks.
        var worker = new System.Threading.Thread(() => {
            foreach (var collect in requests.GetConsumingEnumerable()) collect();
        });
        worker.IsBackground = true;
        worker.Name = "Qw3ReadOnlyUia";
        worker.SetApartmentState(System.Threading.ApartmentState.MTA);
        worker.Start();
    }'''),
        ('        pending = Task.Run(() => Collect(handle, width, height));',
         '''        var completion = new TaskCompletionSource<ControlHints>();
        pending = completion.Task;
        requests.Add(() => {
            try { completion.SetResult(Collect(handle, width, height)); }
            catch (Exception) { completion.SetResult(new ControlHints { status = "unavailable" }); }
        });'''),
        ('            var cache = new CacheRequest();',
         '            var cache = new CacheRequest();\n'
         '            cache.AutomationElementMode = AutomationElementMode.None;'),
        ('    public bool focused;\n', '    public bool focused;\n    public string value;\n    public string value_status;\n'),
        ('                ControlType.Edit, ControlType.MenuItem,',
         '                ControlType.Edit, ControlType.Text, ControlType.MenuItem,'),
        ('                new PropertyCondition(AutomationElement.IsEnabledProperty, true));',
         '                new PropertyCondition(AutomationElement.IsEnabledProperty, true),\n'
         '                new PropertyCondition(AutomationElement.IsPasswordProperty, false));'),
        ('            AutomationElementCollection elements;',
         '            cache.Add(ValuePattern.Pattern);\n            cache.Add(ValuePattern.ValueProperty);\n'
         '            AutomationElementCollection elements;'),
        ('                hints.Add(new ControlHint { name = name,',
         '''                string value = null, valueStatus = "unsupported";
                object cachedValuePattern;
                try {
                    if (element.TryGetCachedPattern(ValuePattern.Pattern, out cachedValuePattern)) {
                        value = ((ValuePattern)cachedValuePattern).Cached.Value;
                        valueStatus = value != null && value.Length <= 4096 ? "available" : "too_long_or_unavailable";
                        if (valueStatus != "available") value = null;
                    }
                } catch (Exception) { value = null; valueStatus = "unavailable"; }
                hints.Add(new ControlHint { name = name,'''),
        ('                    focused = info.HasKeyboardFocus });',
         '                    focused = info.HasKeyboardFocus, value = value, value_status = valueStatus });'),
    ]
    for original,replacement in replacements:
        if candidate.count(original)!=1:raise ValueError('Frozen capture anchor changed')
        candidate=candidate.replace(original,replacement)
    # Assert that all native input/delivery code outside this read-only C# block
    # is unchanged before adding auditable health metadata.
    start='public sealed class ControlHint {';end="'@\n[DesktopNative]::SetProcessDPIAware()"
    def non_capture(source):return source.split(start)[0]+end+source.split(end)[1]
    if non_capture(candidate)!=non_capture(base):raise ValueError('Non-capture native code changed')
    source_digest=hashlib.sha256(base.encode()).hexdigest()
    candidate=candidate.replace('$listener = [Net.HttpListener]::new()',
        "$script:executorSourceSha256 = Hash-Bytes ([IO.File]::ReadAllBytes($PSCommandPath))\n"
        '$listener = [Net.HttpListener]::new()')
    original="process_id=$PID; version='0.1.0'"
    if candidate.count(original)!=1:raise ValueError('Health provenance anchor changed')
    candidate=candidate.replace(original,original+"; capture_profile='uia-text-value-v2'; source_sha256=$script:executorSourceSha256; base_source_sha256='"+source_digest+"'")
    return candidate


def stage(output):
    output.mkdir(parents=True,exist_ok=False)
    base=APP/'vm/executor.ps1';source=generate(base.read_text())
    shutil.copyfile(base,output/'baseline-executor.ps1')
    (output/'candidate-executor.ps1').write_text(source)
    manifest={'kind':'read_only_windows_uia_capture_candidate','created_at':time.time(),
              'base_sha256':digest(base),'candidate_sha256':digest(output/'candidate-executor.ps1'),
              'input_validation_delivery_scopes_receipts_unchanged':True,
              'capture_changes':['visible Text controls','cached ValuePattern values with exact availability',
                                 'password controls excluded before cache retrieval','read-only health source identity'],
              'uia_threading':'single persistent windowless MTA; cached-only element references; one bounded outstanding capture',
              'qualification_passed':False,'real_model':False}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def health():
    token=(SHARE/'executor-token.txt').read_text().strip()
    request=urllib.request.Request('http://127.0.0.1:18082/health',headers={'Authorization':'Bearer '+token})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request,timeout=5) as response:return json.load(response)


def install(output,previous=None):
    manifest=json.loads((output/'manifest.json').read_text())
    if digest(output/'candidate-executor.ps1')!=manifest['candidate_sha256']:
        raise ValueError('Candidate source changed')
    with Path('/tmp/qw3-osworld-vm.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        vm=owned_container('qw3-computer-windows','com.qw3.computer-use.owner','codex-goal-20260930')
        before=health()
        if before.get('paused') is not True:raise ValueError('Pause the owned executor before upgrading capture')
        installed_expected=manifest['base_sha256']
        if previous is not None:
            prior=json.loads((previous/'manifest.json').read_text())
            if (prior['base_sha256']!=manifest['base_sha256']
                    or digest(previous/'candidate-executor.ps1')!=prior['candidate_sha256']
                    or before.get('source_sha256')!=prior['candidate_sha256']):
                raise ValueError('Previous isolated capture identity changed')
            installed_expected=prior['candidate_sha256']
        old_pid=before['process_id']
        if type(old_pid) is not int:raise ValueError('No exact executor process identity')
        staged=SHARE/(output.name+'-candidate.ps1');shutil.copyfile(output/'candidate-executor.ps1',staged)
        script=r'''$ErrorActionPreference='Stop'
$result=@{installed=$false;old_pid=PID}
try {
 $old=Get-CimInstance Win32_Process -Filter 'ProcessId=PID'
 if(!$old -or $old.CommandLine -notmatch '(?i)-File\s+"?C:\\Qw3Computer\\executor\.ps1"?(\s|$)'){throw 'Executor process identity changed'}
 $base='C:\Qw3Computer\executor.ps1'
 if((Get-FileHash $base -Algorithm SHA256).Hash.ToLowerInvariant() -cne 'BASE'){throw 'Installed baseline source differs'}
 $source='\\host.lan\Data\RUN-candidate.ps1'
 if((Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant() -cne 'CANDIDATE'){throw 'Staged candidate changed'}
 $previous='\\host.lan\Data\RUN-prior-shared-executor.ps1'
 if(Test-Path $previous){throw 'Upgrade identity reused'}
 Copy-Item '\\host.lan\Data\executor.ps1' $previous
 Copy-Item $source '\\host.lan\Data\executor.ps1' -Force
 # The existing owned bootstrap relaunches this exact service from the share.
 Stop-Process -Id PID -Force
 $result.installed=$true
}catch{$result.error=$_.Exception.ToString()}
$result | ConvertTo-Json | Set-Content '\\host.lan\Data\RUN-install.json' -Encoding UTF8
'''
        for marker,replacement in [('PID',str(old_pid)),('BASE',installed_expected),
                                   ('CANDIDATE',manifest['candidate_sha256']),('RUN',output.name)]:
            script=script.replace(marker,replacement)
        target=output/'install.ps1';target.write_text(script,encoding='utf-8-sig')
        subprocess.run([sys.executable,str(APP/'vm/provision.py'),str(target)],cwd=ROOT,check=True)
        deadline=time.monotonic()+45
        while True:
            try:
                after=health()
                if after.get('process_id')!=old_pid and after.get('source_sha256')==manifest['candidate_sha256']:
                    if after.get('paused') is not True:raise ValueError('Upgraded service is active')
                    break
            except (OSError,urllib.error.URLError):pass
            if time.monotonic()>deadline:raise TimeoutError('No matching paused upgraded executor')
            time.sleep(.5)
        report={'kind':'owned_read_only_capture_install','container_id':vm['Id'],
                'before':before,'after':after,'manifest_sha256':digest(output/'manifest.json'),
                'previous_installed_source_sha256':installed_expected,
                'installed':True,'qualification_passed':False,'at':time.time()}
        (output/'install-result.json').write_text(json.dumps(report,indent=2)+'\n')
        return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['stage','install'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path,help='Verified preceding isolated capture, required for upgrading it')
    args=parser.parse_args()
    report=stage(args.output) if args.operation=='stage' else install(args.output,args.previous)
    print({key:report.get(key) for key in ['candidate_sha256','installed','qualification_passed']})
