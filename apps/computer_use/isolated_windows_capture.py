"""Isolate read-only UIA providers in an owned, time-limited child process.

The input owner and its ledger never restart for a UIA timeout. No expired
worker's controls can become a newer screenshot's evidence.
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

from windows_capture_candidate import APP, ROOT, SHARE, digest, generate, health, install
from platform_pilots import owned_container


WIRE=r'''
public static class UiaWire {
    private static Type SerializerType() {
        return Type.GetType("System.Web.Script.Serialization.JavaScriptSerializer, System.Web.Extensions, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35", true);
    }
    public static string Write(object value) {
        Type type = SerializerType(); object serializer = Activator.CreateInstance(type);
        return (string)type.GetMethod("Serialize", new [] { typeof(object) }).Invoke(serializer, new [] { value });
    }
    public static ControlHints Read(string value) {
        Type type = SerializerType(); object serializer = Activator.CreateInstance(type);
        foreach (var method in type.GetMethods())
            if (method.Name == "Deserialize" && method.IsGenericMethodDefinition && method.GetParameters().Length == 1)
                return (ControlHints)method.MakeGenericMethod(typeof(ControlHints)).Invoke(serializer, new object[] { value });
        throw new InvalidOperationException("Cached observation decoder unavailable");
    }
}
'''


def sources(base):
    candidate=generate(base)
    start=candidate.index('public sealed class ControlHint {')
    cls=candidate.index('public static class DesktopAccessibility {',start)
    collect=candidate.index('    private static ControlHints Collect(',cls)
    end=candidate.index("'@\n[DesktopNative]::SetProcessDPIAware()",collect)
    types=candidate[start:cls]
    collector=candidate[collect:end].replace('private static ControlHints Collect(', 'public static ControlHints Collect(',1)
    reader='''using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Windows.Automation;
'''+types+'public static class IsolatedCollector {\n'+collector+WIRE+r'''
public static class UiaReaderProgram {
    [MTAThread] public static void Main(string[] args) {
        Console.OutputEncoding = new System.Text.UTF8Encoding(false);
        ControlHints result;
        try {
            if (args.Length != 3) throw new ArgumentException("Observation arguments required");
            result = IsolatedCollector.Collect(new IntPtr(long.Parse(args[0])), int.Parse(args[1]), int.Parse(args[2]));
        } catch (Exception) { result = new ControlHints { status = "unavailable" }; }
        Console.Write(UiaWire.Write(result));
    }
}
'''
    source_id=hashlib.sha256(reader.encode()).hexdigest()
    executable='C:\\Qw3Computer\\captures\\uia-'+source_id[:16]+'.exe'
    read=r'''public static class DesktopAccessibility {
    private static Process activeReader;
    public static ControlHints Read(IntPtr handle, int width, int height) {
        // A provider RPC may ignore cancellation. Kill only this reader child,
        // which can observe but has no input or filesystem mutation methods.
        if (activeReader != null) {
            if (!activeReader.HasExited) return new ControlHints { status = "provider_busy" };
            activeReader.Dispose(); activeReader = null;
        }
        var worker = new Process();
        worker.StartInfo = new ProcessStartInfo {
            FileName = @"EXECUTABLE", Arguments = handle.ToInt64()+" "+width+" "+height,
            UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true,
            RedirectStandardError = true, StandardOutputEncoding = new System.Text.UTF8Encoding(false)
        };
        try {
            worker.Start(); activeReader = worker;
            var output = worker.StandardOutput.ReadToEndAsync();
            var error = worker.StandardError.ReadToEndAsync();
            if (!worker.WaitForExit(1500)) {
                worker.Kill(); worker.WaitForExit(250);
                return new ControlHints { status = "timeout" };
            }
            if (!output.Wait(250)) return new ControlHints { status = "output_timeout" };
            if (worker.ExitCode != 0) return new ControlHints { status = "reader_unavailable" };
            return UiaWire.Read(output.Result);
        } catch (Exception) { return new ControlHints { status = "reader_unavailable" }; }
        finally {
            if (activeReader == worker && worker.HasExited) {
                worker.Dispose(); activeReader = null;
            }
        }
    }
}
'''.replace('EXECUTABLE',executable)
    candidate=candidate[:cls]+read+WIRE+candidate[end:]
    candidate=candidate.replace("capture_profile='uia-text-value-v2'", "capture_profile='uia-isolated-value-v3'")
    # Compare all code outside the read-only C# observation region and the
    # already declared health fields, including the entire input/receipt code.
    input_start='function Validate-Action'
    input_end='$script:executorSourceSha256'
    if candidate[candidate.index(input_start):candidate.index(input_end)]!=generate(base)[generate(base).index(input_start):generate(base).index(input_end)]:
        raise ValueError('Native input code changed')
    return candidate,reader,source_id,executable


def stage(output):
    output.mkdir(parents=True,exist_ok=False)
    base=APP/'vm/executor.ps1'
    candidate,reader,reader_id,executable=sources(base.read_text())
    shutil.copyfile(base,output/'baseline-executor.ps1')
    (output/'candidate-executor.ps1').write_text(candidate)
    (output/'reader.cs').write_text(reader)
    manifest={'kind':'isolated_read_only_windows_uia_capture','created_at':time.time(),
        'base_sha256':digest(base),'candidate_sha256':digest(output/'candidate-executor.ps1'),
        'reader_source_sha256':reader_id,'reader_executable':executable,
        'input_validation_delivery_scopes_receipts_unchanged':True,
        'reader_deadline_ms':1500,'reader_can_dispatch_input':False,'qualification_passed':False,'real_model':False}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def compile_reader(output):
    with Path('/tmp/qw3-osworld-vm.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        owned_container('qw3-computer-windows','com.qw3.computer-use.owner','codex-goal-20260930')
        if health().get('paused') is not True:raise ValueError('Pause the input owner before compiling its read-only helper')
        return _compile_reader(output)


def _compile_reader(output):
    manifest=json.loads((output/'manifest.json').read_text())
    if digest(output/'reader.cs')!=manifest['reader_source_sha256']:raise ValueError('Reader source changed')
    target=SHARE/(output.name+'-reader.cs');shutil.copyfile(output/'reader.cs',target)
    script=r'''$ErrorActionPreference='Stop'
$result=@{kind='readonly_uia_reader_compilation';passed=$false;source_sha256='SOURCE'}
try {
 $source='\\host.lan\Data\RUN-reader.cs'
 if((Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant() -cne 'SOURCE'){throw 'Reader source changed'}
 Add-Type -AssemblyName UIAutomationClient,UIAutomationTypes,WindowsBase
 $refs=@('System.dll','System.Core.dll',[Windows.Automation.AutomationElement].Assembly.Location,[Windows.Automation.AutomationPattern].Assembly.Location,[Windows.Rect].Assembly.Location)
 $destination='EXECUTABLE'
 if(Test-Path $destination){throw 'Reader identity reused'}
 New-Item -ItemType Directory -Force (Split-Path $destination) | Out-Null
 Add-Type -TypeDefinition (Get-Content $source -Raw) -ReferencedAssemblies $refs -OutputType ConsoleApplication -OutputAssembly $destination
 $result.executable=$destination;$result.binary_sha256=(Get-FileHash $destination -Algorithm SHA256).Hash.ToLowerInvariant();$result.passed=$true
}catch{$result.error=$_.Exception.ToString()}
$result | ConvertTo-Json | Set-Content '\\host.lan\Data\RUN-reader-compilation.json' -Encoding UTF8
'''
    for marker,value in [('RUN',output.name),('SOURCE',manifest['reader_source_sha256']),('EXECUTABLE',manifest['reader_executable'])]:script=script.replace(marker,value)
    path=output/'compile-reader.ps1';path.write_text(script,encoding='utf-8-sig')
    subprocess.run([sys.executable,str(APP/'vm/provision.py'),str(path)],cwd=ROOT,check=True)
    proof_path=SHARE/(output.name+'-reader-compilation.json')
    deadline=time.monotonic()+40
    while not proof_path.exists():
        if time.monotonic()>deadline:raise TimeoutError('No reader compilation receipt')
        time.sleep(.5)
    proof=json.loads(proof_path.read_text(encoding='utf-8-sig'))
    (output/'reader-compilation.json').write_text(json.dumps(proof,indent=2)+'\n')
    if not proof.get('passed'):raise RuntimeError('Read-only worker compilation failed')
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['stage','install'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path)
    args=parser.parse_args()
    if args.operation=='stage':result=stage(args.output)
    else:
        compile_reader(args.output)
        result=install(args.output,args.previous)
    print({key:result.get(key) for key in ['candidate_sha256','installed','qualification_passed']})
