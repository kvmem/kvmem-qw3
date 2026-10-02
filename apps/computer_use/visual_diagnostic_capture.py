"""Capture neutral Windows fixture screens while no model task is active."""
import argparse
import asyncio
import base64
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import aiohttp

from windows_suite import command

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "apps/computer_use"
SHARE = APP / ".runtime/windows/shared"
CASES = [("Edge", "dashboard.html", "Atlas review desk"), ("Notepad", "intake.txt", "intake"),
         ("Calc", "inventory.xlsx", "inventory.xlsx"), ("Writer", "review.docx", "review.docx"),
         ("Impress", "schedule.pptx", "schedule.pptx"), ("GIMP", "panel.png", "panel")]

ACTIVATE = r'''
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type @'
using System; using System.Text; using System.Runtime.InteropServices;
public class CaptureWin {
 public delegate bool EnumProc(IntPtr h, IntPtr p);
 [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc f, IntPtr p);
 [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
 [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
 [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
 [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
 [DllImport("user32.dll")] public static extern void keybd_event(byte key, byte scan, uint flags, UIntPtr extra);
 public static IntPtr Find(string title) {
  IntPtr found=IntPtr.Zero; EnumWindows((h,p)=> { var s=new StringBuilder(2048); GetWindowText(h,s,2048);
   if(IsWindowVisible(h) && s.ToString().IndexOf(title,StringComparison.OrdinalIgnoreCase)>=0) found=h; return true;
  },IntPtr.Zero); return found;
 }
}
'@
$dismissed=New-Object System.Collections.Generic.List[string]
function Dismiss-KnownFixtureDialog {
 $root=[System.Windows.Automation.AutomationElement]::RootElement
 $h=[CaptureWin]::GetForegroundWindow()
 $dialog=[System.Windows.Automation.AutomationElement]::FromHandle($h)
 if ((Get-Process -Id $dialog.Current.ProcessId).ProcessName -ne 'soffice.bin') {return}
 $name=$dialog.Current.Name
 if ($name -eq 'Default file formats not registered') {$checkbox='Perform check on startup';$button='Cancel'}
 elseif ($name -like 'Tip of the Day:*') {$checkbox='Show tips on startup';$button='OK'}
 else {return}
 $check=$dialog.FindFirst([System.Windows.Automation.TreeScope]::Descendants,(New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::NameProperty,$checkbox)))
 $toggle=$check.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern)
 if ($toggle.Current.ToggleState -eq [System.Windows.Automation.ToggleState]::On) {$toggle.Toggle()}
 $control=$dialog.FindFirst([System.Windows.Automation.TreeScope]::Descendants,(New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::NameProperty,$button)))
 $control.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
 $dismissed.Add($name)
 Start-Sleep -Milliseconds 500
}
'''


def save(path, value):
    temp = path.with_suffix(".tmp"); temp.write_text(json.dumps(value, indent=2) + "\n"); temp.replace(path)


async def run(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    lock = open("/tmp/qw3-osworld-vm.lock", "a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = {"kind": "neutral_visual_screen_capture", "status": "preparing", "model_used": False,
              "agent_success_claimed": False, "started_at": time.time(), "screens": [],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        info = json.loads(await command("docker", "inspect", "qw3-computer-windows"))[0]
        if info["Config"].get("Labels", {}).get("com.qw3.computer-use.owner") != "codex-goal-20260930":
            raise ValueError("Not the task-owned Windows VM")
        token = (SHARE / "executor-token.txt").read_text().strip()
        async with aiohttp.ClientSession(trust_env=False, headers={"Authorization": "Bearer " + token},
                                         timeout=aiohttp.ClientTimeout(total=10)) as client:
            async def desktop(route):
                async with client.get("http://127.0.0.1:18082" + route) as response:
                    response.raise_for_status(); return await response.json()
            if not (await desktop("/health"))["paused"]:
                raise ValueError("Cannot prepare screenshots over an active model task")
            manifest = json.loads((args.materials / "materials.json").read_text())
            for name, expected in manifest["files"].items():
                if hashlib.sha256((args.materials / name).read_bytes()).hexdigest() != expected:
                    raise ValueError("Neutral input changed")
            stage = SHARE / out.name
            shutil.copytree(args.materials, stage)
            report["materials"] = manifest
            for app, filename, title in CASES:
                if app not in args.apps:
                    continue
                record = {"app": app, "file": filename, "title_contains": title, "started_at": time.time()}
                script = "$ErrorActionPreference='Stop'\n" + ACTIVATE + r'''
try {
 $root='C:\Qw3Computer\NEUTRAL_ID'
 if (!(Test-Path $root)) {Copy-Item '\\host.lan\Data\NEUTRAL_ID' $root -Recurse}
 $inputFile=Join-Path $root 'INPUT_FILE'
 if ((Get-FileHash $inputFile -Algorithm SHA256).Hash.ToLowerInvariant() -cne 'INPUT_SHA') {throw 'Neutral input hash changed'}
 if ('APP_NAME' -eq 'Edge') {
   Start-Process "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe" -ArgumentList @('--no-first-run','--no-default-browser-check','--disable-sync','--user-data-dir=C:/Qw3Computer/NeutralVisualBrowser','--app=file:///C:/Qw3Computer/NEUTRAL_ID/INPUT_FILE')
 } elseif ('APP_NAME' -eq 'Notepad') {
   Start-Process notepad.exe -ArgumentList $inputFile
 } elseif ('APP_NAME' -eq 'GIMP') {
   Start-Process 'C:\Program Files\GIMP 3\bin\gimp-3.exe' -ArgumentList $inputFile
 } else {
   Start-Process 'C:\Program Files\LibreOffice\program\soffice.exe' -ArgumentList @('-env:UserInstallation=file:///C:/Qw3Computer/NeutralVisualOffice','--norestore','--nofirststartwizard',$inputFile)
 }
 $limit=(Get-Date).AddSeconds(60);$h=[IntPtr]::Zero
 while ($h -eq [IntPtr]::Zero -and (Get-Date) -lt $limit) {$h=[CaptureWin]::Find('TITLE_TEXT');Start-Sleep -Milliseconds 300}
 if ($h -eq [IntPtr]::Zero) {throw 'Expected neutral document window did not appear'}
 [CaptureWin]::ShowWindow($h,3) | Out-Null
 if (![CaptureWin]::SetForegroundWindow($h)) {
   [CaptureWin]::keybd_event(18,0,0,[UIntPtr]::Zero)
   try {[CaptureWin]::SetForegroundWindow($h) | Out-Null} finally {[CaptureWin]::keybd_event(18,0,2,[UIntPtr]::Zero)}
 }
 Start-Sleep -Seconds 2
 Dismiss-KnownFixtureDialog
 Dismiss-KnownFixtureDialog
 $result=@{prepared=([CaptureWin]::GetForegroundWindow() -eq $h);handle=$h.ToInt64();input_sha256='INPUT_SHA';dismissed_fixture_dialogs=@($dismissed.ToArray())}
} catch {$result=@{prepared=$false;error=$_.Exception.Message}}
$result | ConvertTo-Json | Set-Content '\\host.lan\Data\NEUTRAL_ID-APP_NAME-prepared.json' -Encoding UTF8
'''
                replacements = {"NEUTRAL_ID": out.name, "INPUT_FILE": filename, "INPUT_SHA": manifest["files"][filename],
                                "APP_NAME": app, "TITLE_TEXT": title}
                for name, value in replacements.items(): script = script.replace(name, value)
                source = out / (app + "-prepare.ps1"); source.write_text(script, encoding="utf-8-sig")
                await command(sys.executable, APP / "vm/provision.py", source)
                receipt = SHARE / (out.name + "-" + app + "-prepared.json")
                async with asyncio.timeout(90):
                    while not receipt.exists(): await asyncio.sleep(.5)
                record["preparation"] = json.loads(receipt.read_text(encoding="utf-8-sig"))
                # Even a failed preparation retains its actual screen for analysis.
                observation = await desktop("/observe")
                image = out / (app + ".png"); image.write_bytes(base64.b64decode(observation.pop("image")))
                record.update(image=image.name, image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(), observation=observation,
                              prepared=record["preparation"].get("prepared") is True, finished_at=time.time())
                report["screens"].append(record); save(out / "result.json", report)
                print(json.dumps({"app": app, "prepared": record["prepared"], "foreground": observation["foreground"]}), flush=True)
                if not record["prepared"]:
                    raise ValueError("Neutral window preparation failed; inspect preserved screenshot before continuing")
            report["status"] = "captured_pending_ground_truth_review"
    except Exception as exc:
        report.update(status="halted", error=f"{type(exc).__name__}: {exc}")
    finally:
        report["finished_at"] = time.time(); save(out / "result.json", report); lock.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apps", nargs="+", choices=[case[0] for case in CASES],
                        default=[case[0] for case in CASES])
    result = asyncio.run(run(parser.parse_args()))
    print(json.dumps({"status": result["status"], "error": result.get("error")}))
