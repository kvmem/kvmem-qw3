param([ValidateSet('kill','focus')][string]$Mode='kill',
    [ValidatePattern('^[a-zA-Z0-9_-]{1,40}$')][string]$Run='01', [switch]$Worker)
# Administration-only fault injection on the owned Windows executor. A host
# test provides a fresh action ID; this script cannot dispatch desktop actions.
$ErrorActionPreference='Stop'
$share='\\host.lan\Data'
if (!$Worker) {
    Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @('-NoProfile','-STA',
        '-ExecutionPolicy','Bypass','-File',$PSCommandPath,'-Worker','-Mode',$Mode,'-Run',$Run) | Out-Null
    exit
}
$result=@{kind='real_drag_interruption'; mode=$Mode; passed=$false}
try {
    $request=Get-Content "$share\drag-interruption-$Mode-$Run-request.json" -Raw | ConvertFrom-Json
    if ($request.action_id -notmatch '^[a-zA-Z0-9_-]{8,80}$') {throw 'Invalid action ID'}
    $recordPath="C:\Qw3Computer\actions\$($request.action_id).json"
    if (Test-Path $recordPath) {throw 'Choose a fresh action ID'}
    $targets=@(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object {
        $_.CommandLine -match '(?i)-File\s+"?C:\\Qw3Computer\\executor\.ps1"?(\s|$)'
    })
    if ($targets.Count -ne 1) {throw 'Expected one owned executor'}
    $targetId=[int]$targets[0].ProcessId
    $created=$targets[0].CreationDate
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class DragFault {
 [DllImport("user32.dll")] public static extern short GetAsyncKeyState(int key);
 [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
 [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr window, out uint process);
 [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
 [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint first, uint second, bool attach);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr window);

}
'@
    function Held { return @(@(1,16,17,18) | ForEach-Object {
        @{key=$_; down=(([DragFault]::GetAsyncKeyState($_) -band 0x8000) -ne 0)}
    }) }
    $result.action_id=$request.action_id
    $result.old_executor_pid=$targetId
    @{ready=$true; action_id=$request.action_id} | ConvertTo-Json |
        Set-Content "$share\drag-interruption-$Mode-$Run-ready.json" -Encoding UTF8
    $watch=[Diagnostics.Stopwatch]::StartNew()
    while ($watch.Elapsed.TotalSeconds -lt 90) {
        if (([DragFault]::GetAsyncKeyState(1) -band 0x8000) -ne 0 -and (Test-Path $recordPath)) {
            $record=Get-Content $recordPath -Raw | ConvertFrom-Json
            if ($record.status -ne 'dispatching' -or $record.request.action.kind -ne 'drag') {
                throw 'Missed selected drag dispatch window'
            }
            $result.before=Held
            if (@($result.before | Where-Object {$_.key -eq 18 -and $_.down}).Count -ne 1) {
                throw 'ALT was not held during selected drag'
            }
            if ($Mode -eq 'kill') {
                $target=Get-CimInstance Win32_Process -Filter "ProcessId=$targetId"
                if (!$target -or $target.CreationDate -ne $created) {throw 'Owned executor identity changed'}
                Stop-Process -Id $targetId -Force
                $recoveryDeadline=[Diagnostics.Stopwatch]::StartNew()
                while ((Test-Path 'C:\Qw3Computer\held-drag-input.json') -and $recoveryDeadline.Elapsed.TotalSeconds -lt 30) {
                    Start-Sleep -Milliseconds 100
                }
                $recovered=@(Get-ChildItem 'C:\Qw3Computer\input-recovery-*.json' | ForEach-Object {
                    Get-Content $_.FullName -Raw | ConvertFrom-Json
                } | Where-Object {$_.action_id -eq $request.action_id})
                $result.recovery=$recovered
                if ($recovered.Count -ne 1) {throw 'No unique restart recovery record'}
            } else {
                $original=[DragFault]::GetForegroundWindow()
                $result.foreground_before=$original.ToInt64()
                [uint32]$foregroundPid=0
                $foregroundThread=[DragFault]::GetWindowThreadProcessId($original,[ref]$foregroundPid)
                $workerThread=[DragFault]::GetCurrentThreadId()
                $form=New-Object Windows.Forms.Form
                $form.Text='Synthetic foreground queue fault';$form.TopMost=$true
                $form.Show()
                $result.fault_kind='foreground_input_queue_attachment'
                $attached=[DragFault]::AttachThreadInput($workerThread,$foregroundThread,$true)
                $result.input_queues_attached=$attached
                try {
                    if (!$attached) {throw 'Foreground queue fault was not injected'}
                    [DragFault]::SetForegroundWindow($form.Handle) | Out-Null
                    $until=[Diagnostics.Stopwatch]::StartNew()
                    while ($until.Elapsed.TotalMilliseconds -lt 700) {
                        [Windows.Forms.Application]::DoEvents();Start-Sleep -Milliseconds 10
                    }
                    $result.foreground_after=[DragFault]::GetForegroundWindow().ToInt64()
                } finally {
                    if ($attached) {[DragFault]::AttachThreadInput($workerThread,$foregroundThread,$false) | Out-Null}
                    $form.Close();$form.Dispose()
                }
                $record=Get-Content $recordPath -Raw | ConvertFrom-Json
                # Queue attachment may transiently deactivate the old window
                # before the new one is visible. Require the executor's actual
                # foreground-change receipt, rather than claiming that Show()
                # proved a successful focus change. Key release effects are
                # independently covered by normal and process-kill probes.
                $result.focus_injected=($record.status -eq 'uncertain' -and $record.error -eq 'Foreground changed during drag')
                $result.held_marker_removed=!(Test-Path 'C:\Qw3Computer\held-drag-input.json')
                if (!$result.focus_injected -or !$result.held_marker_removed) {throw 'Executor did not record foreground interruption and cleanup'}
            }
            $result.after=Held
            $result.record=Get-Content $recordPath -Raw | ConvertFrom-Json
            $result.passed=(@($result.after | Where-Object {$_.down}).Count -eq 0 -and
                $result.record.status -in @('dispatching','uncertain'))
            if (!$result.passed) {throw 'Held input or unexpected receipt after interruption'}
            break
        }
        Start-Sleep -Milliseconds 5
    }
    if (!$result.passed) {throw 'Selected drag did not enter held state before deadline'}
} catch {$result.error=$_.Exception.ToString()}
finally {
    $result | ConvertTo-Json -Depth 12 | Set-Content "$share\drag-interruption-$Mode-$Run-result.json" -Encoding UTF8
}
