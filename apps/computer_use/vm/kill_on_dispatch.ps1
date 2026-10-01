param([switch]$Worker)
# Dedicated VM fault-injection helper, never a Computer MCP tool.
# Prepared only; do not arm while the frozen qualification queue is running.
$ErrorActionPreference = 'Stop'
$share = '\\host.lan\Data'
if (!$Worker) {
    # Use a unique staged path so provision.py cannot overwrite a running worker.
    $scriptPath = Join-Path $share 'input-kill-worker.ps1'
    if (!(Test-Path $scriptPath)) { throw 'Stage input-kill-worker.ps1 before arming' }
    Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @(
        '-NoProfile','-STA','-ExecutionPolicy','Bypass','-File',$scriptPath,'-Worker') | Out-Null
    exit
}
$result = @{kind='kill_owned_executor_after_durable_dispatch'; passed=$false}
try {
    $request = Get-Content (Join-Path $share 'input-kill-request.json') -Raw | ConvertFrom-Json
    if ($request.action_id -notmatch '^[a-zA-Z0-9_-]{8,80}$') { throw 'Invalid action identifier' }
    $action = Join-Path 'C:\Qw3Computer\actions' ($request.action_id + '.json')
    if (Test-Path $action) { throw 'Choose a fresh action identifier' }
    $targets = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object {
        $_.CommandLine -match '(?i)-File\s+"?C:\\Qw3Computer\\executor\.ps1"?(\s|$)'
    })
    if ($targets.Count -ne 1) { throw "Expected exactly one owned executor, found $($targets.Count)" }
    $targetId = [int]$targets[0].ProcessId
    Add-Type @'
using System.Runtime.InteropServices;
public static class InputKillClipboard {
    [DllImport("user32.dll")] public static extern uint GetClipboardSequenceNumber();
}
'@
    $result.action_id = $request.action_id
    $result.executor_pid = $targetId
    $result.executor_sha256 = (Get-FileHash 'C:\Qw3Computer\executor.ps1' -Algorithm SHA256).Hash.ToLowerInvariant()
    $result.clipboard_sequence_before = [InputKillClipboard]::GetClipboardSequenceNumber()
    @{action_id=$request.action_id; executor_pid=$targetId; worker_pid=$PID; ready=$true} |
        ConvertTo-Json | Set-Content (Join-Path $share 'input-kill-ready.json') -Encoding UTF8
    $watch = [Diagnostics.Stopwatch]::StartNew()
    while ($watch.Elapsed.TotalSeconds -lt 90) {
        if (Test-Path $action) {
            try { $record = Get-Content $action -Raw | ConvertFrom-Json } catch { $record = $null }
            if ($record) {
                $result.status_before_kill = $record.status
                if ($record.status -cne 'dispatching') { throw 'Missed the dispatch window; no executor was killed' }
                # Identity was resolved before arming; verify it is still the
                # exact owned command rather than a subsequently reused PID.
                $target = Get-CimInstance Win32_Process -Filter "ProcessId=$targetId"
                if (!$target -or $target.CommandLine -notmatch '(?i)-File\s+"?C:\\Qw3Computer\\executor\.ps1"?(\s|$)') {
                    throw 'Owned executor identity changed'
                }
                Stop-Process -Id $targetId -Force
                $result.killed_at = Get-Date -Format o
                $after = Get-Content $action -Raw | ConvertFrom-Json
                $result.status_after_kill = $after.status
                if ($after.status -cne 'dispatching') { throw 'Executor completed before termination; this does not qualify the dispatch window' }
                $result.clipboard_sequence_after = [InputKillClipboard]::GetClipboardSequenceNumber()
                $result.clipboard_sequence_unchanged = $result.clipboard_sequence_before -eq $result.clipboard_sequence_after
                $result.passed = $true
                break
            }
        }
        Start-Sleep -Milliseconds 1
    }
    if (!$result.passed) { throw 'No selected action was dispatched within 90 seconds' }
} catch { $result.error = $_.Exception.Message }
finally {
    $result | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $share 'input-kill-result.json') -Encoding UTF8
}
