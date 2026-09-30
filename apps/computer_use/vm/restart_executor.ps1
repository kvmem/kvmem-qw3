# Setup-only fault injection inside the dedicated VM. The bootstrap restarts
# the executor; no unrelated PowerShell process is terminated.
$ErrorActionPreference = 'Stop'
$targets = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object {
    $_.CommandLine -match '(?i)-File\s+"?C:\\Qw3Computer\\executor\.ps1"?(\s|$)'
})
if ($targets.Count -ne 1) { throw "Expected one task-owned executor; found $($targets.Count)" }
$oldPid = $targets[0].ProcessId
Stop-Process -Id $oldPid -Force
@{ stopped_pid=$oldPid; stopped_at=(Get-Date -Format o) } |
    ConvertTo-Json | Set-Content '\\host.lan\Data\executor-restart.json' -Encoding UTF8
