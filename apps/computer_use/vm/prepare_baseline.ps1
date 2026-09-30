# Setup-only script for the dedicated VM. Run with vm/provision.py after the
# application is stopped. Existing test outputs are archived on the host share.
$ErrorActionPreference = 'Stop'
$root = 'C:\Qw3Computer'
if (!(Test-Path "$root\executor-ready.txt")) { throw 'Expected test executor installation is missing' }
$share = '\\host.lan\Data'
if (Test-Path 'Z:\') { $share = 'Z:\' }
Start-Transcript -Path "$share\baseline-preparation.log" -Force
$archive = Join-Path $share ('before-baseline-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory $archive | Out-Null
Get-Process notepad,msedge -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
# Drop only the dedicated VM's test clipboard between qualification cases.
Add-Type -AssemblyName System.Windows.Forms
[Windows.Forms.Clipboard]::Clear()
foreach ($directory in @('outputs','actions')) {
    if (Test-Path "$root\$directory") {
        Copy-Item "$root\$directory" "$archive\$directory" -Recurse
        Remove-Item "$root\$directory\*" -Recurse -Force
    }
}
$notepad = Get-AppxPackage -Name Microsoft.WindowsNotepad
if ($notepad) {
    $tabs = Join-Path $env:LOCALAPPDATA "Packages\$($notepad.PackageFamilyName)\LocalState\TabState"
    if (Test-Path $tabs) { Copy-Item $tabs "$archive\NotepadTabState" -Recurse; Remove-Item $tabs -Recurse -Force }
}
if (Test-Path "$root\BrowserProfile") { Remove-Item "$root\BrowserProfile" -Recurse -Force }
Copy-Item "$share\bootstrap.ps1" "$root\bootstrap.ps1" -Force
Copy-Item "$share\fixture.html" "$root\fixture.html" -Force
[guid]::NewGuid().ToString() | Set-Content "$root\desktop-id.txt"
@{
    timestamp=(Get-Date -Format o)
    memory_bytes=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    os=(Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber)
    executor_sha256=(Get-FileHash "$root\executor.ps1" -Algorithm SHA256).Hash
    fixture_sha256=(Get-FileHash "$root\fixture.html" -Algorithm SHA256).Hash
    archive=$archive
} | ConvertTo-Json -Depth 5 | Set-Content "$share\baseline-ready.json" -Encoding UTF8
Stop-Transcript
