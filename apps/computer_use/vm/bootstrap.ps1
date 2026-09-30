# Runs inside the dedicated test VM's logged-in desktop, never on the host.
$ErrorActionPreference = 'Stop'
$root = 'C:\Qw3Computer'
New-Item -ItemType Directory -Force $root | Out-Null
Set-Content "$root\bootstrap-ready.txt" (Get-Date -Format o)
# The host stages the reviewed executor into this task-owned shared directory.
while ($true) {
    $share = '\\host.lan\Data'
    if (Test-Path 'Z:\') { $share = 'Z:\' }
    if ((Test-Path "$share\executor.ps1") -and (Test-Path "$share\executor-token.txt")) {
        Copy-Item "$share\executor.ps1" "$root\executor.ps1" -Force
        Copy-Item "$share\executor-token.txt" "$root\executor-token.txt" -Force
        if (Test-Path "$share\fixture.html") { Copy-Item "$share\fixture.html" "$root\fixture.html" -Force }
        New-Item -ItemType Directory -Force "$root\outputs" | Out-Null
        # Dedicated VM test profile: keep downloaded fixtures inside its outputs.
        $policy = 'HKCU:\Software\Policies\Microsoft\Edge'
        New-Item $policy -Force | Out-Null
        New-ItemProperty $policy -Name DownloadDirectory -Value "$root\outputs" -PropertyType String -Force | Out-Null
        New-ItemProperty $policy -Name PromptForDownloadLocation -Value 0 -PropertyType DWord -Force | Out-Null
        & powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File "$root\executor.ps1" *> "$root\executor.log"
    }
    Start-Sleep -Seconds 5
}
