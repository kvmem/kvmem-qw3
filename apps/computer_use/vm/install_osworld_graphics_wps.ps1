# Setup-only: pinned GIMP and actual WPS for tasks 051 and 080.
$ErrorActionPreference = 'Stop'
$share = '\\host.lan\Data\osworld-installers-01'
$resultFile = "$share\graphics-wps-install-result-01.json"
if (Test-Path -LiteralPath $resultFile) { throw 'Installation result already exists' }
$result = @{kind='pinned_windows_graphics_wps_installation'; passed=$false; packages=@(); started=[DateTime]::UtcNow.ToString('o')}
$packages = @(
    @{file='gimp-3.2.6-setup.exe'; sha256='9337cccbc01d4098ee7a3dab215b3afbe6ece99c5287c92441d6f12cf541ebca';
      arguments='/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP- /ALLUSERS /LOG="C:\Qw3Computer\installers\osworld-01\gimp-install.log"'},
    @{file='WPSOffice.exe'; sha256='eba314934969598e5eeb03c27578c66baaede235972d7407faba8023232ed2b1'; arguments='/S'}
)
try {
    $root = 'C:\Qw3Computer\installers\osworld-01'
    New-Item -ItemType Directory -Path $root -Force | Out-Null
    foreach ($package in $packages) {
        $target = Join-Path $root $package.file
        Copy-Item -LiteralPath (Join-Path $share $package.file) -Destination $target
        $hash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -cne $package.sha256) { throw "Checksum mismatch: $($package.file)" }
        $signature = Get-AuthenticodeSignature -LiteralPath $target
        $row = @{file=$package.file; sha256=$hash; signature_status=[string]$signature.Status;
            publisher=$signature.SignerCertificate.Subject; passed=$false; stage='signature_checked'}
        $result.packages += $row
        if ($signature.Status -ne 'Valid') { throw "Invalid signature: $($package.file)" }
        if ($package.file -eq 'WPSOffice.exe' -and $row.publisher -notmatch 'CN=WPS SOFTWARE PTE\. LTD\.') {
            throw 'Unexpected WPS publisher'
        }
        $process = Start-Process -FilePath $target -ArgumentList $package.arguments -PassThru
        $processHandle = $process.Handle
        $row.stage = 'installing'; $row.process_id = $process.Id
        $result | ConvertTo-Json -Depth 6 | Set-Content "$share\graphics-wps-install-progress-01.json" -Encoding UTF8
        if (!$process.WaitForExit(600000)) { throw "Installer did not finish within ten minutes: $($package.file)" }
        $process.Refresh()
        $row.exit_code = $process.ExitCode
        $row.passed = $process.ExitCode -eq 0
        $row.stage = 'exited'
        if (!$row.passed) { throw "Installation failed: $($package.file), exit $($process.ExitCode)" }
    }
    Copy-Item -LiteralPath "$root\gimp-install.log" -Destination "$share\gimp-install.log"
    $result.passed = $true
} catch {
    $result.error = $_.Exception.ToString()
} finally {
    $result.finished = [DateTime]::UtcNow.ToString('o')
    $result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $resultFile -Encoding UTF8
}
