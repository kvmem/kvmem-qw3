# Setup-only installation of pinned, official packages in the dedicated VM.
# No real mail account is configured, and this script is not an agent tool.
$ErrorActionPreference = 'Stop'
$share = '\\host.lan\Data\osworld-installers-01'
$resultFile = "$share\office-install-result-01.json"
if (Test-Path -LiteralPath $resultFile) { throw 'Installation result already exists' }
$result = @{kind='pinned_windows_office_installation'; passed=$false; packages=@(); started=[DateTime]::UtcNow.ToString('o')}
$packages = @(
    @{file='LibreOffice_26.8.0_Win_x86-64.msi'; sha256='4aa6c6e1895f4055104effcb556bd3362d20c6ad707c149543304f395ef9db95'; publisher='The Document Foundation'},
    @{file='Thunderbird_Setup_157.0.msi'; sha256='13653e0ce6393536650cd2410bbbac7ff0ff0e73d999f06e99a213069f07ac91'; publisher='Mozilla Corporation|MZLA Technologies Corporation'}
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
            publisher=$signature.SignerCertificate.Subject; passed=$false}
        $result.packages += $row
        if ($signature.Status -ne 'Valid' -or $row.publisher -notmatch $package.publisher) {
            throw "Unexpected package signature: $($package.file)"
        }
        $log = Join-Path $root ($package.file + '.log')
        $arguments = '/i "' + $target + '" /qn /norestart /l*v "' + $log + '"'
        $process = Start-Process msiexec.exe -ArgumentList $arguments -PassThru
        $processHandle = $process.Handle
        $process.WaitForExit()
        $process.Refresh()
        $row.exit_code = $process.ExitCode
        $row.passed = $process.ExitCode -in @(0,3010)
        Copy-Item -LiteralPath $log -Destination (Join-Path $share ($package.file + '.install.log'))
        if (!$row.passed) { throw "Installation failed: $($package.file), exit $($process.ExitCode)" }
    }
    $result.passed = $true
} catch {
    $result.error = $_.Exception.ToString()
} finally {
    $result.finished = [DateTime]::UtcNow.ToString('o')
    $result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $resultFile -Encoding UTF8
}
