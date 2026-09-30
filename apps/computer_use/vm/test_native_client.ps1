# Setup-only: run in the dedicated test VM after the baseline suite has ended.
$ErrorActionPreference = 'Stop'
$share = '\\host.lan\Data'
$archive = Join-Path $share 'client-candidate.zip'
$expected = (Get-Content (Join-Path $share 'client-candidate.sha256') -Raw).Trim()
$actual = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw 'Native client bundle checksum mismatch' }
$destination = Join-Path 'C:\Qw3Computer\clients' $actual.Substring(0,12)
if (!(Test-Path (Join-Path $destination 'manifest.json'))) {
    New-Item -ItemType Directory -Force $destination | Out-Null
    Expand-Archive -Path $archive -DestinationPath $destination -Force
}
$env:PATH = "$destination\codex\bin;$env:PATH"
$env:PYTHONUTF8 = '1'
$python = Join-Path $destination 'python\python.exe'
$codexVersion = & "$destination\codex\bin\codex.exe" --version
$pythonVersion = & $python --version
$started = Get-Date -Format o
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$destination\test.ps1" *> "$share\native-client-tests.log"
$code = $LASTEXITCODE
@{
    kind = 'windows_native_client_contract_tests'
    directory = $destination
    bundle_sha256 = $actual
    codex = [string]$codexVersion
    python = [string]$pythonVersion
    process_exit_code = $code
    passed = ($code -eq 0)
    started = $started
    finished = (Get-Date -Format o)
} | ConvertTo-Json -Depth 5 | Set-Content "$share\native-client-tests.json" -Encoding UTF8
exit $code
