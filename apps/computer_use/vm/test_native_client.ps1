# Setup-only: run in the dedicated test VM after the baseline suite has ended.
$ErrorActionPreference = 'Stop'
$share = '\\host.lan\Data'
$result = @{ kind='windows_native_client_contract_tests'; passed=$false; started=(Get-Date -Format o) }
try {
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
    $env:QW3_TEST_CODEX = '1'
    $python = Join-Path $destination 'python\python.exe'
    $result.directory = $destination
    $result.bundle_sha256 = $actual
    $result.codex = [string](& "$destination\codex\bin\codex.exe" --version)
    $result.python = [string](& $python --version)
    # unittest writes ordinary progress to stderr. Capture it as native output,
    # without promoting it into PowerShell's terminating error stream.
    $arguments = '-X utf8 -m unittest discover -s "' + $destination + '\app\tests" -v'
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$share\native-client-tests.stdout.log" `
        -RedirectStandardError "$share\native-client-tests.stderr.log"
    # Keep a process handle before it exits; Windows PowerShell 5 otherwise
    # can return a Process object whose ExitCode remains null after waiting.
    $nativeHandle = $process.Handle
    @{stage='running'; process_id=$process.Id; directory=$destination; started=(Get-Date -Format o)} |
        ConvertTo-Json | Set-Content "$share\native-client-tests-progress.json" -Encoding UTF8
    $process.WaitForExit()
    $process.Refresh()
    $result.process_exit_code = $process.ExitCode
    $result.passed = ($process.ExitCode -eq 0)
} catch {
    $result.error = $_.Exception.ToString()
} finally {
    $result.finished = Get-Date -Format o
    $result | ConvertTo-Json -Depth 5 | Set-Content "$share\native-client-tests.json" -Encoding UTF8
}
