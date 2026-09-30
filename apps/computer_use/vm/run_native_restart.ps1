# Setup-only crash/resume driver; never exposed to the model.
param([switch]$HiddenWorker)
$ErrorActionPreference = 'Stop'
if (!$HiddenWorker) {
    $worker = 'C:\Qw3Computer\native-restart-worker.ps1'
    Copy-Item $PSCommandPath $worker -Force
    Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$worker,'-HiddenWorker')
    exit
}
$share = '\\host.lan\Data'
$result = @{kind='windows_native_crash_resume'; passed=$false; started=(Get-Date -Format o)}
try {
    $configuration = Get-Content "$share\native-case.json" -Raw | ConvertFrom-Json
    $result.run_id = $configuration.run_id
    $result.case = $configuration.case
    if ($configuration.run_id -notmatch '^[a-zA-Z0-9_-]{8,80}$') { throw 'Invalid native run ID' }
    if ($configuration.case -notin @('notepad','cross_app')) { throw 'Unsupported recovery task' }
    if ($configuration.backend -notmatch '^http://[0-9.]+:18084/v1$') { throw 'Unexpected VM relay address' }
    $qualification = Get-Content "$share\native-client-tests.json" -Raw | ConvertFrom-Json
    $expectedBundle = (Get-Content "$share\client-candidate.sha256" -Raw).Trim()
    if (!$qualification.passed -or $qualification.process_exit_code -ne 0 -or $qualification.bundle_sha256 -ne $expectedBundle) {
        throw 'The staged native client has not passed contract qualification'
    }
    $client = [string]$qualification.directory
    if (!$client.StartsWith('C:\Qw3Computer\clients\')) { throw 'Unexpected client directory' }
    $runtime = Join-Path 'C:\Qw3Computer\native-runs' $configuration.run_id
    if (Test-Path $runtime) { throw 'Native run directory already exists' }
    New-Item -ItemType Directory $runtime | Out-Null
    $result.runtime = $runtime
    $result.bundle_sha256 = $expectedBundle
    $result.application_runtime_relative = 'restart/app'
    Copy-Item "$share\native-model-token.txt" "$runtime\model-token.txt"
    Copy-Item "$share\native_restart.py" "$runtime\native_restart.py"
    $env:PATH = "$client\codex\bin;$env:PATH"
    $env:PYTHONUTF8 = '1'
    $arguments = '-X utf8 "' + $runtime + '\native_restart.py" --case ' + $configuration.case +
        ' --backend ' + $configuration.backend + ' --backend-token "' + $runtime + '\model-token.txt"' +
        ' --run-dir "' + $runtime + '\restart"'
    $driver = Start-Process "$client\python\python.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$runtime\driver.stdout.log" -RedirectStandardError "$runtime\driver.stderr.log"
    $driverHandle = $driver.Handle
    @{stage='recovery_running'; run_id=$configuration.run_id; driver_pid=$driver.Id} |
        ConvertTo-Json | Set-Content "$share\native-case-progress.json" -Encoding UTF8
    if (!$driver.WaitForExit(560000)) {
        # Kill only the driver and descendants created by this setup invocation.
        $killer = Start-Process taskkill.exe -ArgumentList @('/PID',$driver.Id,'/T','/F') -WindowStyle Hidden -PassThru
        $killer.WaitForExit(10000) | Out-Null
        throw 'Native recovery exceeded its deadline'
    }
    $driver.Refresh()
    $result.driver_exit_code = $driver.ExitCode
    if (Test-Path "$runtime\restart\result.json") {
        $result.task = Get-Content "$runtime\restart\result.json" -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    if ($driver.ExitCode -ne 0) { throw "Native recovery driver failed: $($driver.ExitCode)" }
    $result.passed = $result.task.passed
} catch {
    $result.passed = $false
    $result.error = $_.Exception.ToString()
} finally {
    $result.finished = Get-Date -Format o
    if ($result.runtime) {
        $export = Join-Path "$share\native-results" $result.run_id
        New-Item -ItemType Directory -Force $export | Out-Null
        try { Copy-Item "$runtime\*" $export -Recurse -Force }
        catch { $result.export_error=$_.Exception.ToString(); $result.passed=$false }
        $result | ConvertTo-Json -Depth 20 | Set-Content "$export\result.json" -Encoding UTF8
    }
    $result | ConvertTo-Json -Depth 20 | Set-Content "$share\native-case-result.json" -Encoding UTF8
}
