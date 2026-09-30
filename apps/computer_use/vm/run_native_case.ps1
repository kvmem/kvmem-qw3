# Setup-only driver for a real Windows-hosted client. Not exposed to the model.
param([switch]$HiddenWorker)
$ErrorActionPreference = 'Stop'
if (!$HiddenWorker) {
    # Leave the desktop to the task; a visible setup console must not remain
    # in front of the applications that the model is asked to operate.
    $worker = 'C:\Qw3Computer\native-case-worker.ps1'
    Copy-Item $PSCommandPath $worker -Force
    Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$worker,'-HiddenWorker')
    exit
}
$share = '\\host.lan\Data'
$app = $null
$result = @{kind='windows_native_client_real_model'; passed=$false; started=(Get-Date -Format o)}
try {
    $configuration = Get-Content "$share\native-case.json" -Raw | ConvertFrom-Json
    $result.run_id = $configuration.run_id
    $result.case = $configuration.case
    if ($configuration.run_id -notmatch '^[a-zA-Z0-9_-]{8,80}$') { throw 'Invalid native run ID' }
    if ($configuration.case -notin @('notepad','browser','cross_app')) { throw 'Unknown native task' }
    if ($configuration.backend -notmatch '^http://[0-9.]+:18084/v1$') { throw 'Unexpected VM model relay address' }
    $qualification = Get-Content "$share\native-client-tests.json" -Raw | ConvertFrom-Json
    if (!$qualification.passed -or $qualification.process_exit_code -ne 0) { throw 'Native client contract tests have not passed' }
    $expectedBundle = (Get-Content "$share\client-candidate.sha256" -Raw).Trim()
    if ($qualification.bundle_sha256 -ne $expectedBundle) { throw 'Native qualification belongs to another bundle' }
    $client = [string]$qualification.directory
    if (!$client.StartsWith('C:\Qw3Computer\clients\')) { throw 'Client path is outside the dedicated test root' }
    $python = Join-Path $client 'python\python.exe'
    if (!(Test-Path $python)) { throw 'Install and qualify the client after restoring this snapshot' }
    $runtime = Join-Path 'C:\Qw3Computer\native-runs' $configuration.run_id
    if (Test-Path $runtime) { throw 'Native run directory already exists' }
    New-Item -ItemType Directory -Force $runtime | Out-Null
    $result.runtime = $runtime
    $result.bundle_sha256 = $expectedBundle
    Copy-Item "$share\native-model-token.txt" "$runtime\model-token.txt"
    $env:PATH = "$client\codex\bin;$env:PATH"
    $env:PYTHONUTF8 = '1'
    # Detect a pre-existing application before starting another one.
    $occupied = $false
    try {
        Invoke-RestMethod 'http://127.0.0.1:18083/api/state' -TimeoutSec 2 | Out-Null
        $occupied = $true
    } catch {}
    if ($occupied) { throw 'Native application port is already occupied' }
    $arguments = '-X utf8 -m qw3_computer.app --desktop http://127.0.0.1:5000' +
        ' --desktop-token C:\Qw3Computer\executor-token.txt --backend ' + $configuration.backend +
        ' --backend-token "' + $runtime + '\model-token.txt" --runtime "' + $runtime + '\app"'
    $app = Start-Process -FilePath $python -ArgumentList $arguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$runtime\app.stdout.log" -RedirectStandardError "$runtime\app.stderr.log"
    $appHandle = $app.Handle
    $result.application_pid = $app.Id
    $deadline = [DateTime]::UtcNow.AddSeconds(60)
    $ready = $false
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($app.HasExited) { throw 'Windows-hosted application exited during startup' }
        try {
            $state = Invoke-RestMethod 'http://127.0.0.1:18083/api/state' -TimeoutSec 2
            if ($state.ready) { $ready=$true; break }
        } catch {}
        Start-Sleep -Milliseconds 500
    }
    if (!$ready) { throw 'Windows-hosted application did not become ready' }
    # Keep a native HTTP/UI read in addition to the model task evidence.
    (Invoke-WebRequest 'http://127.0.0.1:18083/' -UseBasicParsing).Content |
        Set-Content "$runtime\served-ui.html" -Encoding UTF8
    $taskArguments = '-X utf8 "' + $client + '\app\windows_task.py" --case ' + $configuration.case +
        ' --timeout 300 --run-dir "' + $runtime + '\task"'
    $driver = Start-Process -FilePath $python -ArgumentList $taskArguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$runtime\task.stdout.log" -RedirectStandardError "$runtime\task.stderr.log"
    $driverHandle = $driver.Handle
    @{stage='task_running'; run_id=$configuration.run_id; application_pid=$app.Id; driver_pid=$driver.Id} |
        ConvertTo-Json | Set-Content "$share\native-case-progress.json" -Encoding UTF8
    if (!$driver.WaitForExit(330000)) {
        Stop-Process -Id $driver.Id -Force
        throw 'Native task driver exceeded its deadline'
    }
    $driver.Refresh()
    $result.driver_exit_code = $driver.ExitCode
    if ($driver.ExitCode -ne 0) { throw "Native task driver failed: $($driver.ExitCode)" }
    $taskResult = Get-Content "$runtime\task\result.json" -Raw -Encoding UTF8 | ConvertFrom-Json
    $result.task = $taskResult
    $result.passed = $taskResult.passed
    # Render the actual local UI in Windows Edge after the task. A separate
    # headless profile avoids changing the desktop that the agent just verified.
    $edge = "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
    $uiArguments = '--headless=new --disable-gpu --no-first-run --no-default-browser-check' +
        ' --window-size=1280,1100 --virtual-time-budget=3000 --screenshot="' + $runtime + '\native-ui.png"' +
        ' --user-data-dir="' + $runtime + '\ui-profile" http://127.0.0.1:18083/'
    $ui = Start-Process $edge -ArgumentList $uiArguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$runtime\ui.stdout.log" -RedirectStandardError "$runtime\ui.stderr.log"
    $uiHandle = $ui.Handle
    if (!$ui.WaitForExit(20000)) {
        Stop-Process -Id $ui.Id -Force
        throw 'Native UI render exceeded its deadline'
    }
    if (!(Test-Path "$runtime\native-ui.png")) { throw 'Windows Edge did not produce the UI screenshot' }
    $result.ui_screenshot = 'native-ui.png'
} catch {
    $result.passed = $false
    $result.error = $_.Exception.ToString()
} finally {
    if ($null -ne $app -and !$app.HasExited) {
        try {
            $state = Invoke-RestMethod 'http://127.0.0.1:18083/api/state' -TimeoutSec 2
            if ($state.status -in @('starting','running','verifying')) {
                Invoke-RestMethod 'http://127.0.0.1:18083/api/control' -Method Post -ContentType 'application/json' `
                    -Body '{"command":"stop"}' -TimeoutSec 10 | Out-Null
            }
        } catch {}
        # Only the process tree started above belongs to this driver.
        try {
            $killer = Start-Process taskkill.exe -ArgumentList @('/PID',$app.Id,'/T','/F') -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput "$runtime\shutdown.stdout.log" -RedirectStandardError "$runtime\shutdown.stderr.log"
            $killerHandle = $killer.Handle
            if (!$killer.WaitForExit(10000)) { throw 'Application tree shutdown exceeded its deadline' }
            $killer.Refresh()
            $result.shutdown_exit_code = $killer.ExitCode
            if (!$app.WaitForExit(10000)) { throw 'Owned application survived task cleanup' }
        } catch {
            $result.cleanup_error = $_.Exception.ToString()
            $result.passed = $false
        }
    }
    $result.finished = Get-Date -Format o
    if ($result.runtime) {
        $export = Join-Path "$share\native-results" $result.run_id
        New-Item -ItemType Directory -Force $export | Out-Null
        try { Copy-Item "$runtime\*" $export -Recurse -Force }
        catch { $result.export_error = $_.Exception.ToString(); $result.passed = $false }
        $result | ConvertTo-Json -Depth 20 | Set-Content "$export\result.json" -Encoding UTF8
    }
    $result | ConvertTo-Json -Depth 20 | Set-Content "$share\native-case-result.json" -Encoding UTF8
}
