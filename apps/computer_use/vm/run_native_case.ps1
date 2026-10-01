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
# Read-only process accounting. PID reuse must not attach old system processes.
function Get-ClientBirthTicks([DateTime]$Value) {
    # Win32_Process CreationDate is represented at microsecond precision.
    $ticks = $Value.ToUniversalTime().Ticks
    return [long]($ticks - ($ticks % 10))
}
function Select-ClientProcessTree($Entries, [int]$RootId, [DateTime]$RootStarted) {
    $expected = Get-ClientBirthTicks $RootStarted
    $byId = @{}
    foreach ($entry in $Entries) {
        $processId = [int]$entry.ProcessId
        if ($byId.ContainsKey($processId)) { throw "Duplicate process ID in snapshot: $processId" }
        $byId[$processId] = $entry
    }
    if (!$byId.ContainsKey($RootId)) { throw 'Client root disappeared before process snapshot' }
    $root = $byId[$RootId]
    if (!$root.CreationDate -or (Get-ClientBirthTicks $root.CreationDate) -ne $expected) {
        throw 'Client root PID was reused or its creation identity is unavailable'
    }
    $owned = @{}
    $owned[$RootId] = @{pid=$RootId; parent_pid=[int]$root.ParentProcessId;
        start_time_utc=$root.CreationDate.ToUniversalTime().ToString('o'); birth_ticks=[string]$expected;
        parent_start_time_utc=$null; parent_birth_ticks=$null}
    $rejected = @{}
    do {
        $changed = $false
        foreach ($entry in $Entries) {
            $processId = [int]$entry.ProcessId
            $parentId = [int]$entry.ParentProcessId
            if ($owned.ContainsKey($processId) -or !$owned.ContainsKey($parentId)) { continue }
            if (!$entry.CreationDate) {
                $rejected[$processId] = @{pid=$processId; parent_pid=$parentId; reason='creation_time_unavailable'}
                continue
            }
            $birth = Get-ClientBirthTicks $entry.CreationDate
            $parent = $owned[$parentId]
            if ($birth -lt [long]$parent.birth_ticks) {
                $rejected[$processId] = @{pid=$processId; parent_pid=$parentId; reason='child_predates_current_parent';
                    birth_ticks=[string]$birth; parent_birth_ticks=$parent.birth_ticks}
                continue
            }
            $owned[$processId] = @{pid=$processId; parent_pid=$parentId;
                start_time_utc=$entry.CreationDate.ToUniversalTime().ToString('o'); birth_ticks=[string]$birth;
                parent_start_time_utc=$parent.start_time_utc; parent_birth_ticks=$parent.birth_ticks}
            $changed = $true
        }
    } while ($changed)
    return @{nodes=@($owned.Values | Sort-Object pid); rejected_edges=@($rejected.Values | Sort-Object pid)}
}
function Get-ClientResourceSample($Application) {
    $row = @{schema_version=2; time_utc=[DateTime]::UtcNow.ToString('o'); app_pid=$Application.Id;
        ownership_verified=$false; snapshot_complete=$false; processes=@(); omitted_processes=@(); rejected_edges=@()}
    try {
        $rootStarted = $Application.StartTime.ToUniversalTime()
        $row.app_start_time_utc = $rootStarted.ToString('o')
        $row.app_birth_ticks = [string](Get-ClientBirthTicks $rootStarted)
        $entries = @(Get-CimInstance Win32_Process -OperationTimeoutSec 2)
        $tree = Select-ClientProcessTree $entries $Application.Id $rootStarted
        $row.rejected_edges = $tree.rejected_edges
        foreach ($node in $tree.nodes) {
            $process = $null
            try {
                $process = Get-Process -Id $node.pid -ErrorAction Stop
                # Retain this process instance while reading its identity/counters.
                $processHandle = $process.Handle
                $started = $process.StartTime.ToUniversalTime()
                if ((Get-ClientBirthTicks $started) -ne [long]$node.birth_ticks) {
                    throw 'Process PID changed identity after the CIM snapshot'
                }
                $row.processes += @{pid=$node.pid; parent_pid=$node.parent_pid; name=$process.ProcessName;
                    start_time_utc=$node.start_time_utc; sampled_start_time_utc=$started.ToString('o');
                    birth_ticks=$node.birth_ticks; parent_start_time_utc=$node.parent_start_time_utc;
                    parent_birth_ticks=$node.parent_birth_ticks; cpu_seconds=$process.TotalProcessorTime.TotalSeconds;
                    working_set_bytes=$process.WorkingSet64; peak_working_set_bytes=$process.PeakWorkingSet64;
                    private_bytes=$process.PrivateMemorySize64}
            } catch {
                $row.omitted_processes += @{pid=$node.pid; birth_ticks=$node.birth_ticks; reason=$_.Exception.Message}
            } finally {
                if ($process) { $process.Dispose() }
            }
        }
        $rootSample = @($row.processes | Where-Object { $_.pid -eq $Application.Id })
        if ($rootSample.Count -ne 1) { throw 'Root process identity was not retained through counter collection' }
        $row.ownership_verified = $true
        $unknown = @($row.rejected_edges | Where-Object { $_.reason -eq 'creation_time_unavailable' })
        $row.snapshot_complete = $row.omitted_processes.Count -eq 0 -and $unknown.Count -eq 0
    } catch {
        $row.error = $_.Exception.Message
        # Partial rows are diagnostic only, never silently valid memory totals.
        $row.ownership_verified = $false
        $row.snapshot_complete = $false
    }
    return $row
}

function Sample-ClientProcesses {
    # Sums are shared-page-inclusive; per-process identities/counters are retained.
    $row = Get-ClientResourceSample $app
    $row | ConvertTo-Json -Compress -Depth 8 | Add-Content "$runtime\client-resources.jsonl" -Encoding UTF8
}

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
    $taskDeadline = [DateTime]::UtcNow.AddSeconds(330)
    Sample-ClientProcesses
    while (!$driver.WaitForExit(1000)) {
        Sample-ClientProcesses
        if ([DateTime]::UtcNow -ge $taskDeadline) {
            Stop-Process -Id $driver.Id -Force
            throw 'Native task driver exceeded its deadline'
        }
    }
    Sample-ClientProcesses
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
