# Generic task launcher in the owned Windows VM. Host grading remains separate.
param([string]$RunId, [switch]$HiddenWorker)
$ErrorActionPreference='Stop'
if ($RunId -cnotmatch '^[a-zA-Z0-9_-]{8,60}$') {throw 'Invalid native attempt identity'}
$share='\\host.lan\Data'
$public=Join-Path $share "$RunId-inputs"
$runtime=Join-Path 'C:\Qw3Computer\native-runs' $RunId
$export=Join-Path "$share\native-results" $RunId
if (!$HiddenWorker) {
    if (Test-Path $runtime) {throw 'Native attempt already exists; use a new identity'}
    New-Item -ItemType Directory -Path $runtime | Out-Null
    Copy-Item -LiteralPath $PSCommandPath -Destination "$runtime\worker.ps1"
    Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',"$runtime\worker.ps1",'-RunId',$RunId,'-HiddenWorker')
    exit
}
$app=$null; $driver=$null; $ownedReady=$false; $configuration=$null
$result=@{kind='windows_native_benchmark_task';run=$RunId;infrastructure_ok=$false;graded=$false;started_utc=[DateTime]::UtcNow.ToString('o')}
function Write-Progress([string]$Stage) {
    @{kind='native_task_progress';run=$RunId;stage=$Stage;guest_monotonic_ticks=[Diagnostics.Stopwatch]::GetTimestamp();
      guest_monotonic_frequency=[Diagnostics.Stopwatch]::Frequency;guest_unix_ms=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()} |
        ConvertTo-Json | Set-Content "$share\$RunId-progress.tmp" -Encoding UTF8
    Move-Item "$share\$RunId-progress.tmp" "$share\$RunId-progress.json" -Force
}
function Sample-Client {
    (Get-ClientResourceSample $app) | ConvertTo-Json -Compress -Depth 8 | Add-Content "$runtime\client-resources.jsonl" -Encoding UTF8
}
try {
    $configuration=Get-Content "$public\native-task.json" -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($configuration.run -cne $RunId -or $configuration.task -notin @('013','004','006','010','038','022','035','043','080','051','cap01','cap02','cap03','cap04','cap05','cap06','cap07','cap08','cap09','cap10','cap11','cap12')) {throw 'Task configuration identity mismatch'}
    if ($configuration.bundle_sha256 -cnotmatch '^[a-f0-9]{64}$' -or $configuration.backend -cnotmatch '^http://[0-9.]+:18084/v1$') {throw 'Invalid frozen client/backend identity'}
    if ($configuration.timeout_seconds -lt 1 -or $configuration.timeout_seconds -gt 14400 -or $configuration.tool_call_limit -lt 1 -or $configuration.tool_call_limit -gt 10000) {throw 'Invalid frozen task budget'}
    $modelWait=600; $modelReadWait=180
    $waitFields=@($configuration.PSObject.Properties.Name | Where-Object {$_ -in @('backend_timeout_seconds','backend_read_timeout_seconds')})
    if ($waitFields.Count -notin @(0,2)) {throw 'Incomplete model wait configuration'}
    if ($waitFields.Count -eq 2) {
        $modelWait=$configuration.backend_timeout_seconds; $modelReadWait=$configuration.backend_read_timeout_seconds
        if ($modelWait -is [bool] -or $modelReadWait -is [bool] -or "$modelWait" -cnotmatch '^[0-9]+$' -or "$modelReadWait" -cnotmatch '^[0-9]+$') {throw 'Model waits must be integer seconds'}
    }
    if ($modelReadWait -lt 30 -or $modelWait -lt $modelReadWait -or $modelWait -gt 14400) {throw 'Invalid model wait configuration'}
    $result.backend_timeout_seconds=$modelWait; $result.backend_read_timeout_seconds=$modelReadWait
    if ($configuration.review_batches -notin @(0,3) -or ($configuration.review_batches -eq 3 -and $configuration.task -ne '035')) {throw 'Unrecognized dynamic event plan'}
    $client=Join-Path 'C:\Qw3Computer\clients' $configuration.bundle_sha256.Substring(0,12)
    $python=Join-Path $client 'python\python.exe'
    if (!(Test-Path $python)) {throw 'Qualified native client is unavailable'}
    $manifest=Get-Content "$client\manifest.json" -Raw | ConvertFrom-Json
    # Verify actual installed application bytes before every trial. The archive
    # hash and native contract qualification are additionally checked by host.
    foreach ($entry in $manifest.source_files.PSObject.Properties) {
        if ((Get-FileHash (Join-Path "$client\app" $entry.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -cne $entry.Value) {throw "Installed client file changed: $($entry.Name)"}
    }
    foreach ($entry in $configuration.public_sha256.PSObject.Properties) {
        if ($entry.Name -notin @('prompt.txt','collection.json','client_resources.ps1')) {throw 'Unexpected launcher input'}
        if ((Get-FileHash (Join-Path $public $entry.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -cne $entry.Value) {throw 'Launcher input hash mismatch'}
        Copy-Item -LiteralPath (Join-Path $public $entry.Name) -Destination (Join-Path $runtime $entry.Name)
    }
    . "$runtime\client_resources.ps1"
    $occupied=$false
    try {Invoke-RestMethod 'http://127.0.0.1:18083/api/state' -TimeoutSec 2 | Out-Null;$occupied=$true} catch {}
    if ($occupied) {throw 'Another native application is using the task port'}
    Copy-Item "$share\native-model-token.txt" "$runtime\model-token.txt"
    $env:PATH="$client\codex\bin;$env:PATH"; $env:PYTHONUTF8='1'
    $appArguments='-X utf8 -m qw3_computer.app --desktop http://127.0.0.1:5000 --desktop-token C:\Qw3Computer\executor-token.txt'+
        ' --backend '+$configuration.backend+' --backend-token "'+$runtime+'\model-token.txt" --runtime "'+$runtime+'\app"'+
        ' --tool-call-limit '+$configuration.tool_call_limit+' --output-policy task --trace-backend'
    # Historical bundles keep their original default CLI. Extended waits
    # require a newly qualified bundle that supports these explicit options.
    if ($modelWait -ne 600 -or $modelReadWait -ne 180) {
        $appArguments+=' --backend-timeout '+$modelWait+' --backend-read-timeout '+$modelReadWait
    }
    $app=Start-Process $python -ArgumentList $appArguments -WindowStyle Hidden -PassThru -RedirectStandardOutput "$runtime\app.stdout.log" -RedirectStandardError "$runtime\app.stderr.log"
    $appHandle=$app.Handle
    $deadline=[DateTime]::UtcNow.AddSeconds(60)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($app.HasExited) {throw 'Native application exited at startup'}
        try {$state=Invoke-RestMethod 'http://127.0.0.1:18083/api/state' -TimeoutSec 2;if ($state.ready) {$ownedReady=$true;break}} catch {}
        Start-Sleep -Milliseconds 250
    }
    if (!$ownedReady) {throw 'Native application did not become ready'}
    $taskArguments='-X utf8 "'+$client+'\app\task_driver.py" --prompt "'+$runtime+'\prompt.txt" --output "'+$runtime+'\task" --timeout '+$configuration.timeout_seconds
    if ($configuration.review_batches) {$taskArguments+=' --review-mailbox "'+$share+'\'+$RunId+'-reviews" --review-batches '+$configuration.review_batches}
    Write-Progress 'task_starting'
    $driver=Start-Process $python -ArgumentList $taskArguments -WindowStyle Hidden -PassThru -RedirectStandardOutput "$runtime\task.stdout.log" -RedirectStandardError "$runtime\task.stderr.log"
    $driverHandle=$driver.Handle
    $deadline=[DateTime]::UtcNow.AddSeconds($configuration.timeout_seconds+45)
    do {
        Sample-Client
        if ([DateTime]::UtcNow -ge $deadline) {throw 'Native task driver exceeded cleanup deadline'}
    } while (!$driver.WaitForExit(1000))
    $driver.Refresh();$result.driver_exit_code=$driver.ExitCode
    $result.task=Get-Content "$runtime\task\result.json" -Raw -Encoding UTF8 | ConvertFrom-Json
    $result.infrastructure_ok=$driver.ExitCode -eq 0 -and $result.task.queue_may_continue
    Write-Progress 'task_terminal'
} catch {$result.error=$_.Exception.ToString();$result.infrastructure_ok=$false}
finally {
    # Stop only this launcher's application, even if the driver lost its reply.
    if ($app -and !$app.HasExited) {
        try {
            if ($ownedReady) {
                $state=Invoke-RestMethod 'http://127.0.0.1:18083/api/control' -Method Post -ContentType 'application/json' -Body '{"command":"stop"}' -TimeoutSec 15
                if (!$state.desktop_paused -or $state.status -ne 'stopped') {throw 'Desktop stop was not acknowledged'}
                $result.desktop_stopped=$true
            }
        } catch {$result.stop_error=$_.Exception.ToString();$result.infrastructure_ok=$false}
        try {
            $killer=Start-Process taskkill.exe -ArgumentList @('/PID',$app.Id,'/T','/F') -WindowStyle Hidden -PassThru -RedirectStandardOutput "$runtime\shutdown.stdout.log" -RedirectStandardError "$runtime\shutdown.stderr.log"
            $killerHandle=$killer.Handle
            if (!$killer.WaitForExit(10000) -or !$app.WaitForExit(10000)) {throw 'Native client shutdown failed'}
        } catch {$result.cleanup_error=$_.Exception.ToString();$result.infrastructure_ok=$false}
    }
    if ($driver -and !$driver.HasExited) {$driver.Kill();$driver.WaitForExit()}
    if (!$result.desktop_stopped) {$result.infrastructure_ok=$false}
    if ($result.desktop_stopped -and (Test-Path "$runtime\collection.json")) {
        try {
            $collection=Get-Content "$runtime\collection.json" -Raw | ConvertFrom-Json
            if ($collection.files.Count) {
                $collectorArguments='-X utf8 "'+$client+'\app\collect_artifacts.py" --specification "'+$runtime+'\collection.json" --output "'+$runtime+'\artifacts"'
                $collector=Start-Process $python -ArgumentList $collectorArguments -WindowStyle Hidden -PassThru -RedirectStandardOutput "$runtime\collection.stdout.log" -RedirectStandardError "$runtime\collection.stderr.log"
                $collectorHandle=$collector.Handle
                if (!$collector.WaitForExit(180000)) {$collector.Kill();throw 'Artifact collection timeout'}
                $collector.Refresh();$result.collector_exit_code=$collector.ExitCode
                if ($collector.ExitCode -ne 0) {throw 'Artifact collection failed'}
            }
        } catch {$result.collection_error=$_.Exception.ToString();$result.infrastructure_ok=$false}
    }
    try {
        if (Test-Path $export) {throw 'Native export path already exists'}
        New-Item -ItemType Directory -Path $export | Out-Null
        # Never publish the model credential into result artifacts.
        Get-ChildItem -LiteralPath $runtime | Where-Object {$_.Name -ne 'model-token.txt'} | Copy-Item -Destination $export -Recurse
    } catch {$result.export_error=$_.Exception.ToString();$result.infrastructure_ok=$false}
    $result.finished_utc=[DateTime]::UtcNow.ToString('o')
    $result | ConvertTo-Json -Depth 30 | Set-Content "$share\$RunId-result.tmp" -Encoding UTF8
    Move-Item "$share\$RunId-result.tmp" "$share\$RunId-result.json" -Force
}
