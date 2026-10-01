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

