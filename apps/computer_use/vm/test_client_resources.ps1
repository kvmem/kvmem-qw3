param([string]$Source='\\host.lan\Data\client-resource-helpers.ps1', [string]$ReportName='client-resource-probe.json')
$ErrorActionPreference='Stop'
. $Source
$result=@{kind='windows_client_process_identity_probe'; passed=$false; checks=@();
    source_sha256=(Get-FileHash $Source -Algorithm SHA256).Hash.ToLowerInvariant()}
function Record([string]$Name,[bool]$Passed) {
    $script:result.checks+=@{name=$Name; passed=$Passed}
    if (!$Passed) { throw "Failed: $Name" }
}
function Entry([int]$Number,[int]$Parent,[DateTime]$Started) {
    return [pscustomobject]@{ProcessId=$Number; ParentProcessId=$Parent; CreationDate=$Started}
}
$child=$null
try {
    $t=[DateTime]::Parse('2026-10-01T00:00:00Z').ToUniversalTime()
    # PID676 used to own the old system subtree. It now belongs to this app.
    $entries=@((Entry 912 768 ($t.AddSeconds(-50))), (Entry 2001 2000 ($t.AddSeconds(2))),
        (Entry 692 676 ($t.AddSeconds(-70))), (Entry 768 692 ($t.AddSeconds(-60))),
        (Entry 2000 676 ($t.AddSeconds(1))), (Entry 676 5188 $t))
    $tree=Select-ClientProcessTree $entries 676 $t
    Record 'pid_reuse_excludes_old_system_branch' ((@($tree.nodes.pid) -join ',') -eq '676,2000,2001')
    Record 'old_parent_edge_has_reason' ($tree.rejected_edges.Count -eq 1 -and $tree.rejected_edges[0].reason -eq 'child_predates_current_parent')
    Record 'descendant_parent_birth_retained' ($tree.nodes[2].parent_birth_ticks -eq $tree.nodes[1].birth_ticks)
    $refused=$false
    try { Select-ClientProcessTree $entries 676 ($t.AddSeconds(5)) | Out-Null } catch { $refused=$true }
    Record 'root_pid_reuse_refused' $refused
    $refused=$false
    try { Select-ClientProcessTree $entries 777 $t | Out-Null } catch { $refused=$true }
    Record 'missing_root_refused' $refused
    $unknown=@($entries)+@([pscustomobject]@{ProcessId=3000; ParentProcessId=676; CreationDate=$null})
    $tree=Select-ClientProcessTree $unknown 676 $t
    Record 'missing_child_creation_not_assumed' (@($tree.nodes | Where-Object {$_.pid -eq 3000}).Count -eq 0 -and @($tree.rejected_edges | Where-Object {$_.reason -eq 'creation_time_unavailable'}).Count -eq 1)
    $self=Get-Process -Id $PID
    $before=Get-ClientResourceSample $self
    Record 'actual_windows_root_identity' ($before.ownership_verified -and $before.snapshot_complete)
    $child=Start-Process powershell.exe -ArgumentList @('-NoProfile','-Command','Start-Sleep -Seconds 20') -WindowStyle Hidden -PassThru
    $childHandle=$child.Handle
    $sample=Get-ClientResourceSample $self
    $result.real_sample=$sample
    Record 'actual_windows_child_included' ($sample.ownership_verified -and $sample.snapshot_complete -and @($sample.processes | Where-Object {$_.pid -eq $child.Id}).Count -eq 1)
    Record 'actual_windows_child_birth_matches' ((@($sample.processes | Where-Object {$_.pid -eq $child.Id}))[0].birth_ticks -eq [string](Get-ClientBirthTicks $child.StartTime))
    Record 'actual_windows_only_current_descendants' (@($sample.processes | Where-Object {[long]$_.birth_ticks -lt [long]$sample.app_birth_ticks}).Count -eq 0)
    $result.passed=$true
} catch { $result.error=$_.Exception.ToString() }
finally {
    if ($child -and !$child.HasExited) { $child.Kill(); $child.WaitForExit() }
    $result.finished=Get-Date -Format o
    $result | ConvertTo-Json -Depth 12 | Set-Content (Join-Path '\\host.lan\Data' $ReportName) -Encoding UTF8
}
