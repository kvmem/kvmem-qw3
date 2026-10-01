# Administrative preparation of the dedicated OSWorld VM after a disk backup.
param([string]$RunId, [string]$PlanSHA, [string]$HelperSHA)
$ErrorActionPreference='Stop'
if ($RunId -cnotmatch '^baseline-cleanup-[0-9]{2}$' -or $PlanSHA -cnotmatch '^[0-9a-f]{64}$' -or $HelperSHA -cnotmatch '^[0-9a-f]{64}$') {throw 'Invalid reviewed cleanup identity'}
$share='\\host.lan\Data'
$result=@{kind='windows_osworld_clean_baseline_preparation';passed=$false;run=$RunId;formal_run=$false;stopped_processes=@()}
try {
    $plan="$share\$RunId-plan.json"; $helper="$share\$RunId.py"
    if ((Get-FileHash $plan -Algorithm SHA256).Hash.ToLowerInvariant() -cne $PlanSHA -or
        (Get-FileHash $helper -Algorithm SHA256).Hash.ToLowerInvariant() -cne $HelperSHA) {throw 'Cleanup inputs changed'}
    if ($env:USERNAME -cne 'Qw3Test') {throw 'This helper belongs to the dedicated test account'}
    $occupied=$false
    try {Invoke-RestMethod 'http://127.0.0.1:18083/api/state' -TimeoutSec 2 | Out-Null;$occupied=$true} catch {}
    if ($occupied) {throw 'A native agent application is still running'}
    $token=(Get-Content 'C:\Qw3Computer\executor-token.txt' -Raw).Trim()
    $health=Invoke-RestMethod 'http://127.0.0.1:5000/health' -Headers @{Authorization="Bearer $token"} -TimeoutSec 5
    if (!$health.paused) {throw 'Executor must be paused before cleanup'}
    $mailRegistration="$env:APPDATA\Thunderbird\installs.ini"
    if (Test-Path $mailRegistration) {
        $installationIds=@([regex]::Matches([IO.File]::ReadAllText($mailRegistration),'(?m)^\[([A-F0-9]+)\]\s*$') | ForEach-Object {$_.Groups[1].Value})
    } else {
        # A preceding verified cleanup may have removed the old registration
        # before stopping on another path. Its ID can be recovered from that
        # preserved archive and included in this hash-checked, reviewed plan.
        $reviewed=Get-Content $plan -Raw | ConvertFrom-Json
        if (!$reviewed.recovered_mail_registration.archive_verified -or
            $reviewed.recovered_mail_registration.installation_id -cnotmatch '^[A-F0-9]{16}$') {throw 'No preserved Thunderbird registration'}
        $installationIds=@($reviewed.recovered_mail_registration.installation_id)
    }
    if ($installationIds.Count -ne 1) {throw 'Expected one Thunderbird installation identity'}
    # Prevent the browser from replacing a just-closed parent with a new
    # background process that keeps the default cache locked during archival.
    $edgePolicy='HKCU:\Software\Policies\Microsoft\Edge'
    New-Item -Path $edgePolicy -Force | Out-Null
    foreach ($name in @('StartupBoostEnabled','BackgroundModeEnabled')) {
        New-ItemProperty -Path $edgePolicy -Name $name -Value 0 -PropertyType DWord -Force | Out-Null
    }
    # Only applications installed for this owned VM and its native test clients.
    $prefixes=@('C:\Program Files\LibreOffice\','C:\Program Files\GIMP 3\','C:\Program Files\Mozilla Thunderbird\',
        'C:\Program Files (x86)\Microsoft\Edge\',"$env:LOCALAPPDATA\Kingsoft\",'C:\Qw3Computer\clients\')
    function Get-OwnedApplications {
      @(Get-CimInstance Win32_Process | Where-Object {
        $path=$_.ExecutablePath
        $match=$false
        if ($path) {
            foreach ($prefix in $prefixes) {if ($path.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) {$match=$true}}
            if ($_.Name -ieq 'notepad.exe' -and ($path -like 'C:\Program Files\WindowsApps\Microsoft.WindowsNotepad_*' -or $path -ieq 'C:\Windows\System32\notepad.exe')) {$match=$true}
        }
        $match
      })
    }
    $owned=@(Get-OwnedApplications)
    foreach ($row in $owned) {
        $process=Get-Process -Id $row.ProcessId -ErrorAction SilentlyContinue
        if ($process -and $process.MainWindowHandle -ne 0) {$process.CloseMainWindow() | Out-Null}
    }
    Start-Sleep -Seconds 3
    foreach ($row in $owned) {
        $current=Get-CimInstance Win32_Process -Filter "ProcessId=$($row.ProcessId)"
        if ($current -and $current.CreationDate -eq $row.CreationDate -and $current.ExecutablePath -eq $row.ExecutablePath) {
            Stop-Process -Id $row.ProcessId -Force
        }
        $result.stopped_processes+=@{id=$row.ProcessId;path=$row.ExecutablePath}
    }
    for ($attempt=0; $attempt -lt 4; $attempt++) {
        $remaining=@(Get-OwnedApplications)
        if (!$remaining.Count) {break}
        foreach ($row in $remaining) {
            $current=Get-CimInstance Win32_Process -Filter "ProcessId=$($row.ProcessId)"
            if ($current -and $current.CreationDate -eq $row.CreationDate -and $current.ExecutablePath -eq $row.ExecutablePath) {
                Stop-Process -Id $row.ProcessId -Force -ErrorAction SilentlyContinue
                $result.stopped_processes+=@{id=$row.ProcessId;path=$row.ExecutablePath}
            }
        }
        Start-Sleep -Seconds 1
    }
    if (@(Get-OwnedApplications).Count) {throw 'Owned applications keep restarting; do not archive changing caches'}
    Add-Type -AssemblyName System.Windows.Forms
    [Windows.Forms.Clipboard]::Clear()
    $client='C:\Qw3Computer\clients\482d558bc3e4'
    $arguments='-X utf8 "'+$helper+'" --plan "'+$plan+'" --output "'+$share+'\'+$RunId+'-export"'
    $worker=Start-Process "$client\python\python.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput "$share\$RunId.stdout.log" -RedirectStandardError "$share\$RunId.stderr.log"
    $handle=$worker.Handle
    @{kind='baseline_cleanup_live_worker';run=$RunId;pid=$worker.Id;started_utc=$worker.StartTime.ToUniversalTime().ToString('o')} |
        ConvertTo-Json | Set-Content "$share\$RunId-worker.json" -Encoding UTF8
    if (!$worker.WaitForExit(1800000)) {throw 'Cleanup worker exceeded deadline; inspect its live process before proceeding'}
    if ($worker.ExitCode -ne 0) {throw "Cleanup worker failed: $($worker.ExitCode)"}
    $archive=Get-Content "$share\$RunId-export\result.json" -Raw | ConvertFrom-Json
    if (!$archive.complete -or !$archive.archive_verified_before_removal) {throw 'Data preservation/removal incomplete'}
    # Export only the task account's recent-file keys before resetting them.
    $registry=@('Software\Microsoft\Windows\CurrentVersion\Explorer\ComDlg32',
        'Software\Microsoft\Windows\CurrentVersion\Explorer\RecentDocs',
        'Software\Microsoft\Windows\CurrentVersion\Explorer\TypedPaths',
        'Software\Microsoft\Windows\CurrentVersion\Explorer\RunMRU')
    foreach ($app in @('et','ofd','pdf','wpp','wps')) {$registry+="Software\Kingsoft\Office\6.0\$app\RecentFiles"}
    $registry+='Software\Kingsoft\Office\6.0\et\RecentFunction'
    $index=0
    foreach ($key in $registry) {
        if (Test-Path "HKCU:\$key") {
            & reg.exe export "HKCU\$key" "$share\$RunId-export\registry-$index.reg" /y | Out-Null
            if ($LASTEXITCODE -ne 0) {throw 'Recent-file registry export failed'}
            Remove-Item -LiteralPath "HKCU:\$key" -Recurse -Force
        }
        $index++
    }
    # Keep a valid, empty offline default registration for ordinary mailbox
    # reopen; scenario initializers replace it with their own original inbox.
    $profile='C:\Qw3Computer\EmptyMailProfile'
    New-Item -ItemType Directory -Path $profile -Force | Out-Null
    New-Item -ItemType Directory -Path "$env:APPDATA\Thunderbird" -Force | Out-Null
    [IO.File]::WriteAllText("$profile\prefs.js",'user_pref("offline.startup_state", 3);'+"`r`n"+'user_pref("network.online", false);')
    $id=$installationIds[0]
    [IO.File]::WriteAllText("$env:APPDATA\Thunderbird\profiles.ini","[General]`r`nStartWithLastProfile=1`r`nVersion=2`r`n`r`n[Profile0]`r`nName=qw3-empty-local`r`nIsRelative=0`r`nPath=$profile`r`nDefault=1`r`n`r`n[Install$id]`r`nDefault=$profile`r`nLocked=1`r`n")
    [IO.File]::WriteAllText("$env:APPDATA\Thunderbird\installs.ini","[$id]`r`nDefault=$profile`r`nLocked=1`r`n")
    New-Item -ItemType Directory -Path 'C:\Qw3Computer\actions','C:\Qw3Computer\outputs' -Force | Out-Null
    [guid]::NewGuid().ToString() | Set-Content 'C:\Qw3Computer\desktop-id.txt'
    $result.archive_sha256=$archive.archive_sha256
    $result.removed_paths=$archive.removed.Count
    $result.mailbox_installation_identity_preserved=$true
    $result.passed=$true
} catch {$result.error=$_.Exception.ToString()}
$result | ConvertTo-Json -Depth 7 | Set-Content "$share\$RunId-result.json" -Encoding UTF8
