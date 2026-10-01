param([string]$RunId='baseline-clean-audit-01')
$ErrorActionPreference='Stop'
$result=@{kind='clean_windows_base_audit';passed=$false;formal_run=$false;run=$RunId;checks=@{}}
try {
 if ($RunId -cnotmatch '^baseline-clean-audit-[0-9]{2}$' -or $env:USERNAME -cne 'Qw3Test') {throw 'Invalid audit scope'}
 $allowed=@{
   "$env:USERPROFILE\Desktop"=@('Shared','desktop.ini','WPS Office.lnk','WPS PDF.lnk')
   "$env:USERPROFILE\Downloads"=@('desktop.ini')
   "$env:USERPROFILE\Documents"=@('My Music','My Pictures','My Videos','desktop.ini','KingsoftData','WPSDrive')
   'C:\Qw3Computer'=@('actions','clients','installers','outputs','EmptyMailProfile','bootstrap-ready.txt','bootstrap.ps1','desktop-id.txt','executor-ready.txt','executor-token.txt','executor.log','executor.ps1','fixture.html')
 }
 $result.roots=@()
 foreach ($root in $allowed.Keys) {
   $children=@(Get-ChildItem -LiteralPath $root -Force)
   $unexpected=@($children | Where-Object {$_.Name -notin $allowed[$root]})
   $result.roots+=@{path=$root;names=@($children.Name);unexpected=@($unexpected | ForEach-Object {$_.Name})}
   if ($unexpected.Count) {throw "Unexpected pre-existing task data in $root"}
 }
 foreach ($root in @('C:\Qw3Computer\outputs',"$env:USERPROFILE\Documents\KingsoftData","$env:USERPROFILE\Documents\WPSDrive")) {
   if ((Test-Path $root) -and @(Get-ChildItem $root -Recurse -Force -File).Count) {throw "Nonempty task output/data directory: $root"}
 }
 $result.checks.task_data_absent=$true
 $emptyTemps=@(Get-ChildItem "$env:LOCALAPPDATA\Temp" -File | Where-Object {$_.Name -match '^[0-9a-f-]{36}\.tmp$'})
 $result.guid_temporary_files=@($emptyTemps | Select-Object Name,Length)
 if (@($emptyTemps | Where-Object {$_.Length -ne 0}).Count) {throw 'A retained GUID temp file now has content'}
 $result.checks.retained_guid_temps_empty=$true
 $profile='C:\Qw3Computer\EmptyMailProfile'
 $mail=[IO.File]::ReadAllText("$env:APPDATA\Thunderbird\installs.ini")
 if (!$mail.Contains("Default=$profile") -or (Test-Path "$profile\Mail")) {throw 'Default mailbox is not empty'}
 $result.checks.empty_registered_mailbox=$true
 $client='C:\Qw3Computer\clients\482d558bc3e4'
 $manifest=Get-Content "$client\manifest.json" -Raw | ConvertFrom-Json
 foreach ($entry in $manifest.source_files.PSObject.Properties) {
   $file=Join-Path "$client\app" $entry.Name
   if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -cne $entry.Value) {throw "Installed client source differs: $($entry.Name)"}
 }
 $result.checks.qualified_client_source_hashes=$true
 $result.client_manifest_sha256=(Get-FileHash "$client\manifest.json" -Algorithm SHA256).Hash.ToLowerInvariant()
 $result.timezone=(Get-TimeZone).Id
 if ($result.timezone -cne 'UTC') {throw 'Guest timezone changed'}
 $token=(Get-Content 'C:\Qw3Computer\executor-token.txt' -Raw).Trim()
 $result.executor=Invoke-RestMethod 'http://127.0.0.1:5000/health' -Headers @{Authorization="Bearer $token"} -TimeoutSec 5
 $result.desktop_id=(Get-Content 'C:\Qw3Computer\desktop-id.txt' -Raw).Trim()
 $result.passed=$true
} catch {$result.error=$_.Exception.ToString()}
$result | ConvertTo-Json -Depth 8 | Set-Content "\\host.lan\Data\$RunId.json" -Encoding UTF8
