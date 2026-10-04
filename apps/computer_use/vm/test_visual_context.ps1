# Administrative qualification only: the desktop and model are synthetic.
# Generated constants name a unique payload and a previously qualified runtime.
$ErrorActionPreference = 'Stop'
$prefix = '__PREFIX__'
$share = '\\host.lan\Data'
$expectedPayload = '__PAYLOAD_SHA__'
$runtimeArchive = '__RUNTIME_SHA__'
$client = Join-Path 'C:\Qw3Computer\clients' $runtimeArchive.Substring(0,12)
$root = Join-Path 'C:\Qw3Computer\qualifications' $expectedPayload.Substring(0,16)
$result = @{kind='windows_native_image_context_transport'; passed=$false; started=(Get-Date -Format o); real_qw3_model_calls=0; native_device_inputs=0; business_success_claimed=$false}
try {
    $payload = Join-Path $share "$prefix.zip"
    if ((Get-FileHash $payload -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedPayload) {throw 'Payload digest mismatch'}
    if (Test-Path $root) {throw 'Fresh qualification directory required'}
    New-Item -ItemType Directory $root | Out-Null
    Expand-Archive -Path $payload -DestinationPath $root
    $fixture = Get-Content (Join-Path $root 'fixture.json') -Raw | ConvertFrom-Json
    foreach ($property in $fixture.runtime_files.PSObject.Properties) {
        $path = Join-Path $client $property.Name
        if ((Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value) {throw "Native runtime changed: $($property.Name)"}
    }
    foreach ($property in $fixture.payload_files.PSObject.Properties) {
        $path = Join-Path $root $property.Name
        if ((Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value) {throw "Qualification source changed: $($property.Name)"}
    }
    $env:PATH = "$client\codex\bin;$env:PATH"
    $env:PYTHONUTF8 = '1'
    $python = Join-Path $client 'python\python.exe'
    $result.python = [string](& $python --version)
    $result.codex = [string](& "$client\codex\bin\codex.exe" --version)
    $result.runtime_files_verified = @($fixture.runtime_files.PSObject.Properties).Count
    $result.qualified_runtime_archive_sha256 = $runtimeArchive
    $result.payload_sha256 = $expectedPayload
    $output = Join-Path $root 'evidence'
    $arguments = '-X utf8 "' + $root + '\visual_context_qualification.py" --candidate "' + $root + '\candidate" --output "' + $output + '"'
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$share\$prefix.stdout.log" `
        -RedirectStandardError "$share\$prefix.stderr.log"
    $nativeHandle = $process.Handle
    $result.process_id = $process.Id
    @{stage='running'; process_id=$process.Id; root=$root; started=(Get-Date -Format o)} |
        ConvertTo-Json | Set-Content "$share\$prefix-progress.json" -Encoding UTF8
    if (!$process.WaitForExit(240000)) {throw 'Bounded native qualification wait expired; process state retained for inspection'}
    $process.Refresh()
    $result.process_exit_code = $process.ExitCode
    if (Test-Path "$output\result.json") {
        $report = Get-Content "$output\result.json" -Raw | ConvertFrom-Json
        $result.transport_checks_passed = @($report.platforms | ForEach-Object { $_.checks } | Where-Object {$_.passed}).Count
        Copy-Item "$output\result.json" "$share\$prefix-transport.json"
    }
    if (Test-Path $output) {Compress-Archive -Path "$output\*" -DestinationPath "$share\$prefix-evidence.zip"}
    $result.passed = ($process.ExitCode -eq 0 -and $report.passed -and $result.transport_checks_passed -eq 144)
} catch {
    $result.error = $_.Exception.ToString()
} finally {
    $result.finished = Get-Date -Format o
    $result | ConvertTo-Json -Depth 6 | Set-Content "$share\$prefix-result.json" -Encoding UTF8
}
