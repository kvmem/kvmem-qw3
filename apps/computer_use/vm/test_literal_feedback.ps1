# Qualification fixture only; desktop/backend inputs are synthetic.
# Launching this script through the reviewed console helper is administrative input.
$ErrorActionPreference = 'Stop'
$prefix = '__PREFIX__'
$share = '\\host.lan\Data'
$expectedPayload = '__PAYLOAD_SHA__'
$runtimeArchive = '__RUNTIME_SHA__'
$client = Join-Path 'C:\Qw3Computer\clients' $runtimeArchive.Substring(0,12)
$root = Join-Path 'C:\Qw3Computer\qualifications' $expectedPayload.Substring(0,16)
$result = @{kind='windows_native_literal_feedback_transport'; passed=$false; started=(Get-Date -Format o); qw3_model_calls=0; fixture_device_inputs=0; business_success_claimed=$false; administrative_console_launch=$true}
try {
    $payload = Join-Path $share "$prefix.zip"
    if ((Get-FileHash $payload -Algorithm SHA256).Hash.ToLowerInvariant() -cne $expectedPayload) {throw 'Payload digest mismatch'}
    if (Test-Path $root) {throw 'Fresh qualification directory required'}
    New-Item -ItemType Directory $root | Out-Null
    Expand-Archive -LiteralPath $payload -DestinationPath $root
    $fixture = Get-Content (Join-Path $root 'fixture.json') -Raw | ConvertFrom-Json
    foreach ($property in $fixture.runtime_files.PSObject.Properties) {
        if ((Get-FileHash (Join-Path $client $property.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -cne $property.Value) {throw "Native runtime source changed: $($property.Name)"}
    }
    foreach ($property in $fixture.payload_files.PSObject.Properties) {
        if ((Get-FileHash (Join-Path $root $property.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -cne $property.Value) {throw "Qualification source changed: $($property.Name)"}
    }
    $env:PATH = "$client\codex\bin;$env:PATH"
    $env:PYTHONUTF8 = '1'
    $python = Join-Path $client 'python\python.exe'
    $result.python = [string](& $python --version)
    $result.codex = [string](& "$client\codex\bin\codex.exe" --version)
    $result.runtime_files_verified = @($fixture.runtime_files.PSObject.Properties).Count
    $result.payload_sha256 = $expectedPayload
    $result.qualified_runtime_archive_sha256 = $runtimeArchive
    $output = Join-Path $root 'evidence'
    $arguments = '-X utf8 "' + $root + '\literal_completion_feedback_qualification.py" --candidate "' + $root + '\candidate" --output "' + $output + '" --fixture-tests "' + $client + '\app\tests" --build-sha256 ' + $fixture.candidate_build_sha256
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$share\$prefix.stdout.log" `
        -RedirectStandardError "$share\$prefix.stderr.log"
    $nativeHandle = $process.Handle
    $result.process_id = $process.Id
    @{stage='running'; process_id=$process.Id; root=$root; started=(Get-Date -Format o)} |
        ConvertTo-Json | Set-Content "$share\$prefix-progress.json" -Encoding UTF8
    if (!$process.WaitForExit(240000)) {throw 'Observation deadline; preserve the process for inspection'}
    $process.Refresh()
    $result.process_exit_code = $process.ExitCode
    if (Test-Path "$output\result.json") {
        $report = Get-Content "$output\result.json" -Raw | ConvertFrom-Json
        $result.transport_checks_passed = $report.checks_passed
        Copy-Item "$output\result.json" "$share\$prefix-transport.json"
    }
    if (Test-Path $output) {Compress-Archive -Path "$output\*" -DestinationPath "$share\$prefix-evidence.zip"}
    $result.passed = ($process.ExitCode -eq 0 -and $report.passed -and $report.runtime_platform -ceq 'win32' -and $report.checks_passed -eq 36)
} catch {
    $result.error = $_.Exception.ToString()
} finally {
    $result.finished = Get-Date -Format o
    $result | ConvertTo-Json -Depth 6 | Set-Content "$share\$prefix-result.json" -Encoding UTF8
}
