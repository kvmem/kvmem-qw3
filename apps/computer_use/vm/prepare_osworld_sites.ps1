# Only the dedicated VM's synthetic benchmark hostnames are added.
$ErrorActionPreference = 'Stop'
$result = @{kind='windows_private_site_connectivity'; passed=$false}
try {
$hostsFile = "$env:SystemRoot\System32\drivers\etc\hosts"
$names = @('formcraft.qw3.test','teamchat.qw3.test','cloudcrm.qw3.test','calendar.qw3.test','mailhub.qw3.test')
$text = [IO.File]::ReadAllText($hostsFile)
foreach ($name in $names) {
    $existing = @($text -split '\r?\n' | Where-Object { $_ -notmatch '^\s*#' -and $_ -match ('\b' + [regex]::Escape($name) + '(\s|$)') })
    if ($existing.Count -and @($existing | Where-Object { $_ -notmatch '^172\.17\.0\.1\s+' }).Count) {
        throw "Existing test hostname maps elsewhere: $name"
    }
    if (!$existing.Count) { Add-Content -LiteralPath $hostsFile -Value "172.17.0.1 $name # QW3 OSWorld test" -Encoding ASCII }
}
$results = @()
foreach ($name in $names) {
    $url = "http://${name}:18090/"
    $request = [Net.HttpWebRequest]::Create($url)
    $request.Proxy = $null
    $request.Timeout = 15000
    try {
        $response = $request.GetResponse()
        $results += @{url=$url; status=[int]$response.StatusCode}
        $response.Close()
    } catch { $results += @{url=$url; error=$_.Exception.Message} }
}

$result.sites=$results
$result.passed=(@($results | Where-Object {$_.status -ne 200}).Count -eq 0)
} catch {$result.error=$_.Exception.ToString()}
$result.recorded_at=[DateTime]::UtcNow.ToString('o')
$result | ConvertTo-Json -Depth 5 | Set-Content '\\host.lan\Data\osworld-sites-connectivity-02.json' -Encoding UTF8
