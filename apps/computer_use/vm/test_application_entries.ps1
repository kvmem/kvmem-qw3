# Run the actual entry-resolution functions without starting another executor.
$ErrorActionPreference='Stop'
$share='\\host.lan\Data'
$root=Join-Path $env:TEMP ('qw3-entry-tests-'+[guid]::NewGuid().ToString('N'))
$outputs=Join-Path $root 'outputs'
$result=@{kind='native_application_entry_contracts';passed=$false;cases=@()}
try {
    New-Item -ItemType Directory -Path $root,$outputs | Out-Null
    $tokens=$null; $errors=$null
    $ast=[Management.Automation.Language.Parser]::ParseFile('C:\Qw3Computer\executor.ps1',[ref]$tokens,[ref]$errors)
    if ($errors.Count) {throw "Executor has parse errors: $errors"}
    foreach ($name in @('Hash-Bytes','Require','Properties','Application-Entries')) {
        $nodes=@($ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq $name},$false))
        if ($nodes.Count -ne 1) {throw "Missing or ambiguous function: $name"}
        Invoke-Expression $nodes[0].Extent.Text
    }
    $entry=Application-Entries
    if (!$entry.legacy -or $entry.explorer_path -cne $outputs -or $entry.browser_urls[0] -cne 'file:///C:/Qw3Computer/fixture.html') {throw 'Legacy entry changed'}
    $result.cases+=@{name='legacy_without_configuration';passed=$true}
    $valid=@{version=1;run='entry-native-01';task='038';browser_profile="$root\OsworldBrowserProfile";
        browser_urls=@('http://cloudcrm.qw3.test:18090/?cookie=entry-native-01','http://teamchat.qw3.test:18090/?cookie=entry-native-01','http://mailhub.qw3.test:18090/?cookie=entry-native-01');
        explorer_path="$env:USERPROFILE\Desktop"}
    $path=Join-Path $root 'application-entries.json'
    $valid | ConvertTo-Json | Set-Content $path -Encoding UTF8
    $entry=Application-Entries
    if ($entry.legacy -or $entry.browser_urls.Count -ne 3 -or $entry.run -cne $valid.run -or $entry.sha256 -cne (Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant()) {throw 'Current task configuration lost'}
    $result.cases+=@{name='three_sites_and_exact_hash';passed=$true}
    $valid.task='022'
    $valid.browser_profile="$root\LoanTasks\$($valid.run)\BrowserProfile"
    $valid.browser_urls=@('http://localhost:8081/dashboard.html?cookie=local')
    $valid | ConvertTo-Json | Set-Content $path -Encoding UTF8
    $entry=Application-Entries
    if ($entry.browser_profile -cne $valid.browser_profile -or $entry.browser_urls.Count -ne 1) {throw 'Loan profile changed'}
    $result.cases+=@{name='single_loopback_site';passed=$true}
    foreach ($mapping in @(@{task='035';folder='PurchaseTasks'},@{task='043';folder='CalendarTasks'})) {
        $copy=$valid.Clone();$copy.task=$mapping.task;$copy.browser_profile="$root\$($mapping.folder)\$($valid.run)\BrowserProfile"
        $copy | ConvertTo-Json | Set-Content $path -Encoding UTF8
        $entry=Application-Entries
        if ($entry.browser_profile -cne $copy.browser_profile) {throw 'Task-specific profile changed'}
        $result.cases+=@{name=($mapping.folder+'_profile');passed=$true}
    }
    $bad=@(
        @{name='external_url';field='browser_urls';value=@('https://example.com/')},
        @{name='browser_argument_injection';field='browser_urls';value=@('http://localhost:8081/ --disable-web-security')},
        @{name='network_share_profile';field='browser_profile';value='\\host.lan\Data\profile'},
        @{name='other_attempt_profile';field='browser_profile';value="$root\LoanTasks\entry-other-01\BrowserProfile"},
        @{name='profile_path_traversal';field='browser_profile';value="$root\LoanTasks\$($valid.run)\..\BrowserProfile"},
        @{name='other_explorer_root';field='explorer_path';value='C:\Windows'},
        @{name='unknown_task';field='task';value='999'},
        @{name='non_array_urls';field='browser_urls';value='about:blank'},
        @{name='empty_urls';field='browser_urls';value=@()},
        @{name='unknown_property';field='executable';value='cmd.exe'}
    )
    foreach ($case in $bad) {
        $copy=$valid.Clone();$copy[$case.field]=$case.value
        $copy | ConvertTo-Json | Set-Content $path -Encoding UTF8
        $rejected=$false
        try {$ignored=Application-Entries} catch {$rejected=$true}
        if (!$rejected) {throw "Accepted invalid entry: $($case.name)"}
        $result.cases+=@{name=$case.name;passed=$true}
    }
    $result.executor_sha256=(Get-FileHash 'C:\Qw3Computer\executor.ps1' -Algorithm SHA256).Hash.ToLowerInvariant()
    $result.passed=$true
} catch {$result.error=$_.Exception.ToString()}
finally {
    if (Test-Path $root) {Remove-Item -LiteralPath $root -Recurse -Force}
    $result | ConvertTo-Json -Depth 8 | Set-Content "$share\application-entry-contracts.json" -Encoding UTF8
}
