# Setup-only workload installation after restoring the clean guest snapshot.
$ErrorActionPreference = 'Stop'
$share = '\\host.lan\Data'
$source = Join-Path $share 'long_fixture.html'
$target = 'C:\Qw3Computer\long_fixture.html'
Copy-Item $source $target -Force
@{path=$target; sha256=(Get-FileHash $target -Algorithm SHA256).Hash.ToLowerInvariant()} |
    ConvertTo-Json | Set-Content "$share\long-task-ready.json" -Encoding UTF8
