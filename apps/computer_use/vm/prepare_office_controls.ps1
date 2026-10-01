param([ValidatePattern('^[a-zA-Z0-9_-]{1,40}$')][string]$Run = '01')
$ErrorActionPreference = 'Stop'
$root = 'C:\Qw3Computer'
Copy-Item '\\host.lan\Data\office-controls.html' "$root\office-controls.html" -Force
$edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
Start-Process -FilePath $edge -ArgumentList @('--no-first-run','--no-default-browser-check',
    '--disable-background-networking','--disable-component-update','--disable-sync',
    '--force-renderer-accessibility',"--user-data-dir=$root\OfficeControlsProfile",
    "--app=file:///C:/Qw3Computer/office-controls.html?run=$Run")
