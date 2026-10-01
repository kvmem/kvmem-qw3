# Read-only inventory in the task-owned VM. Never exposed as an agent tool.
param([string]$Output = '\\host.lan\Data\osworld-app-inventory-01.json')
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Output) { throw "Inventory already exists: $Output" }
$apps = @(
    Get-ItemProperty 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*' -ErrorAction SilentlyContinue |
    Where-Object { $_.DisplayName -match 'LibreOffice|Thunderbird|GIMP|WPS Office' } |
    Select-Object DisplayName, DisplayVersion, Publisher, InstallLocation
)
$os = Get-CimInstance Win32_OperatingSystem
$disks = @(Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' |
    Select-Object DeviceID, Size, FreeSpace)
$paths = @(
    'C:\Program Files\LibreOffice\program\soffice.exe',
    'C:\Program Files\Mozilla Thunderbird\thunderbird.exe',
    'C:\Program Files\GIMP 3\bin\gimp-3.exe',
    'C:\Program Files\GIMP 2\bin\gimp-2.10.exe'
)
$known = @($paths | Where-Object { Test-Path -LiteralPath $_ } | ForEach-Object {
    $file = Get-Item -LiteralPath $_
    @{path=$file.FullName; version=$file.VersionInfo.FileVersion}
})
@{
    kind='read_only_windows_application_inventory'; recorded_at=[DateTime]::UtcNow.ToString('o')
    os=@{caption=$os.Caption; version=$os.Version; build=$os.BuildNumber; memory_kib=$os.TotalVisibleMemorySize}
    applications=$apps; known_executables=$known; disks=$disks
    winget_available=[bool](Get-Command winget.exe -ErrorAction SilentlyContinue)
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Output -Encoding UTF8
