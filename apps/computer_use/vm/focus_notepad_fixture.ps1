param([long]$WindowHandle = 0)
# Setup-only focus preparation in the dedicated VM; never a Computer MCP tool.
$ErrorActionPreference = 'Stop'
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class InputFixtureFocus {
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
}
'@
if ($WindowHandle -eq 0 -and !(Get-Process notepad -ErrorAction SilentlyContinue)) { Start-Process notepad.exe }
$shell = New-Object -ComObject WScript.Shell
$watch = [Diagnostics.Stopwatch]::StartNew()
$passed = $false
while ($watch.Elapsed.TotalSeconds -lt 10) {
    $targets = @(Get-Process notepad -ErrorAction SilentlyContinue | Where-Object {
        $_.MainWindowHandle -ne 0 -and ($WindowHandle -eq 0 -or $_.MainWindowHandle.ToInt64() -eq $WindowHandle)
    })
    foreach ($target in $targets) {
        $shell.AppActivate($target.Id) | Out-Null
        Start-Sleep -Milliseconds 200
        if ([InputFixtureFocus]::GetForegroundWindow() -eq $target.MainWindowHandle) {
            $passed = $true
            break
        }
    }
    if ($passed) { break }
    Start-Sleep -Milliseconds 200
}
@{kind='dedicated_notepad_focus_setup'; passed=$passed; process_id=if($passed){$target.Id}else{0}} |
    ConvertTo-Json | Set-Content '\\host.lan\Data\input-focus-ready.json' -Encoding UTF8
if (!$passed) { throw 'Notepad fixture did not gain foreground' }
