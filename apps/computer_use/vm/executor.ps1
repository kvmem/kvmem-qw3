param([int]$Port = 5000)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Collections.Generic;
public static class DesktopNative {
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, int data, UIntPtr extra);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr window, out uint process);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr window, StringBuilder text, int count);
    [StructLayout(LayoutKind.Sequential)] public struct Keyboard { public ushort vk, scan; public uint flags, time; public UIntPtr extra; }
    [StructLayout(LayoutKind.Sequential)] public struct Mouse { public int x, y; public uint data, flags, time; public UIntPtr extra; }
    [StructLayout(LayoutKind.Explicit)] public struct InputUnion {
        [FieldOffset(0)] public Keyboard keyboard;
        [FieldOffset(0)] public Mouse mouse;
    }
    [StructLayout(LayoutKind.Sequential)] public struct Input { public uint type; public InputUnion value; }
    [DllImport("user32.dll", SetLastError=true)] public static extern uint SendInput(uint count, Input[] inputs, int size);
    public static void TypeText(string text) {
        var events = new List<Input>();
        for (int i = 0; i < text.Length; i++) {
            char c = text[i];
            if (c == '\r' && i + 1 < text.Length && text[i+1] == '\n') continue;
            Input down = new Input(), up = new Input();
            down.type = up.type = 1;
            if (c == '\r' || c == '\n') {
                down.value.keyboard.vk = up.value.keyboard.vk = 0x0D;
                up.value.keyboard.flags = 2;
            } else {
                down.value.keyboard.scan = up.value.keyboard.scan = c;
                down.value.keyboard.flags = 4;
                up.value.keyboard.flags = 6;
            }
            events.Add(down); events.Add(up);
        }
        Input[] inputs = events.ToArray();
        if (SendInput((uint)inputs.Length, inputs, Marshal.SizeOf(typeof(Input))) != inputs.Length)
            throw new InvalidOperationException("SendInput did not deliver all input events");
    }
}
'@
[DesktopNative]::SetProcessDPIAware() | Out-Null
$root = 'C:\Qw3Computer'
$ledger = Join-Path $root 'actions'
$outputs = Join-Path $root 'outputs'
New-Item -ItemType Directory -Force $ledger,$outputs | Out-Null
$token = (Get-Content (Join-Path $root 'executor-token.txt') -Raw).Trim()
if ($token.Length -lt 32) { throw 'Executor token is too short' }
$idFile = Join-Path $root 'desktop-id.txt'
if (!(Test-Path $idFile)) { Set-Content $idFile ([guid]::NewGuid().ToString()) }
$desktopId = (Get-Content $idFile -Raw).Trim()
$script:latest = $null
$script:paused = $true
$script:noProgress = 0
$script:lastSignature = ''
$allowedApps = @('explorer','msedge','notepad','ApplicationFrameHost','StartMenuExperienceHost','SearchHost','ShellExperienceHost')
$keyMap = @{
    'ENTER'='{ENTER}'; 'TAB'='{TAB}'; 'SHIFT+TAB'='+{TAB}'; 'ESC'='{ESC}';
    'BACKSPACE'='{BACKSPACE}'; 'DELETE'='{DELETE}'; 'UP'='{UP}'; 'DOWN'='{DOWN}';
    'LEFT'='{LEFT}'; 'RIGHT'='{RIGHT}'; 'HOME'='{HOME}'; 'END'='{END}';
    'CTRL+A'='^a'; 'CTRL+C'='^c'; 'CTRL+V'='^v'; 'CTRL+X'='^x';
    'CTRL+S'='^s'; 'CTRL+SHIFT+S'='^+s'; 'CTRL+N'='^n'; 'CTRL+O'='^o';
    'CTRL+L'='^l'; 'CTRL+W'='^w'; 'CTRL+Z'='^z'; 'ALT+TAB'='%{TAB}'; 'ALT+F4'='%{F4}'
}

function Save-Json($Path, $Value) {
    $temp = "$Path.tmp"
    [IO.File]::WriteAllText($temp, ($Value | ConvertTo-Json -Depth 16 -Compress), [Text.UTF8Encoding]::new($false))
    Move-Item -Force $temp $Path
}
function Hash-Bytes([byte[]]$Bytes) {
    $hash = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hash.ComputeHash($Bytes))).Replace('-','').ToLowerInvariant() }
    finally { $hash.Dispose() }
}
function Foreground {
    $handle = [DesktopNative]::GetForegroundWindow()
    [uint32]$processId = 0
    [DesktopNative]::GetWindowThreadProcessId($handle, [ref]$processId) | Out-Null
    $title = [Text.StringBuilder]::new(1024)
    [DesktopNative]::GetWindowText($handle, $title, $title.Capacity) | Out-Null
    $name = ''
    if ($processId) { $name = (Get-Process -Id $processId -ErrorAction SilentlyContinue).ProcessName }
    return @{ handle=$handle.ToInt64(); process=$name; title=$title.ToString() }
}
function Observe {
    $bounds = [Windows.Forms.Screen]::PrimaryScreen.Bounds
    $bitmap = [Drawing.Bitmap]::new($bounds.Width, $bounds.Height)
    $graphics = [Drawing.Graphics]::FromImage($bitmap)
    $stream = [IO.MemoryStream]::new()
    try {
        $graphics.CopyFromScreen($bounds.Location, [Drawing.Point]::Empty, $bounds.Size)
        $bitmap.Save($stream, [Drawing.Imaging.ImageFormat]::Png)
        $bytes = $stream.ToArray()
        $script:latest = @{ desktop_id=$desktopId; observation_id=[guid]::NewGuid().ToString();
            captured_at=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds(); width=$bounds.Width; height=$bounds.Height;
            sha256=(Hash-Bytes $bytes); image=[Convert]::ToBase64String($bytes); mime_type='image/png';
            foreground=(Foreground); paused=$script:paused }
        return $script:latest
    } finally { $stream.Dispose(); $graphics.Dispose(); $bitmap.Dispose() }
}
function Require($Condition, [string]$Message) { if (!$Condition) { throw $Message } }
function Properties($Value, [string[]]$Allowed) {
    foreach ($property in $Value.PSObject.Properties.Name) { Require ($property -in $Allowed) "Unexpected action property: $property" }
}
function Validate-Action($Action, $Observation) {
    Require ($null -ne $Action) 'Missing action'
    $kind = [string]$Action.kind
    switch ($kind) {
        'click' {
            Properties $Action @('kind','x','y','button','clicks')
            Require (($Action.x -is [int] -or $Action.x -is [long]) -and ($Action.y -is [int] -or $Action.y -is [long])) 'Coordinates must be integers'
            Require ($Action.x -ge 0 -and $Action.x -lt $Observation.width -and $Action.y -ge 0 -and $Action.y -lt $Observation.height) 'Coordinates outside screenshot'
            Require ($Action.button -in @('left','right')) 'Invalid mouse button'
            Require ($Action.clicks -in @(1,2)) 'Only one or two clicks are supported'
        }
        'type' { Properties $Action @('kind','text'); Require ($Action.text -is [string] -and $Action.text.Length -ge 1 -and $Action.text.Length -le 4096) 'Text must contain 1..4096 characters' }
        'key' { Properties $Action @('kind','key'); Require ($keyMap.ContainsKey([string]$Action.key)) 'Unsupported key chord' }
        'scroll' { Properties $Action @('kind','steps'); Require (($Action.steps -is [int] -or $Action.steps -is [long]) -and $Action.steps -ne 0 -and [Math]::Abs($Action.steps) -le 8) 'Scroll steps must be -8..8, excluding zero' }
        'open_app' { Properties $Action @('kind','app'); Require ($Action.app -in @('notepad','edge','explorer')) 'Application is outside the allowed scope' }
        default { throw 'Unsupported action kind' }
    }
}
function Act($Request) {
    Properties $Request @('desktop_id','observation_id','action_id','action')
    Require ($Request.action_id -is [string] -and $Request.action_id -match '^[a-zA-Z0-9_-]{8,80}$') 'Invalid action_id'
    $path = Join-Path $ledger ($Request.action_id + '.json')
    $signature = Hash-Bytes ([Text.Encoding]::UTF8.GetBytes(($Request | ConvertTo-Json -Depth 8 -Compress)))
    if (Test-Path $path) {
        $prior = Get-Content $path -Raw | ConvertFrom-Json
        Require ($prior.signature -eq $signature) 'action_id was reused with different arguments'
        Require ($prior.status -eq 'completed') 'Previous delivery is uncertain; observe and reconcile before using a new action_id'
        return $prior.result
    }
    Require (!$script:paused) 'Desktop is paused; wait for user resume'
    Require ($Request.desktop_id -ceq $desktopId) 'Wrong desktop identity'
    Require ($null -ne $script:latest -and $Request.observation_id -ceq $script:latest.observation_id) 'Stale observation; observe again'
    Require (([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() - $script:latest.captured_at) -lt 60000) 'Observation expired; observe again'
    $before = $script:latest
    Validate-Action $Request.action $before
    $front = Foreground
    Require ($front.handle -eq $before.foreground.handle) 'Foreground window changed; observe again'
    if ($Request.action.kind -ne 'open_app') { Require ($front.process -in $allowedApps) 'Foreground application is outside the allowed scope' }
    $record = @{ signature=$signature; status='dispatching'; request=$Request; dispatched_at=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() }
    Save-Json $path $record
    # From this durable point onward, errors imply uncertain delivery. Never
    # dispatch the same action_id again, including after executor restart.
    try {
        $a = $Request.action
        switch ($a.kind) {
            'click' {
                [DesktopNative]::SetCursorPos($a.x,$a.y) | Out-Null
                $down = 2; $up = 4
                if ($a.button -eq 'right') { $down=8; $up=16 }
                for ($i=0; $i -lt $a.clicks; $i++) {
                    [DesktopNative]::mouse_event($down,0,0,0,[UIntPtr]::Zero)
                    [DesktopNative]::mouse_event($up,0,0,0,[UIntPtr]::Zero)
                    if ($a.clicks -eq 2) { Start-Sleep -Milliseconds 80 }
                }
            }
            'type' { [DesktopNative]::TypeText($a.text) }
            'key' { [Windows.Forms.SendKeys]::SendWait($keyMap[$a.key]) }
            'scroll' { [DesktopNative]::mouse_event(0x800,0,0,([int]$a.steps * 120),[UIntPtr]::Zero) }
            'open_app' {
                switch ($a.app) {
                    'notepad' { Start-Process notepad.exe }
                    'explorer' { Start-Process explorer.exe $outputs }
                    'edge' {
                        $edge = "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
                        Start-Process $edge @('--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-component-update','--disable-sync',"--user-data-dir=$root\BrowserProfile",'file:///C:/Qw3Computer/fixture.html')
                    }
                }
                Start-Sleep -Milliseconds 800
            }
        }
        Start-Sleep -Milliseconds 250
        $after = Observe
        $actionSignature = Hash-Bytes ([Text.Encoding]::UTF8.GetBytes(($a | ConvertTo-Json -Depth 6 -Compress)))
        if ($before.sha256 -eq $after.sha256 -and $script:lastSignature -eq $actionSignature) { $script:noProgress++ }
        else { $script:noProgress = 0 }
        $script:lastSignature = $actionSignature
        if ($script:noProgress -ge 2) { $script:paused = $true; $after.paused = $true }
        $result = @{ action_id=$Request.action_id; status='completed'; observation=$after; no_progress=$script:paused }
        $record.status='completed'; $record.result=$result
        Save-Json $path $record
        return $result
    } catch {
        $script:paused=$true
        $record.status='uncertain'; $record.error=$_.Exception.Message
        Save-Json $path $record
        throw "Action delivery is uncertain; desktop paused. $($_.Exception.Message)"
    }
}

# The VM bootstrap is initially elevated. This rule is restricted to the
# executor port; the host publishes it only on loopback and every request
# requires the task-specific bearer token.
try {
    if (!(Get-NetFirewallRule -DisplayName 'QW3 Computer test executor' -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName 'QW3 Computer test executor' -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port | Out-Null
    }
    & netsh http add urlacl "url=http://+:$Port/" "user=$env:USERDOMAIN\$env:USERNAME" 2>$null | Out-Null
} catch { Write-Warning $_ }
$listener = [Net.HttpListener]::new()
$listener.Prefixes.Add("http://+:$Port/")
$listener.Start()
Set-Content (Join-Path $root 'executor-ready.txt') (Get-Date -Format o)
try {
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $response = $context.Response
        try {
            Require ($context.Request.Headers['Authorization'] -ceq "Bearer $token") 'Unauthorized'
            Require ($context.Request.ContentLength64 -le 65536) 'Request too large'
            $body = $null
            if ($context.Request.HttpMethod -eq 'POST') {
                $reader = [IO.StreamReader]::new($context.Request.InputStream, [Text.Encoding]::UTF8)
                try { $body = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
            }
            $route = $context.Request.HttpMethod + ' ' + $context.Request.Url.AbsolutePath
            switch ($route) {
                'GET /health' { $result=@{ desktop_id=$desktopId; platform='windows'; paused=$script:paused; process_id=$PID; version='0.1.0' } }
                'GET /observe' { $result=Observe }
                'POST /act' { $result=Act $body }
                'POST /control' {
                    Require ($body.command -in @('pause','resume','stop')) 'Unknown control command'
                    $script:paused = $body.command -ne 'resume'
                    $script:noProgress=0; $script:lastSignature=''; $script:latest=$null
                    $result=@{ paused=$script:paused }
                }
                'GET /files' {
                    $files = @(Get-ChildItem $outputs -File | Where-Object { $_.Length -le 1048576 } | ForEach-Object {
                        $bytes=[IO.File]::ReadAllBytes($_.FullName)
                        @{ name=$_.Name; size=$_.Length; sha256=(Hash-Bytes $bytes); content=[Text.Encoding]::UTF8.GetString($bytes) }
                    })
                    $result=@{ files=$files }
                }
                default {
                    if ($route -match '^GET /actions/([a-zA-Z0-9_-]{8,80})$') {
                        $receipt = Join-Path $ledger ($Matches[1] + '.json')
                        if (Test-Path $receipt) { $result = Get-Content $receipt -Raw | ConvertFrom-Json }
                        else { $result = @{ status='unknown' } }
                    } else { throw 'Unsupported route' }
                }
            }
            $response.StatusCode=200
        } catch {
            $response.StatusCode=409
            if ($_.Exception.Message -eq 'Unauthorized') { $response.StatusCode=401 }
            $delivery = 'rejected'
            if ($_.Exception.Message -match 'uncertain') { $delivery='uncertain' }
            $result=@{ error=$_.Exception.Message; delivery=$delivery }
        }
        $bytes=[Text.Encoding]::UTF8.GetBytes(($result | ConvertTo-Json -Depth 16 -Compress))
        $response.ContentType='application/json; charset=utf-8'
        $response.ContentLength64=$bytes.Length
        try { $response.OutputStream.Write($bytes,0,$bytes.Length) } catch { Write-Warning $_ }
        finally { $response.Close() }
    }
} finally { $listener.Stop(); $listener.Close() }
