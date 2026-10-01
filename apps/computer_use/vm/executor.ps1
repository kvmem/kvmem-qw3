param([int]$Port = 5000)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName UIAutomationClient,UIAutomationTypes,WindowsBase
$automationRefs = @('System.dll','System.Core.dll',
    [Windows.Automation.AutomationElement].Assembly.Location,
    [Windows.Automation.AutomationPattern].Assembly.Location,
    [Windows.Rect].Assembly.Location,
    [Windows.Forms.Clipboard].Assembly.Location, [Drawing.Bitmap].Assembly.Location)
Add-Type -ReferencedAssemblies $automationRefs @'
using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Windows.Automation;
public static class DesktopNative {
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, int data, UIntPtr extra);
    [DllImport("user32.dll")] public static extern void keybd_event(byte key, byte scan, uint flags, UIntPtr extra);
    [DllImport("user32.dll")] public static extern short GetAsyncKeyState(int key);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [StructLayout(LayoutKind.Sequential)] public struct WindowRect { public int left, top, right, bottom; }
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr window, out WindowRect rect);
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
    [StructLayout(LayoutKind.Sequential)] public struct GuiThreadInfo {
        public uint size, flags;
        public IntPtr active, focus, capture, menuOwner, moveSize, caret;
        public WindowRect caretRect;
    }
    [DllImport("user32.dll", SetLastError=true)] private static extern bool GetGUIThreadInfo(uint thread, ref GuiThreadInfo info);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] private static extern int GetClassName(IntPtr window, StringBuilder name, int count);
    [DllImport("user32.dll")] private static extern bool IsWindowEnabled(IntPtr window);
    [DllImport("user32.dll")] private static extern bool IsWindowUnicode(IntPtr window);
    [DllImport("user32.dll")] private static extern bool IsWindowVisible(IntPtr window);
    [DllImport("user32.dll")] private static extern IntPtr GetAncestor(IntPtr window, uint flags);
    [DllImport("user32.dll", EntryPoint="GetWindowLongW", ExactSpelling=true)] private static extern int GetWindowStyle(IntPtr window, int index);
    [DllImport("user32.dll", EntryPoint="SendMessageTimeoutW", CharSet=CharSet.Unicode, ExactSpelling=true, SetLastError=true)]
    private static extern IntPtr SendTextMessageTimeout(IntPtr window, uint message, UIntPtr wparam,
        [MarshalAs(UnmanagedType.LPWStr)] string text, uint flags, uint timeout, out UIntPtr result);
    private static IntPtr Focus(IntPtr window) {
        uint process;
        var info = new GuiThreadInfo(); info.size = (uint)Marshal.SizeOf(typeof(GuiThreadInfo));
        if (GetForegroundWindow() != window ||
            !GetGUIThreadInfo(GetWindowThreadProcessId(window, out process), ref info) || info.focus == IntPtr.Zero)
            throw new InvalidOperationException("Focused input control changed or unavailable");
        return info.focus;
    }
    private static bool IsEdit(IntPtr focus) {
        var name = new StringBuilder(256); GetClassName(focus, name, name.Capacity);
        string kind = name.ToString();
        return kind.Equals("Edit", StringComparison.OrdinalIgnoreCase) ||
            kind.StartsWith("RichEdit", StringComparison.OrdinalIgnoreCase);
    }
    private static bool InputEnabled(IntPtr focus) {
        if (!IsWindowVisible(focus)) return false;
        // GA_PARENT excludes owners: an enabled Save As dialog can have a
        // disabled owner, while a disabled parent prevents child input.
        IntPtr current = focus;
        for (int depth = 0; current != IntPtr.Zero && depth < 64; depth++) {
            if (!IsWindowEnabled(current)) return false;
            if ((GetWindowStyle(current, -16) & 0x40000000) == 0) return true;
            current = GetAncestor(current, 1);
        }
        return false;
    }
    private static IntPtr CheckedTextFocus(IntPtr window, string text) {
        // EM_REPLACESEL receives a null-terminated UTF-16 string. Reject NUL
        // before dispatch rather than silently inserting only a prefix.
        if (text.IndexOf('\0') >= 0) throw new InvalidOperationException("Text must not contain NUL");
        IntPtr focus = Focus(window);
        if (IsEdit(focus) && (!InputEnabled(focus) || !IsWindowUnicode(focus) ||
            (GetWindowStyle(focus, -16) & 0x0800) != 0))
            throw new InvalidOperationException("Focused edit control is disabled, read-only, or not Unicode");
        return focus;
    }
    public static void ValidateText(string text) { CheckedTextFocus(GetForegroundWindow(), text); }
    public static void TypeText(string text) {
        IntPtr window = GetForegroundWindow(), focus = CheckedTextFocus(window, text);
        if (IsEdit(focus)) {
            // Insert at the current selection/caret with one undoable native
            // edit operation. Do not take clipboard ownership: process death
            // must not discard a user's clipboard. Normal UI Save still writes
            // the document; this method never reads or writes document files.
            string literal = text.Replace("\r\n", "\n").Replace("\r", "\n").Replace("\n", "\r\n");
            if (CheckedTextFocus(window, text) != focus)
                throw new InvalidOperationException("Focus changed before text input");
            UIntPtr result;
            if (SendTextMessageTimeout(focus, 0x00C2, new UIntPtr(1), literal, 3, 2000, out result) == IntPtr.Zero)
                throw new InvalidOperationException("Text delivery timed out");
            return;
        }
        // Browser and other non-edit windows retain normal Unicode input.
        var events = new List<Input>();
        for (int i = 0; i < text.Length; i++) {
            char c = text[i];
            if (c == '\r' && i + 1 < text.Length && text[i+1] == '\n') continue;
            Input down = new Input(), up = new Input(); down.type = up.type = 1;
            if (c == '\r' || c == '\n') { down.value.keyboard.vk = up.value.keyboard.vk = 0x0D; up.value.keyboard.flags = 2; }
            else { down.value.keyboard.scan = up.value.keyboard.scan = c; down.value.keyboard.flags = 4; up.value.keyboard.flags = 6; }
            events.Add(down); events.Add(up);
        }
        if (Focus(window) != focus) throw new InvalidOperationException("Focus changed before text input");
        Input[] inputs = events.ToArray();
        var watch = Stopwatch.StartNew();
        for (int offset = 0; offset < inputs.Length;) {
            if (Focus(window) != focus || watch.ElapsedMilliseconds > 14000)
                throw new InvalidOperationException("Focus changed or typing deadline exceeded; remaining text was not sent");
            int count = Math.Min(64, inputs.Length - offset);
            if (offset + count < inputs.Length && Char.IsHighSurrogate((char)inputs[offset+count-2].value.keyboard.scan)) count += 2;
            var batch = new Input[count]; Array.Copy(inputs, offset, batch, 0, count);
            if (SendInput((uint)count, batch, Marshal.SizeOf(typeof(Input))) != count)
                throw new InvalidOperationException("SendInput did not deliver all input events");
            offset += count;
            // Bound each queued burst. A 4096-unit browser field otherwise
            // leaves the UI processing input after the following click.
            System.Threading.Thread.Sleep(20);
        }
    }
}
public sealed class ControlHint {
    public string name;
    public string kind;
    public int x, y, width, height, center_x, center_y;
    public bool focused;
}
public sealed class ControlHints {
    public string status;
    public long elapsed_ms;
    public bool truncated;
    public ControlHint[] controls = new ControlHint[0];
}
public static class DesktopAccessibility {
    private static Task<ControlHints> pending;
    public static ControlHints Read(IntPtr handle, int width, int height) {
        // UIA providers can hang. Keep at most one MTA worker and never return
        // an old worker's controls as evidence for a newer observation.
        if (pending != null && !pending.IsCompleted)
            return new ControlHints { status = "provider_busy" };
        pending = Task.Run(() => Collect(handle, width, height));
        if (!pending.Wait(750)) return new ControlHints { status = "timeout" };
        return pending.Result;
    }
    private static ControlHints Collect(IntPtr handle, int width, int height) {
        var watch = Stopwatch.StartNew();
        try {
            var root = AutomationElement.FromHandle(handle);
            var types = new [] { ControlType.Button, ControlType.CheckBox, ControlType.ComboBox,
                ControlType.Edit, ControlType.MenuItem, ControlType.RadioButton, ControlType.TabItem,
                ControlType.Hyperlink, ControlType.ListItem, ControlType.TreeItem };
            var conditions = new List<Condition>();
            foreach (var type in types)
                conditions.Add(new PropertyCondition(AutomationElement.ControlTypeProperty, type));
            var filter = new AndCondition(new OrCondition(conditions.ToArray()),
                new PropertyCondition(AutomationElement.IsOffscreenProperty, false),
                new PropertyCondition(AutomationElement.IsEnabledProperty, true));
            var cache = new CacheRequest();
            cache.TreeScope = TreeScope.Element;
            cache.AutomationElementMode = AutomationElementMode.None;
            foreach (var property in new [] { AutomationElement.NameProperty,
                AutomationElement.ControlTypeProperty, AutomationElement.BoundingRectangleProperty,
                AutomationElement.IsPasswordProperty, AutomationElement.HasKeyboardFocusProperty })
                cache.Add(property);
            AutomationElementCollection elements;
            using (cache.Activate()) { elements = root.FindAll(TreeScope.Descendants, filter); }
            var hints = new List<ControlHint>();
            foreach (AutomationElement element in elements) {
                var info = element.Cached;
                if (info.IsPassword) continue;
                var rect = info.BoundingRectangle;
                if (rect.IsEmpty || rect.Width <= 0 || rect.Height <= 0) continue;
                int cx = (int)Math.Floor(rect.X + rect.Width / 2), cy = (int)Math.Floor(rect.Y + rect.Height / 2);
                if (cx < 0 || cy < 0 || cx >= width || cy >= height) continue;
                string name = info.Name ?? "";
                if (name.Length > 200) name = name.Substring(0, 200);
                hints.Add(new ControlHint { name = name, kind = info.ControlType.ProgrammaticName.Replace("ControlType.", ""),
                    x = (int)Math.Floor(rect.X), y = (int)Math.Floor(rect.Y), width = (int)Math.Ceiling(rect.Width),
                    height = (int)Math.Ceiling(rect.Height), center_x = cx, center_y = cy,
                    focused = info.HasKeyboardFocus });
                if (hints.Count == 80) break;
            }
            return new ControlHints { status = "ok", elapsed_ms = watch.ElapsedMilliseconds,
                truncated = hints.Count == 80, controls = hints.ToArray() };
        } catch (Exception error) {
            return new ControlHints { status = "unavailable:" + error.GetType().Name, elapsed_ms = watch.ElapsedMilliseconds };
        }
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
$allowedApps = @('explorer','msedge','notepad','ApplicationFrameHost','StartMenuExperienceHost','SearchHost','ShellExperienceHost',
    'soffice','soffice.bin','thunderbird','gimp-3','et','wps')
$officeApps = @{
    'writer'=@{path='C:\Program Files\LibreOffice\program\soffice.exe'; arguments=@('--writer')}
    'calc'=@{path='C:\Program Files\LibreOffice\program\soffice.exe'; arguments=@('--calc')}
    'impress'=@{path='C:\Program Files\LibreOffice\program\soffice.exe'; arguments=@('--impress')}
    'thunderbird'=@{path='C:\Program Files\Mozilla Thunderbird\thunderbird.exe'; arguments=@()}
    'gimp'=@{path='C:\Program Files\GIMP 3\bin\gimp-3.exe'; arguments=@()}
    'wps_spreadsheets'=@{path="$env:LOCALAPPDATA\Kingsoft\WPS Office\12.1.0.28485\office6\et.exe"; arguments=@()}
}
$keyMap = @{
    'ENTER'='{ENTER}'; 'TAB'='{TAB}'; 'SHIFT+TAB'='+{TAB}'; 'ESC'='{ESC}';
    'BACKSPACE'='{BACKSPACE}'; 'DELETE'='{DELETE}'; 'UP'='{UP}'; 'DOWN'='{DOWN}';
    'LEFT'='{LEFT}'; 'RIGHT'='{RIGHT}'; 'HOME'='{HOME}'; 'END'='{END}';
    'CTRL+A'='^a'; 'CTRL+C'='^c'; 'CTRL+V'='^v'; 'CTRL+X'='^x';
    'CTRL+S'='^s'; 'CTRL+SHIFT+S'='^+s'; 'CTRL+N'='^n'; 'CTRL+O'='^o';
    'CTRL+L'='^l'; 'CTRL+W'='^w'; 'CTRL+Z'='^z'; 'ALT+TAB'='%{TAB}'; 'ALT+F4'='%{F4}';
    'SPACE'=' '; 'PAGEUP'='{PGUP}'; 'PAGEDOWN'='{PGDN}';
    'SHIFT+UP'='+{UP}'; 'SHIFT+DOWN'='+{DOWN}'; 'SHIFT+LEFT'='+{LEFT}'; 'SHIFT+RIGHT'='+{RIGHT}';
    'SHIFT+HOME'='+{HOME}'; 'SHIFT+END'='+{END}';
    'CTRL+HOME'='^{HOME}'; 'CTRL+END'='^{END}'; 'CTRL+SHIFT+HOME'='^+{HOME}'; 'CTRL+SHIFT+END'='^+{END}';
    'CTRL+UP'='^{UP}'; 'CTRL+DOWN'='^{DOWN}'; 'CTRL+LEFT'='^{LEFT}'; 'CTRL+RIGHT'='^{RIGHT}';
    'CTRL+PAGEUP'='^{PGUP}'; 'CTRL+PAGEDOWN'='^{PGDN}'; 'CTRL+TAB'='^{TAB}'; 'CTRL+SHIFT+TAB'='^+{TAB}';
    'CTRL+SHIFT+N'='^+n'; 'CTRL+SHIFT+O'='^+o'; 'CTRL+Y'='^y'; 'CTRL+SHIFT+Z'='^+z';
    'CTRL+F'='^f'; 'CTRL+H'='^h'; 'CTRL+B'='^b'; 'CTRL+I'='^i'; 'CTRL+U'='^u'; 'CTRL+P'='^p';
    'CTRL+1'='^1'; 'CTRL+0'='^0'; 'F2'='{F2}'; 'F4'='{F4}'; 'F5'='{F5}'; 'F6'='{F6}'; 'F11'='{F11}'; 'F12'='{F12}';
    'M'='m'; 'R'='r'; 'S'='s'; 'T'='t'; 'V'='v'; 'Z'='z'; 'SHIFT+R'='+r'; 'SHIFT+T'='+t'; 'SHIFT+M'='+m'
}

function Save-Json($Path, $Value) {
    $temp = "$Path.tmp"
    [IO.File]::WriteAllText($temp, ($Value | ConvertTo-Json -Depth 16 -Compress), [Text.UTF8Encoding]::new($false))
    Move-Item -Force $temp $Path
}
$heldInputFile = Join-Path $root 'held-drag-input.json'
function Release-DragInput([int]$Modifier) {
    [DesktopNative]::mouse_event(4,0,0,0,[UIntPtr]::Zero)
    if ($Modifier) { [DesktopNative]::keybd_event($Modifier,0,2,[UIntPtr]::Zero) }
}
# A process kill bypasses finally. The durable marker is written before any
# button/key down; restart releases only this executor's recorded drag input.
# Its action stays uncertain in the ledger and is never dispatched again.
if (Test-Path -LiteralPath $heldInputFile) {
    $held = Get-Content -LiteralPath $heldInputFile -Raw | ConvertFrom-Json
    if ($held.action_id -notmatch '^[a-zA-Z0-9_-]{8,80}$' -or
        $held.modifier -notin @(0,16,17,18)) { throw 'Invalid held input recovery record' }
    Release-DragInput $held.modifier
    Save-Json (Join-Path $root ('input-recovery-' + [guid]::NewGuid().ToString() + '.json')) @{
        action_id=$held.action_id; modifier=$held.modifier; released_at=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        reason='executor_restart'; original_delivery='uncertain'
    }
    Remove-Item -LiteralPath $heldInputFile
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
function Observe([int]$Attempt = 0) {
    $bounds = [Windows.Forms.Screen]::PrimaryScreen.Bounds
    $bitmap = [Drawing.Bitmap]::new($bounds.Width, $bounds.Height)
    $graphics = [Drawing.Graphics]::FromImage($bitmap)
    $stream = [IO.MemoryStream]::new()
    try {
        $front = Foreground
        $graphics.CopyFromScreen($bounds.Location, [Drawing.Point]::Empty, $bounds.Size)
        $bitmap.Save($stream, [Drawing.Imaging.ImageFormat]::Png)
        $bytes = $stream.ToArray()
        $progressHash = Hash-Bytes $bytes
        $windowRect = [DesktopNative+WindowRect]::new()
        if ([DesktopNative]::GetWindowRect([IntPtr]$front.handle, [ref]$windowRect)) {
            $left = [Math]::Max(0, $windowRect.left - $bounds.X)
            $top = [Math]::Max(0, $windowRect.top - $bounds.Y)
            $right = [Math]::Min($bounds.Width, $windowRect.right - $bounds.X)
            $bottom = [Math]::Min($bounds.Height, $windowRect.bottom - $bounds.Y)
            if ($right -gt $left -and $bottom -gt $top) {
                $region = [Drawing.Rectangle]::new($left, $top, ($right-$left), ($bottom-$top))
                $crop = $bitmap.Clone($region, $bitmap.PixelFormat)
                $progressStream = [IO.MemoryStream]::new()
                try {
                    $crop.Save($progressStream, [Drawing.Imaging.ImageFormat]::Png)
                    $progressHash = Hash-Bytes $progressStream.ToArray()
                } finally { $crop.Dispose(); $progressStream.Dispose() }
            }
        }
        $accessibility = [DesktopAccessibility]::Read([IntPtr]$front.handle, $bounds.Width, $bounds.Height)
        $afterRead = Foreground
        if ($afterRead.handle -ne $front.handle -or $afterRead.title -ne $front.title) {
            if ($Attempt -lt 2) { return (Observe -Attempt ($Attempt + 1)) }
            # Keep the screenshot's original window identity; dispatch will
            # reject a changed foreground. Never pair it with another window's
            # control coordinates during a continuing transition.
            $accessibility = @{status='foreground_changed'; controls=@(); truncated=$false}
        }
        $script:latest = @{ desktop_id=$desktopId; observation_id=[guid]::NewGuid().ToString();
            captured_at=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds(); width=$bounds.Width; height=$bounds.Height;
            sha256=(Hash-Bytes $bytes); image=[Convert]::ToBase64String($bytes); mime_type='image/png';
            foreground=$front; accessibility=$accessibility; progress_sha256=$progressHash; paused=$script:paused }
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
        'drag' {
            Properties $Action @('kind','from_x','from_y','to_x','to_y','duration_ms','modifier')
            foreach ($name in @('from_x','from_y','to_x','to_y','duration_ms')) {
                Require ($Action.$name -is [int] -or $Action.$name -is [long]) 'Drag coordinates and duration must be integers'
            }
            Require ($Action.from_x -ge 0 -and $Action.from_x -lt $Observation.width -and
                $Action.to_x -ge 0 -and $Action.to_x -lt $Observation.width -and
                $Action.from_y -ge 0 -and $Action.from_y -lt $Observation.height -and
                $Action.to_y -ge 0 -and $Action.to_y -lt $Observation.height) 'Drag coordinates outside screenshot'
            Require ($Action.duration_ms -ge 100 -and $Action.duration_ms -le 2000) 'Drag duration must be 100..2000 ms'
            Require ($Action.modifier -in @('none','SHIFT','CTRL','ALT')) 'Unsupported drag modifier'
        }
        'key' { Properties $Action @('kind','key'); Require ($keyMap.ContainsKey([string]$Action.key)) 'Unsupported key chord' }
        'scroll' {
            Properties $Action @('kind','direction','steps','x','y')
            Require ($Action.direction -in @('up','down')) 'Scroll direction must be up or down'
            Require (($Action.steps -is [int] -or $Action.steps -is [long]) -and $Action.steps -ge 1 -and $Action.steps -le 8) 'Scroll steps must be 1..8'
            Require (($Action.x -is [int] -or $Action.x -is [long]) -and ($Action.y -is [int] -or $Action.y -is [long])) 'Scroll coordinates must be integers'
            Require ($Action.x -ge 0 -and $Action.x -lt $Observation.width -and $Action.y -ge 0 -and $Action.y -lt $Observation.height) 'Scroll coordinates outside screenshot'
        }
        'open_app' {
            Properties $Action @('kind','app')
            Require ($Action.app -in @('notepad','edge','explorer') -or $officeApps.ContainsKey([string]$Action.app)) 'Application is outside the allowed scope'
            if ($officeApps.ContainsKey([string]$Action.app)) {
                Require (Test-Path -LiteralPath $officeApps[$Action.app].path -PathType Leaf) 'Requested office application is not installed'
            }
        }
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
    if ($Request.action.kind -eq 'type') { [DesktopNative]::ValidateText($Request.action.text) }
    if ($Request.action.kind -eq 'drag') {
        foreach ($key in @(1,16,17,18)) {
            Require (([DesktopNative]::GetAsyncKeyState($key) -band 0x8000) -eq 0) 'Release held mouse/modifier input before dragging'
        }
    }
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
            'drag' {
                $modifier = @{none=0; SHIFT=16; CTRL=17; ALT=18}[$a.modifier]
                Save-Json $heldInputFile @{action_id=$Request.action_id; modifier=$modifier}
                [DesktopNative]::SetCursorPos($a.from_x,$a.from_y) | Out-Null
                try {
                    if ($modifier) { [DesktopNative]::keybd_event($modifier,0,0,[UIntPtr]::Zero) }
                    [DesktopNative]::mouse_event(2,0,0,0,[UIntPtr]::Zero)
                    $steps = [Math]::Max(5, [Math]::Ceiling($a.duration_ms / 20.0))
                    for ($step=1; $step -le $steps; $step++) {
                        # Losing foreground during a drag makes delivery uncertain;
                        # always release held input, then let the ledger pause it.
                        if ([DesktopNative]::GetForegroundWindow().ToInt64() -ne $front.handle) {
                            throw 'Foreground changed during drag'
                        }
                        Start-Sleep -Milliseconds ([int][Math]::Ceiling($a.duration_ms / $steps))
                        $x = [int][Math]::Round($a.from_x + ($a.to_x - $a.from_x) * $step / $steps)
                        $y = [int][Math]::Round($a.from_y + ($a.to_y - $a.from_y) * $step / $steps)
                        [DesktopNative]::SetCursorPos($x,$y) | Out-Null
                    }
                } finally {
                    Release-DragInput $modifier
                    Remove-Item -LiteralPath $heldInputFile
                }
            }
            'key' { [Windows.Forms.SendKeys]::SendWait($keyMap[$a.key]) }
            'scroll' {
                [DesktopNative]::SetCursorPos($a.x,$a.y) | Out-Null
                $delta = [int]$a.steps * 120
                if ($a.direction -eq 'down') { $delta = -$delta }
                [DesktopNative]::mouse_event(0x800,0,0,$delta,[UIntPtr]::Zero)
                # Edge's smooth wheel animation can still be moving when the
                # normal 250ms post-input screenshot is captured.
                Start-Sleep -Milliseconds 600
            }
            'open_app' {
                if ($officeApps.ContainsKey([string]$a.app)) {
                    $launch = $officeApps[$a.app]
                    if ($launch.arguments.Count) { Start-Process -FilePath $launch.path -ArgumentList $launch.arguments }
                    else { Start-Process -FilePath $launch.path }
                }
                switch ($a.app) {
                    'notepad' { Start-Process notepad.exe }
                    'explorer' { Start-Process explorer.exe $outputs }
                    'edge' {
                        $edge = "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
                        Start-Process $edge @('--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-component-update','--disable-sync','--force-renderer-accessibility',"--user-data-dir=$root\BrowserProfile",'file:///C:/Qw3Computer/fixture.html')
                    }
                }
                Start-Sleep -Milliseconds 800
            }
        }
        Start-Sleep -Milliseconds 250
        $after = Observe
        $actionSignature = Hash-Bytes ([Text.Encoding]::UTF8.GetBytes(($a | ConvertTo-Json -Depth 6 -Compress)))
        # Taskbar notifications and background windows do not count as progress
        # in the active application. Preserve the full image/hash for evidence.
        if ($before.foreground.handle -eq $after.foreground.handle -and
            $before.progress_sha256 -eq $after.progress_sha256 -and
            $script:lastSignature -eq $actionSignature) { $script:noProgress++ }
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
