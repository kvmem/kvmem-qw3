param([string]$Source = 'C:\Qw3Computer\executor.ps1', [string]$ReportName = 'input-recovery-semantics.json')
# Setup-only real native EDIT control probe, never a Computer MCP tool.
# Run only after the frozen VM queue has stopped and this candidate is staged.
$ErrorActionPreference = 'Stop'
$share = '\\host.lan\Data'
$code = [IO.File]::ReadAllText($source)
$boundary = $code.IndexOf("`n[DesktopNative]::SetProcessDPIAware()")
if ($boundary -lt 0) { throw 'Candidate compilation boundary not found' }
Invoke-Expression $code.Substring(0, $boundary)
Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public static class NativeEditFixture {
    [DllImport("user32.dll")] public static extern uint GetClipboardSequenceNumber();
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode)] private static extern IntPtr GetModuleHandle(string name);
    [DllImport("user32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    private static extern IntPtr CreateWindowEx(uint extended, string kind, string title, uint style,
        int x, int y, int width, int height, IntPtr parent, IntPtr menu, IntPtr instance, IntPtr data);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern bool SetWindowText(IntPtr window, string text);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] private static extern int GetWindowText(IntPtr window, StringBuilder text, int count);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] private static extern int GetWindowTextLength(IntPtr window);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr SendMessage(IntPtr window, uint message, IntPtr wparam, IntPtr lparam);
    [DllImport("user32.dll")] public static extern IntPtr SetFocus(IntPtr window);
    [DllImport("user32.dll")] public static extern bool EnableWindow(IntPtr window, bool enabled);
    public static IntPtr Create(IntPtr parent) {
        // A WinForms TextBox has a decorated class name. Use the actual EDIT
        // window class that the executor supports, hosted in a visible form.
        IntPtr result = CreateWindowEx(0, "EDIT", "", 0x50810044, 10, 10, 650, 300,
            parent, IntPtr.Zero, GetModuleHandle(null), IntPtr.Zero);
        if (result == IntPtr.Zero) throw new InvalidOperationException("Native EDIT creation failed");
        return result;
    }
    public static string Text(IntPtr window) {
        var text = new StringBuilder(GetWindowTextLength(window) + 1);
        GetWindowText(window, text, text.Capacity); return text.ToString();
    }
    public static void Select(IntPtr window, int first, int last) {
        SendMessage(window, 0x00B1, new IntPtr(first), new IntPtr(last));
    }
    public static void ReadOnly(IntPtr window, bool value) {
        SendMessage(window, 0x00CF, new IntPtr(value ? 1 : 0), IntPtr.Zero);
    }
    public static void Undo(IntPtr window) { SendMessage(window, 0x00C7, IntPtr.Zero, IntPtr.Zero); }
}
'@
$result = @{kind='native_edit_control_semantics_probe'; passed=$false; checks=@();
    executor_sha256=(Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant();
    clipboard_sequence_before=[NativeEditFixture]::GetClipboardSequenceNumber()}
function Record([string]$name, [bool]$passed) {
    $script:result.checks += @{name=$name; passed=$passed}
    if (!$passed) { throw "Failed: $name" }
}
$form = [Windows.Forms.Form]::new()
$form.Text = 'QW3 isolated native edit input contract'
$form.Size = [Drawing.Size]::new(700,400)
$form.TopMost = $true
try {
    $form.Show(); $form.Activate()
    $edit = [NativeEditFixture]::Create($form.Handle)
    [NativeEditFixture]::SetFocus($edit) | Out-Null
    [Windows.Forms.Application]::DoEvents()
    [NativeEditFixture]::SetWindowText($edit, 'prefix OLD suffix') | Out-Null
    [NativeEditFixture]::Select($edit, 7, 10)
    [DesktopNative]::TypeText('NEW')
    Record 'replace_current_selection' ([NativeEditFixture]::Text($edit) -ceq 'prefix NEW suffix')
    [NativeEditFixture]::Undo($edit)
    Record 'selection_replacement_is_undoable' ([NativeEditFixture]::Text($edit) -ceq 'prefix OLD suffix')
    [NativeEditFixture]::Select($edit, 17, 17)
    [DesktopNative]::TypeText(' END')
    Record 'insert_at_caret' ([NativeEditFixture]::Text($edit) -ceq 'prefix OLD suffix END')
    [NativeEditFixture]::Select($edit, 0, -1)
    $unicode = [string][char]0x4E2D + [char]0x6587 + [char]0xD83D + [char]0xDE42 + "`nnext`n"
    [DesktopNative]::TypeText($unicode)
    Record 'unicode_and_newlines' ([NativeEditFixture]::Text($edit) -ceq $unicode.Replace("`n","`r`n"))
    [NativeEditFixture]::ReadOnly($edit, $true)
    [NativeEditFixture]::Select($edit, 0, -1)
    $before = [NativeEditFixture]::Text($edit)
    $rejected = $false
    try { [DesktopNative]::ValidateText('must not write') } catch { $rejected = $true }
    Record 'readonly_rejected_before_dispatch' ($rejected -and [NativeEditFixture]::Text($edit) -ceq $before)
    [NativeEditFixture]::ReadOnly($edit, $false)
    $rejected = $false
    try { [DesktopNative]::ValidateText("prefix$([char]0)suffix") } catch { $rejected = $true }
    Record 'nul_rejected_before_dispatch' ($rejected -and [NativeEditFixture]::Text($edit) -ceq $before)
    [NativeEditFixture]::EnableWindow($edit, $false) | Out-Null
    $rejected = $false
    try { [DesktopNative]::ValidateText('must not write') } catch { $rejected = $true }
    Record 'disabled_control_rejected' ($rejected -and [NativeEditFixture]::Text($edit) -ceq $before)
    [NativeEditFixture]::EnableWindow($edit, $true) | Out-Null
    [NativeEditFixture]::SetFocus($edit) | Out-Null
    [NativeEditFixture]::EnableWindow($form.Handle, $false) | Out-Null
    $rejected = $false
    try { [DesktopNative]::ValidateText('must not write') } catch { $rejected = $true }
    Record 'disabled_parent_rejected' ($rejected -and [NativeEditFixture]::Text($edit) -ceq $before)
    [NativeEditFixture]::EnableWindow($form.Handle, $true) | Out-Null
    $result.clipboard_sequence_after = [NativeEditFixture]::GetClipboardSequenceNumber()
    Record 'clipboard_never_changed' ($result.clipboard_sequence_before -eq $result.clipboard_sequence_after)
    $result.passed = $true
} catch { $result.error = $_.Exception.Message }
finally {
    $form.Close(); $form.Dispose()
    $result | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $share $ReportName) -Encoding UTF8
}
