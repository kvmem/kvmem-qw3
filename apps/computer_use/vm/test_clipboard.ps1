# Setup-only clipboard fixture in the dedicated VM, never a Computer MCP tool.
# The host supplies {"operation":"seed"} or {"operation":"check"} in the share.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms,System.Drawing
$share = '\\host.lan\Data'
$request = Get-Content "$share\clipboard-check-request.json" -Raw | ConvertFrom-Json
$result = @{ operation=$request.operation; passed=$false }
try {
    $text = 'QW3 clipboard sentinel ' + [char]0x4E2D + [char]0x6587 + [char]0xD83D + [char]0xDE42
    $rtf = '{\rtf1\ansi QW3 sentinel}'
    $bytes = [byte[]]@(5,8,13,21,34)
    if ($request.operation -eq 'seed') {
        $data = [Windows.Forms.DataObject]::new()
        $data.SetData([Windows.Forms.DataFormats]::UnicodeText, $false, $text)
        $data.SetData([Windows.Forms.DataFormats]::Rtf, $false, $rtf)
        $data.SetData([Windows.Forms.DataFormats]::FileDrop, $false, [string[]]@('C:\Qw3Computer\fixture.html'))
        $stream = [IO.MemoryStream]::new($bytes)
        $bitmap = [Drawing.Bitmap]::new(2,2)
        $bitmap.SetPixel(0,0,[Drawing.Color]::Red)
        $bitmap.SetPixel(1,1,[Drawing.Color]::Blue)
        try {
            $data.SetData('Qw3FixtureBytes', $false, $stream)
            $data.SetData([Windows.Forms.DataFormats]::Bitmap, $false, $bitmap)
            [Windows.Forms.Clipboard]::SetDataObject($data, $true)
        } finally { $stream.Dispose(); $bitmap.Dispose() }
        $result.passed = $true
    } elseif ($request.operation -eq 'check') {
        $data = [Windows.Forms.Clipboard]::GetDataObject()
        $stream = $data.GetData('Qw3FixtureBytes', $false)
        $bitmap = $data.GetData([Windows.Forms.DataFormats]::Bitmap, $false)
        $files = $data.GetData([Windows.Forms.DataFormats]::FileDrop, $false)
        $result.checks = @{
            unicode_text = $data.GetData([Windows.Forms.DataFormats]::UnicodeText, $false) -ceq $text
            rich_text = $data.GetData([Windows.Forms.DataFormats]::Rtf, $false) -ceq $rtf
            file_drop = @($files).Count -eq 1 -and $files[0] -ceq 'C:\Qw3Computer\fixture.html'
            custom_bytes = [Convert]::ToBase64String($stream.ToArray()) -ceq [Convert]::ToBase64String($bytes)
            bitmap = $bitmap.Width -eq 2 -and $bitmap.Height -eq 2 -and $bitmap.GetPixel(0,0).ToArgb() -eq [Drawing.Color]::Red.ToArgb() -and $bitmap.GetPixel(1,1).ToArgb() -eq [Drawing.Color]::Blue.ToArgb()
        }
        $result.passed = @($result.checks.Values | Where-Object { !$_ }).Count -eq 0
        $result.formats = @($data.GetFormats($false))
    } else { throw 'Unknown clipboard fixture operation' }
} catch { $result.error = $_.Exception.Message }
$result.executor_sha256 = (Get-FileHash 'C:\Qw3Computer\executor.ps1' -Algorithm SHA256).Hash.ToLower()
$result | ConvertTo-Json -Depth 5 | Set-Content "$share\clipboard-check-result.json" -Encoding UTF8
