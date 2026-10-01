# Setup-only changing desktop for independent recorder qualification.
$ErrorActionPreference = 'Stop'
$file = 'C:\Qw3Computer\recorder-probe.html'
@'
<!doctype html><html><head><meta charset="utf-8"><title>Recorder qualification</title></head>
<body style="font-family:Segoe UI;background:#102030;color:white;padding:60px">
<h1>Recorder qualification — synthetic test data</h1>
<p>The counter updates every 100 ms. No network requests or agent observations.</p>
<p id="counter" style="font-size:96px;font-variant-numeric:tabular-nums">0</p>
<div id="bar" style="height:100px;background:#28b"> </div>
<script>
const start=performance.now();
setInterval(()=>{const n=Math.floor((performance.now()-start)/100);
document.getElementById('counter').textContent=n;
document.getElementById('bar').style.width=(100+(n%10)*70)+'px';},100);
</script></body></html>
'@ | Set-Content -LiteralPath $file -Encoding UTF8
$edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
Start-Process -FilePath $edge -ArgumentList '--new-window','--app=file:///C:/Qw3Computer/recorder-probe.html'
