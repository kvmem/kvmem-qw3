"""Public, per-attempt application entry points for the native executor."""
import argparse
import json
from pathlib import Path
import re

from .source import sha256


def configuration(task, run, manifest):
    if task not in {'013', '004', '006', '010', '038', '022', '035', '043', '080', '051'}:
        raise ValueError('Unknown task')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{8,60}', run) or manifest.get('run') != run or manifest.get('task') != task:
        raise ValueError('Entry configuration must belong to this attempt')
    profile = r'C:\Qw3Computer\OsworldBrowserProfile'
    folder = r'C:\Users\Qw3Test\Desktop'
    urls = manifest.get('browser_urls') or ([manifest['browser_url']] if manifest.get('browser_url') else ['about:blank'])
    if task == '035':
        profile = rf'C:\Qw3Computer\PurchaseTasks\{run}\BrowserProfile'
    elif task == '043':
        profile = rf'C:\Qw3Computer\CalendarTasks\{run}\BrowserProfile'
    elif task == '022':
        profile = rf'C:\Qw3Computer\LoanTasks\{run}\BrowserProfile'
        urls = ['http://localhost:8081/dashboard.html?cookie=local'] + [f'http://localhost:{p}/' for p in range(8082, 8087)]
        folder = r'C:\Users\Qw3Test\Documents\LoanDocs'
    elif task == '010':
        folder = r'C:\Users\Qw3Test\Downloads'
    elif task == '006':
        folder = r'C:\Users\Qw3Test\Documents'
    if task in {'013', '035', '038', '043'} and urls == ['about:blank']:
        raise ValueError('Web task manifest has no browser entry URL')
    return dict(version=1, task=task, run=run, browser_profile=profile, browser_urls=urls, explorer_path=folder)


def prepare(task, run, manifest, output, staged):
    entry = configuration(task, run, manifest)
    output.mkdir(parents=True, exist_ok=False)
    source = staged / 'application-entries.json'
    with source.open('x', encoding='utf-8') as handle:
        json.dump(entry, handle, indent=2)
    script = r'''$ErrorActionPreference='Stop'
$result=@{kind='task_application_entries';run='RUN_ID';passed=$false}
try {
    $source='\\host.lan\Data\RUN_ID-inputs\application-entries.json'
    $target='C:\Qw3Computer\application-entries.json'
    if (Test-Path $target) {throw 'Application entries exist; restore a clean task snapshot'}
    if ((Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant() -cne 'ENTRY_SHA') {throw 'Entry configuration hash mismatch'}
    Copy-Item -LiteralPath $source -Destination $target
    $result.sha256=(Get-FileHash $target -Algorithm SHA256).Hash.ToLowerInvariant()
    $result.passed=$true
} catch {$result.error=$_.Exception.ToString()}
$result | ConvertTo-Json | Set-Content '\\host.lan\Data\RUN_ID-entry-result.json' -Encoding UTF8
'''.replace('RUN_ID', run).replace('ENTRY_SHA', sha256(source))
    (output / 'install.ps1').write_text(script)
    (output / 'configuration.json').write_text(json.dumps(entry, indent=2))
    return entry


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--staged', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    prepare(manifest['task'], manifest['run'], manifest, args.output, args.staged)
