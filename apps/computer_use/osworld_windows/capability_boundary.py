"""Larger custom spreadsheet probes, separate from original OSWorld scoring."""
import json
from pathlib import Path
import re
import shutil

from .source import ROOT, sha256

TASKS = ('cap11', 'cap12')
INPUT_ROOT = ROOT / 'apps/computer_use/fixtures/windows_capability_boundary_v1'
DESKTOP = 'C:\\Users\\Qw3Test\\Desktop\\'
ORDER_ROWS = [('Pens', 4, 12), ('Paper', 3, 8), ('Folder', 7, 15),
    ('Marker', 2, 9), ('Clip', 10, 2), ('Ruler', 6, 5), ('Envelope', 8, 3),
    ('Label', 5, 4), ('Tape', 3, 11), ('Notebook', 9, 7), ('Eraser', 4, 6), ('Binder', 2, 18)]
SALES_ROWS = [('A01', 'Mei', 120, 'East'), ('B02', 'Nia', 360, 'West'),
    ('C03', 'Omar', 240, 'North'), ('D04', 'Lina', 180, 'South'),
    ('E05', 'Mei', 360, 'East'), ('F06', 'Nia', 90, 'North'),
    ('G07', 'Omar', 240, 'West'), ('H08', 'Lina', 450, 'South'),
    ('I09', 'Mei', 300, 'North'), ('J10', 'Nia', 180, 'East'),
    ('K11', 'Omar', 60, 'South'), ('L12', 'Lina', 300, 'West')]
DEFINITIONS = {
    'cap11': ('Twelve-row formulas and total', 'order12.xlsx',
        'Open C:\\Users\\Qw3Test\\Desktop\\order12.xlsx in the spreadsheet application. '
        'For all twelve items in rows 2 through 13, enter a formula in column D multiplying '
        'Quantity (column B) by Unit price (column C). In D14 enter a SUM formula for D2:D13. '
        'Keep all existing labels, input values, and row order unchanged. Keep the original file unchanged. '
        'Save the completed workbook as C:\\Users\\Qw3Test\\Desktop\\cap11.xlsx. '
        'Verify the totals and saved filename. Use only the computer tools.'),
    'cap12': ('Twelve-row sort with ties and record preservation', 'sales12.xlsx',
        'Open C:\\Users\\Qw3Test\\Desktop\\sales12.xlsx in the spreadsheet application. '
        'Sort the twelve complete data rows A2:D13 by Revenue (column C) from highest to lowest. '
        'For equal Revenue, sort by ID (column A) in ascending order. Keep every ID, Owner, Revenue '
        'and Region together as a complete record. Keep the headers and Total label unchanged. '
        'In C14 enter a SUM formula for C2:C13. Keep the original file unchanged. '
        'Save as C:\\Users\\Qw3Test\\Desktop\\cap12.xlsx. '
        'Verify the order, total and saved filename. Use only the computer tools.'),
}


def files(task):
    return [(task + '.xlsx', DESKTOP + task + '.xlsx'),
            ('original-' + DEFINITIONS[task][1], DESKTOP + DEFINITIONS[task][1])]


def public_inputs(task, directory):
    from openpyxl import Workbook
    directory = Path(directory); directory.mkdir()
    book = Workbook(); sheet = book.active
    if task == 'cap11':
        sheet.title = 'Order'
        rows = [('Item', 'Quantity', 'Unit price', 'Total')] + [(*r, None) for r in ORDER_ROWS] + [('Grand total', None, None, None)]
    elif task == 'cap12':
        sheet.title = 'Sales'
        rows = [('ID', 'Owner', 'Revenue', 'Region')] + SALES_ROWS + [('Total', None, None, None)]
    else:
        raise ValueError('Unknown boundary task')
    for row in rows: sheet.append(row)
    for col in 'ABCD': sheet.column_dimensions[col].width = 18
    book.save(directory / DEFINITIONS[task][1])


def input_sources():
    manifest = INPUT_ROOT / 'manifest.json'
    index = json.loads(manifest.read_text())
    if index.get('host_expected_outputs_included') is not False or set(index['tasks']) != set(TASKS):
        raise ValueError('Invalid boundary public corpus')
    sources = [manifest]
    for task, members in index['tasks'].items():
        if set(members) != {DEFINITIONS[task][1]}:
            raise ValueError('Unexpected boundary material')
        for name, digest in members.items():
            path = INPUT_ROOT / task / name
            if sha256(path) != digest: raise ValueError('Boundary public material changed')
            sources.append(path)
    return sources


def prepare(task, run, output, bundle=None):
    if task not in TASKS or not re.fullmatch(r'[a-zA-Z0-9_-]{8,60}', run):
        raise ValueError('Invalid boundary attempt')
    input_sources()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    staged = ROOT / 'apps/computer_use/.runtime/windows/shared' / (run + '-inputs')
    staged.mkdir(exist_ok=False)
    name = DEFINITIONS[task][1]
    shutil.copyfile(INPUT_ROOT / task / name, staged / name)
    digest = sha256(staged / name)
    script = r'''$ErrorActionPreference='Stop'
$result=@{kind='windows_boundary_initialization';run='RUN_ID';task='TASK_ID';passed=$false}
try {
    if (Test-Path 'C:\Qw3Computer\application-entries.json') {throw 'Restore a fresh legacy entry snapshot'}
    $source='\\host.lan\Data\RUN_ID-inputs\SOURCE_NAME'
    $target=Join-Path "$env:USERPROFILE\Desktop" 'SOURCE_NAME'
    if (Test-Path $target) {throw 'Material already exists'}
    if ((Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant() -cne 'SOURCE_SHA') {throw 'Material changed'}
    Copy-Item -LiteralPath $source -Destination $target
    $result.material_count=1
    $result.material_sha256='SOURCE_SHA'
    $result.passed=$true
} catch {$result.error=$_.Exception.ToString()}
$result | ConvertTo-Json -Depth 5 | Set-Content '\\host.lan\Data\RUN_ID-setup-result.json' -Encoding UTF8
'''.replace('RUN_ID', run).replace('TASK_ID', task).replace('SOURCE_NAME', name).replace('SOURCE_SHA', digest)
    (output / 'initialize.ps1').write_text(script)
    prompt = DEFINITIONS[task][2]
    manifest = dict(task=task, run=run, instruction=prompt, formal_run=False,
        suite='custom_capability_size_boundary_v1', osworld_official_task=False,
        inputs={name: digest}, rows=12, legacy_application_entries=True,
        public_inputs_manifest_sha256=sha256(INPUT_ROOT / 'manifest.json'))
    (output / 'prompt.txt').write_text(prompt)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def formula(value):
    return str(value).upper().replace(' ', '').replace('$', '')


def grade(task, artifacts):
    from openpyxl import load_workbook
    artifacts = Path(artifacts); target = artifacts / (task + '.xlsx')
    checks = dict(output_saved=target.is_file())
    report = dict(task=task, formal_run=False, osworld_official_task=False, rows=12,
        suite='custom_capability_size_boundary_v1', strict_success=False, checks=checks)
    original = artifacts / ('original-' + DEFINITIONS[task][1])
    checks['original_unchanged'] = original.is_file() and sha256(original) == sha256(INPUT_ROOT / task / DEFINITIONS[task][1])
    if not target.is_file():
        return report | dict(failure='missing_saved_output', partial_check_fraction=sum(checks.values()) / len(checks))
    try:
        book = load_workbook(target, data_only=False)
        cache = load_workbook(target, data_only=True)
        try:
            sheet, cached = book.active, cache.active
            checks['one_sheet'] = len(book.sheetnames) == 1
            checks['no_extra_rows_or_columns'] = sheet.max_row == 14 and sheet.max_column == 4
            if task == 'cap11':
                checks['headers_preserved'] = tuple(sheet.cell(1, c).value for c in range(1, 5)) == ('Item', 'Quantity', 'Unit price', 'Total')
                checks['inputs_and_order_preserved'] = [tuple(sheet.cell(r, c).value for c in range(1, 4)) for r in range(2, 14)] == ORDER_ROWS
                for r, (_, quantity, price) in enumerate(ORDER_ROWS, 2):
                    checks[f'formula_{r}'] = formula(sheet.cell(r, 4).value) in {f'=B{r}*C{r}', f'=C{r}*B{r}'}
                    checks[f'value_{r}'] = cached.cell(r, 4).value == quantity * price
                checks['total_label'] = tuple(sheet.cell(14, c).value for c in range(1, 4)) == ('Grand total', None, None)
                checks['sum_formula'] = formula(sheet['D14'].value) == '=SUM(D2:D13)'
                checks['sum_value'] = cached['D14'].value == sum(q * p for _, q, p in ORDER_ROWS)
            else:
                checks['headers_preserved'] = tuple(sheet.cell(1, c).value for c in range(1, 5)) == ('ID', 'Owner', 'Revenue', 'Region')
                checks['all_sorted_complete_records'] = [tuple(sheet.cell(r, c).value for c in range(1, 5)) for r in range(2, 14)] == sorted(SALES_ROWS, key=lambda r: (-r[2], r[0]))
                checks['total_label'] = tuple(sheet.cell(14, c).value for c in (1, 2, 4)) == ('Total', None, None)
                checks['sum_formula'] = formula(sheet['C14'].value) == '=SUM(C2:C13)'
                checks['sum_value'] = cached['C14'].value == sum(r[2] for r in SALES_ROWS)
        finally:
            book.close(); cache.close()
        report.update(strict_success=all(checks.values()), partial_check_fraction=sum(checks.values()) / len(checks), output_sha256=sha256(target))
    except Exception as exc:
        report.update(failure='invalid_saved_output', error=f'{type(exc).__name__}: {exc}')
    return report
