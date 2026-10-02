"""Custom capability calibration, distinct from the ten original OSWorld tasks.

Only public inputs and instructions enter the VM. Expected results and grading
remain on the host. The qualified executor's legacy application entries suffice;
this suite does not change its allowlist, client bundle, or disk snapshot.
"""
import json
from pathlib import Path
import re
import shutil

from .source import ROOT, sha256
from . import capability_boundary as boundary

BASIC_TASKS = tuple(f'cap{i:02d}' for i in range(1, 11))
TASKS = BASIC_TASKS + boundary.TASKS
INPUT_ROOT = ROOT / 'apps/computer_use/fixtures/windows_capability'
DESKTOP = 'C:\\Users\\Qw3Test\\Desktop\\'
OUTPUTS = 'C:\\Qw3Computer\\outputs\\'
DEFINITIONS = {
    'cap01': ('Create and save a note', 300, 'cap01.txt',
        'Open Notepad and create a NEW document. Write exactly these two lines:\n'
        'QW3 computer use\nLocal Windows calibration.\n'
        'Save it as C:\\Qw3Computer\\outputs\\cap01.txt. Verify the filename and visible text.'),
    'cap02': ('Edit a note without changing other lines', 300, 'cap02.txt',
        'Open C:\\Users\\Qw3Test\\Desktop\\draft.txt in Notepad. Change Status: Draft to Status: Approved '
        'and Hours: 6 to Hours: 8. Preserve the title and Owner line. Save the result as '
        'C:\\Qw3Computer\\outputs\\cap02.txt, keeping the original draft.txt unchanged.'),
    'cap03': ('Browser information to a saved note', 600, 'cap03.txt',
        'Open the Edge browser fixture with computer open_app edge. Find the reference code in its Copy task '
        'section. Put only that reference code into a NEW Notepad document and save it as '
        'C:\\Qw3Computer\\outputs\\cap03.txt. Verify the saved text and filename.'),
    'cap04': ('Fill and submit an offline browser form', 600, 'browser-result.json',
        'Open the Edge browser fixture with computer open_app edge. Fill its Request form with '
        'Name Nia, Reference QC-042, Priority Low, and check the confirmation checkbox. Click Save request '
        'to download browser-result.json. Verify its visible Saved status.'),
    'cap05': ('Spreadsheet formulas and a saved workbook', 600, 'cap05.xlsx',
        'Open C:\\Users\\Qw3Test\\Desktop\\order.xlsx in the spreadsheet application. '
        'In D2:D4 calculate each row total using a formula multiplying Quantity by Unit price. '
        'In D5 use a SUM formula to add D2:D4. Keep all existing labels and inputs unchanged. '
        'Save as C:\\Users\\Qw3Test\\Desktop\\cap05.xlsx and verify the totals.'),
    'cap06': ('Document title and constrained text replacement', 600, 'cap06.docx',
        'Open C:\\Users\\Qw3Test\\Desktop\\weekly.docx in the document application. '
        'Change the title Weekly Update to Weekly Report, and replace both occurrences of Draft with Approved. '
        'Keep all other text, including Owner: Mei and the two project names, unchanged. '
        'Save as C:\\Users\\Qw3Test\\Desktop\\cap06.docx.'),
    'cap07': ('Edit and remove specified presentation slides', 600, 'cap07.pptx',
        'Open C:\\Users\\Qw3Test\\Desktop\\briefing.pptx in the presentation application. '
        'Change only the title of slide 2 from Findings to Results. Delete the last slide titled Obsolete. '
        'Keep slide 1 and the body text on slide 2 unchanged. Save as C:\\Users\\Qw3Test\\Desktop\\cap07.pptx.'),
    'cap08': ('Resolve two rule versions across applications', 900, 'cap08.txt',
        'Read C:\\Users\\Qw3Test\\Desktop\\policy-old.txt and policy-new.txt. Use the latest effective policy '
        'for the 2026-10-02 request with cost 125.00. Also obtain the Copy task reference code from the Edge '
        'browser fixture. Create a NEW Notepad report containing exactly four lines: '
        'Reference: followed by the code; Policy: followed by the effective date; '
        'Amount: followed by the reimbursement with two decimal places; Status: Approved. '
        'Use one space after each colon. Save as C:\\Qw3Computer\\outputs\\cap08.txt.'),
    'cap09': ('Rename and move files, preserve an unrelated file', 600, 'summary.txt',
        'Using File Explorer, open C:\\Users\\Qw3Test\\Desktop\\CapFiles. '
        'Rename draft.txt to summary.txt, create an Archive folder, and move todo.txt into Archive. '
        'Keep keep.txt unchanged. Do not edit file contents or leave extra copies of the original names.'),
    'cap10': ('Sort rows while preserving records and formulas', 900, 'cap10.xlsx',
        'Open C:\\Users\\Qw3Test\\Desktop\\sales.xlsx in the spreadsheet application. '
        'Sort the four complete data rows A2:C5 by Revenue from highest to lowest, keeping IDs and owners '
        'with their revenue. Keep the headers unchanged. In C6 enter a SUM formula adding C2:C5. '
        'Save as C:\\Users\\Qw3Test\\Desktop\\cap10.xlsx and verify the order and total.'),
}


def files(task):
    if task in boundary.TASKS:
        return boundary.files(task)
    if task == 'cap09':
        folder = DESKTOP + 'CapFiles\\'
        return [(name, folder + relative) for name, relative in [
            ('summary.txt', 'summary.txt'), ('archived-todo.txt', 'Archive\\todo.txt'),
            ('keep.txt', 'keep.txt'), ('old-draft.txt', 'draft.txt'), ('old-todo.txt', 'todo.txt')]]
    # The executor's legacy /files route is for text/JSON. Office binaries use
    # explicit Desktop export, as do the original qualified office scenarios.
    folder = DESKTOP if task in {'cap05', 'cap06', 'cap07', 'cap10'} else OUTPUTS
    result = [(DEFINITIONS[task][2], folder + DEFINITIONS[task][2])]
    if task == 'cap02':
        result.append(('original-draft.txt', DESKTOP + 'draft.txt'))
    return result


def public_inputs(task, directory):
    """Generate inputs, never completed artifacts or host grading code."""
    if task in boundary.TASKS:
        return boundary.public_inputs(task, directory)
    directory.mkdir()
    if task == 'cap02':
        (directory / 'draft.txt').write_text('Project note\nOwner: Mei\nStatus: Draft\nHours: 6\n')
    elif task == 'cap05':
        from openpyxl import Workbook
        book = Workbook(); sheet = book.active; sheet.title = 'Order'
        for row in [('Item', 'Quantity', 'Unit price', 'Total'), ('Pens', 2, 12, None),
                    ('Paper', 3, 8, None), ('Folder', 1, 15, None), ('Grand total', None, None, None)]:
            sheet.append(row)
        for col in 'ABCD': sheet.column_dimensions[col].width = 18
        book.save(directory / 'order.xlsx')
    elif task == 'cap06':
        from docx import Document
        doc = Document(); doc.add_heading('Weekly Update', 0)
        for line in ['Owner: Mei', 'Atlas: Draft', 'Birch: Draft']: doc.add_paragraph(line)
        doc.save(directory / 'weekly.docx')
    elif task == 'cap07':
        from pptx import Presentation
        deck = Presentation()
        for title, body in [('Briefing', 'Owner: Mei'), ('Findings', 'Atlas completed 3 milestones.'), ('Obsolete', 'Remove this slide.')]:
            slide = deck.slides.add_slide(deck.slide_layouts[1]); slide.shapes.title.text = title
            slide.placeholders[1].text = body
        deck.save(directory / 'briefing.pptx')
    elif task == 'cap08':
        (directory / 'policy-old.txt').write_text('Effective: 2026-09-01\nReimbursement: 80% of cost\nMaximum: 100.00\n')
        (directory / 'policy-new.txt').write_text('Effective: 2026-10-02\nThis policy replaces the previous policy.\nReimbursement: 60% of cost\nMaximum: 90.00\n')
    elif task == 'cap09':
        folder = directory / 'CapFiles'; folder.mkdir()
        for name, content in [('draft.txt', 'Meeting summary\n'), ('todo.txt', 'Review Atlas\n'), ('keep.txt', 'Do not modify\n')]:
            (folder / name).write_text(content)
    elif task == 'cap10':
        from openpyxl import Workbook
        book = Workbook(); sheet = book.active; sheet.title = 'Sales'
        for row in [('ID', 'Owner', 'Revenue'), ('A1', 'Mei', 120), ('B2', 'Nia', 360),
                    ('C3', 'Omar', 240), ('D4', 'Lina', 180), ('Total', None, None)]: sheet.append(row)
        for col in 'ABC': sheet.column_dimensions[col].width = 18
        book.save(directory / 'sales.xlsx')


def input_sources():
    """Validate the fixed public fixture corpus and expose its freeze inputs."""
    manifest = INPUT_ROOT / 'manifest.json'
    index = json.loads(manifest.read_text())
    if (index.get('host_expected_outputs_included') is not False
            or set(index.get('tasks', {})) != set(BASIC_TASKS)):
        raise ValueError('Invalid public fixture corpus')
    sources = [manifest]
    for members in index['tasks'].values():
        for relative, digest in members.items():
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts or '\\' in relative:
                raise ValueError('Unsafe fixture member')
            # Resolve the task explicitly below; all members must match bytes.
    for task, members in index['tasks'].items():
        for relative, digest in members.items():
            path = INPUT_ROOT / task / relative
            if sha256(path) != digest:
                raise ValueError('Public fixture changed: ' + task + '/' + relative)
            sources.append(path)
    return sources + boundary.input_sources()


def prepare(task, run, output, bundle=None):
    if task in boundary.TASKS:
        return boundary.prepare(task, run, output, bundle)
    if task not in TASKS or not re.fullmatch(r'[a-zA-Z0-9_-]{8,60}', run):
        raise ValueError('Invalid capability attempt')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    staged = ROOT / 'apps/computer_use/.runtime/windows/shared' / (run + '-inputs'); staged.mkdir(exist_ok=False)
    input_sources()
    members = json.loads((INPUT_ROOT / 'manifest.json').read_text())['tasks'][task]
    (staged / 'materials').mkdir()
    for relative in members:
        destination = staged / 'materials' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(INPUT_ROOT / task / relative, destination)
    hashes = {str(p.relative_to(staged / 'materials')).replace('\\', '/'): sha256(p)
              for p in (staged / 'materials').rglob('*') if p.is_file()}
    (staged / 'material-hashes.json').write_text(json.dumps(hashes, indent=2))
    fixture_sha = sha256(ROOT / 'apps/computer_use/vm/fixture.html')
    script = r'''$ErrorActionPreference='Stop'
$result=@{kind='windows_capability_initialization';run='RUN_ID';passed=$false;task='TASK_ID';legacy_application_entries=$true}
try {
    $staged='\\host.lan\Data\RUN_ID-inputs'
    if (Test-Path 'C:\Qw3Computer\application-entries.json') {throw 'Restore a fresh legacy entry snapshot'}
    if ((Get-FileHash 'C:\Qw3Computer\fixture.html' -Algorithm SHA256).Hash.ToLowerInvariant() -cne 'FIXTURE_SHA') {throw 'Offline fixture differs from reviewed source'}
    $hashes=Get-Content "$staged\material-hashes.json" -Raw | ConvertFrom-Json
    foreach ($entry in $hashes.PSObject.Properties) {
        $source=Join-Path "$staged\materials" $entry.Name
        $target=Join-Path "$env:USERPROFILE\Desktop" $entry.Name
        if (Test-Path $target) {throw 'Material already exists'}
        if ((Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant() -cne $entry.Value) {throw 'Material hash mismatch'}
        New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
        Copy-Item -LiteralPath $source -Destination $target
    }
    $result.material_count=@($hashes.PSObject.Properties).Count
    $result.fixture_sha256='FIXTURE_SHA'
    $result.passed=$true
} catch {$result.error=$_.Exception.ToString()}
$result | ConvertTo-Json -Depth 5 | Set-Content '\\host.lan\Data\RUN_ID-setup-result.json' -Encoding UTF8
'''.replace('RUN_ID', run).replace('TASK_ID', task).replace('FIXTURE_SHA', fixture_sha)
    (output / 'initialize.ps1').write_text(script)
    prompt = DEFINITIONS[task][3] + ' Use only the computer tools. Verify the saved result before finishing.'
    manifest = dict(task=task, run=run, instruction=prompt, formal_run=False,
                    suite='custom_capability_calibration', osworld_official_task=False,
                    inputs=hashes, fixture_sha256=fixture_sha, legacy_application_entries=True,
                    public_inputs_manifest_sha256=sha256(INPUT_ROOT / 'manifest.json'))
    (output / 'prompt.txt').write_text(prompt)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def text(path):
    return path.read_text(encoding='utf-8-sig').replace('\r\n', '\n').rstrip('\n')


def formula(value):
    return str(value).upper().replace(' ', '').replace('$', '')


def grade(task, artifacts):
    if task in boundary.TASKS:
        return boundary.grade(task, artifacts)
    artifacts = Path(artifacts)
    target = artifacts / DEFINITIONS[task][2]
    checks = {'output_saved': target.is_file()}
    report = dict(task=task, suite='custom_capability_calibration', osworld_official_task=False,
                  formal_run=False, strict_success=False, checks=checks)
    try:
        if task == 'cap09':
            for name, expected in [('summary.txt', 'Meeting summary'), ('archived-todo.txt', 'Review Atlas'), ('keep.txt', 'Do not modify')]:
                path = artifacts / name
                checks[name] = path.is_file() and text(path) == expected
            checks['original_names_removed'] = not any((artifacts / name).exists() for name in ['old-draft.txt', 'old-todo.txt'])
        elif not target.is_file():
            report['failure'] = 'missing_saved_output'
        elif task in {'cap01', 'cap02', 'cap03', 'cap08'}:
            actual = text(target)
            expected = {
                'cap01': 'QW3 computer use\nLocal Windows calibration.',
                'cap02': 'Project note\nOwner: Mei\nStatus: Approved\nHours: 8',
                'cap03': 'QW3-WINDOWS-2026-ALPHA',
                'cap08': 'Reference: QW3-WINDOWS-2026-ALPHA\nPolicy: 2026-10-02\nAmount: 75.00\nStatus: Approved',
            }[task]
            checks['exact_text'] = actual == expected
            checks['correct_lines'] = sum(a == b for a, b in zip(actual.splitlines(), expected.splitlines())) / len(expected.splitlines())
            if task == 'cap02':
                original = artifacts / 'original-draft.txt'
                checks['original_preserved'] = original.is_file() and text(original) == 'Project note\nOwner: Mei\nStatus: Draft\nHours: 6'
        elif task == 'cap04':
            actual = json.loads(target.read_text(encoding='utf-8-sig'))
            for key, value in dict(name='Nia', reference='QC-042', priority='Low', confirmed=True).items():
                checks[key] = type(actual.get(key)) is type(value) and actual.get(key) == value
            checks['no_extra_fields'] = set(actual) == {'name', 'reference', 'priority', 'confirmed'}
        elif task in {'cap05', 'cap10'}:
            from openpyxl import load_workbook
            book = load_workbook(target, data_only=False); sheet = book.active
            cached = load_workbook(target, data_only=True).active
            if task == 'cap05':
                expected = [('Item', 'Quantity', 'Unit price'), ('Pens', 2, 12), ('Paper', 3, 8), ('Folder', 1, 15)]
                checks['inputs_preserved'] = [tuple(sheet.cell(r, c).value for c in range(1, 4)) for r in range(1, 5)] == expected
                for r, total in [(2, 24), (3, 24), (4, 15)]:
                    checks[f'row_{r}_formula'] = formula(sheet[f'D{r}'].value) in {f'=B{r}*C{r}', f'=C{r}*B{r}'}
                    checks[f'row_{r}_value'] = cached[f'D{r}'].value == total
                checks['sum_formula'] = formula(sheet['D5'].value) == '=SUM(D2:D4)'
                checks['sum_value'] = cached['D5'].value == 63
                checks['labels_preserved'] = sheet['D1'].value == 'Total' and sheet['A5'].value == 'Grand total'
            else:
                checks['headers_preserved'] = tuple(sheet.cell(1, c).value for c in range(1, 4)) == ('ID', 'Owner', 'Revenue')
                checks['sorted_records'] = [tuple(sheet.cell(r, c).value for c in range(1, 4)) for r in range(2, 6)] == [
                    ('B2', 'Nia', 360), ('C3', 'Omar', 240), ('D4', 'Lina', 180), ('A1', 'Mei', 120)]
                checks['sum_formula'] = formula(sheet['C6'].value) == '=SUM(C2:C5)'
                checks['sum_value'] = cached['C6'].value == 900
        elif task == 'cap06':
            from docx import Document
            actual = [p.text for p in Document(target).paragraphs]
            checks['exact_paragraphs'] = actual == ['Weekly Report', 'Owner: Mei', 'Atlas: Approved', 'Birch: Approved']
            checks['correct_paragraphs'] = sum(a == b for a, b in zip(actual, ['Weekly Report', 'Owner: Mei', 'Atlas: Approved', 'Birch: Approved'])) / 4
        elif task == 'cap07':
            from capability_regrade import grade_presentation
            evaluated = grade_presentation(target)
            checks.update(evaluated['checks'])
            if evaluated.get('failure'):
                raise ValueError(evaluated.get('error', evaluated['failure']))
            report.update(grader_revision=evaluated['grader_revision'],
                          actual_titles=evaluated['actual_titles'],
                          actual_body_texts=evaluated['actual_body_texts'])
        # Fractional milestones describe partial correctness; they never weaken success.
        boolean_checks = [value for value in checks.values() if type(value) is bool]
        report['strict_success'] = bool(boolean_checks) and all(boolean_checks)
        report['partial_check_fraction'] = sum(boolean_checks) / len(boolean_checks)
        if target.is_file(): report['output_sha256'] = sha256(target)
    except Exception as exc:
        report.update(failure='invalid_saved_output', error=f'{type(exc).__name__}: {exc}')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate public calibration inputs only')
    parser.add_argument('--public-inputs', choices=TASKS, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    public_inputs(args.public_inputs, args.directory)
