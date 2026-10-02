import json
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

from osworld_windows import capability_boundary as boundary
from osworld_windows.capabilities import INPUT_ROOT, input_sources
from osworld_windows.source import sha256


def cached_values(path, values):
    with zipfile.ZipFile(path) as archive:
        members = {n: archive.read(n) for n in archive.namelist()}
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    tree = ET.fromstring(members['xl/worksheets/sheet1.xml'])
    for cell in tree.iter(ns + 'c'):
        if cell.get('r') in values:
            value = cell.find(ns + 'v')
            if value is None: value = ET.SubElement(cell, ns + 'v')
            value.text = str(values[cell.get('r')])
    members['xl/worksheets/sheet1.xml'] = ET.tostring(tree)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, body in members.items(): archive.writestr(name, body)


def correct_control(task, root):
    """Private evaluator control only; never enters public inputs or VM."""
    from openpyxl import load_workbook
    source = boundary.INPUT_ROOT / task / boundary.DEFINITIONS[task][1]
    shutil.copyfile(source, root / ('original-' + source.name))
    book = load_workbook(source); sheet = book.active
    if task == 'cap11':
        values = {}
        for r in range(2, 14):
            sheet.cell(r, 4, f'=B{r}*C{r}')
            values[f'D{r}'] = sheet.cell(r, 2).value * sheet.cell(r, 3).value
        sheet['D14'] = '=SUM(D2:D13)'; values['D14'] = sum(values.values())
    else:
        rows = sorted(list(sheet.values)[1:13], key=lambda r: (-r[2], r[0]))
        for r, row in enumerate(rows, 2):
            for c, value in enumerate(row, 1): sheet.cell(r, c, value)
        sheet['C14'] = '=SUM(C2:C13)'; values = {'C14': sum(row[2] for row in rows)}
    target = root / (task + '.xlsx'); book.save(target); book.close()
    cached_values(target, values)
    return target, values


class BoundaryControls(unittest.TestCase):
    def test_original_ten_corpus_retains_its_published_bytes(self):
        self.assertEqual(sha256(INPUT_ROOT / 'manifest.json'), '3879517805f8a99685a73cb1e477fb712b87efc36c2d5e61eb31b80955018d97')
        self.assertTrue(all(p.is_file() for p in input_sources()))
        manifest = json.loads((boundary.INPUT_ROOT / 'manifest.json').read_text())
        self.assertFalse(manifest['host_expected_outputs_included'])
        self.assertEqual(set(manifest['tasks']), {'cap11', 'cap12'})

    def test_missing_and_unchanged_workbooks_fail(self):
        for task in boundary.TASKS:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.assertFalse(boundary.grade(task, root)['strict_success'])
                source = boundary.INPUT_ROOT / task / boundary.DEFINITIONS[task][1]
                shutil.copyfile(source, root / ('original-' + source.name))
                shutil.copyfile(source, root / (task + '.xlsx'))
                self.assertFalse(boundary.grade(task, root)['strict_success'])

    def test_all_twelve_formulas_and_original_are_required(self):
        from openpyxl import load_workbook
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target, values = correct_control('cap11', root)
            self.assertTrue(boundary.grade('cap11', root)['strict_success'])
            book = load_workbook(target); book.active['D13'] = values['D13']
            book.save(target); book.close(); cached_values(target, values)
            result = boundary.grade('cap11', root)
            self.assertFalse(result['strict_success'])
            self.assertFalse(result['checks']['formula_13'])
            target, _ = correct_control('cap11', root)
            (root / 'original-order12.xlsx').write_bytes(b'changed')
            self.assertFalse(boundary.grade('cap11', root)['strict_success'])

    def test_tied_sort_and_all_record_fields_are_required(self):
        from openpyxl import load_workbook
        for mutation in ('tie', 'region', 'total'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); target, values = correct_control('cap12', root)
                self.assertTrue(boundary.grade('cap12', root)['strict_success'])
                book = load_workbook(target); sheet = book.active
                if mutation == 'tie':
                    rows = [tuple(sheet.cell(r, c).value for c in range(1, 5)) for r in (3, 4)]
                    for r, row in zip((3, 4), reversed(rows)):
                        for c, value in enumerate(row, 1): sheet.cell(r, c, value)
                elif mutation == 'region': sheet['D2'] = 'wrong region'
                else: sheet['C14'] = '=SUM(C2:C12)'
                book.save(target); book.close(); cached_values(target, values)
                self.assertFalse(boundary.grade('cap12', root)['strict_success'])
