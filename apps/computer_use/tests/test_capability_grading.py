import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

from osworld_windows.capabilities import grade, public_inputs


class CapabilityGradingTests(unittest.TestCase):
    def test_missing_output_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            for task in ['cap01', 'cap02', 'cap03', 'cap04', 'cap05', 'cap06', 'cap07', 'cap08', 'cap09', 'cap10']:
                self.assertFalse(grade(task, temp)['strict_success'])

    def test_exact_text_with_windows_line_endings_and_bom(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cap01.txt'
            path.write_bytes(b'\xef\xbb\xbfQW3 computer use\r\nLocal Windows calibration.\r\n')
            self.assertTrue(grade('cap01', temp)['strict_success'])
            path.write_text('QW3 computer use\nWrong content')
            result = grade('cap01', temp)
            self.assertFalse(result['strict_success'])
            self.assertEqual(result['checks']['correct_lines'], 0.5)

    def test_new_note_does_not_excuse_modified_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'cap02.txt').write_text('Project note\nOwner: Mei\nStatus: Approved\nHours: 8')
            (root / 'original-draft.txt').write_text('Project note\nOwner: Mei\nStatus: Draft\nHours: 6')
            self.assertTrue(grade('cap02', root)['strict_success'])
            (root / 'original-draft.txt').write_text('modified')
            self.assertFalse(grade('cap02', root)['strict_success'])

    def test_json_confirmation_must_be_boolean_and_fields_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'browser-result.json'
            data = dict(name='Nia', reference='QC-042', priority='Low', confirmed=True)
            path.write_text(json.dumps(data))
            self.assertTrue(grade('cap04', temp)['strict_success'])
            path.write_text(json.dumps(data | {'confirmed': 'true'}))
            self.assertFalse(grade('cap04', temp)['strict_success'])

    def test_file_move_requires_removing_original_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, content in [('summary.txt', 'Meeting summary'), ('archived-todo.txt', 'Review Atlas'), ('keep.txt', 'Do not modify')]:
                (root / name).write_text(content)
            self.assertTrue(grade('cap09', root)['strict_success'])
            (root / 'old-todo.txt').write_text('Review Atlas')
            self.assertFalse(grade('cap09', root)['strict_success'])

    def test_old_policy_amount_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cap08.txt'
            path.write_text('Reference: QW3-WINDOWS-2026-ALPHA\nPolicy: 2026-10-02\nAmount: 75.00\nStatus: Approved')
            self.assertTrue(grade('cap08', temp)['strict_success'])
            path.write_text(path.read_text().replace('75.00', '100.00'))
            self.assertFalse(grade('cap08', temp)['strict_success'])

    def test_unchanged_office_inputs_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for task, name, result in [('cap05', 'order.xlsx', 'cap05.xlsx'), ('cap06', 'weekly.docx', 'cap06.docx'),
                                       ('cap07', 'briefing.pptx', 'cap07.pptx'), ('cap10', 'sales.xlsx', 'cap10.xlsx')]:
                folder = root / task; public_inputs(task, folder)
                (folder / result).write_bytes((folder / name).read_bytes())
                self.assertFalse(grade(task, folder)['strict_success'])

    def test_correct_saved_office_outputs_can_pass_and_wrong_records_fail(self):
        # Private evaluator controls, never copied into the VM input corpus.
        from docx import Document
        from pptx import Presentation
        from openpyxl import load_workbook
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for task in ['cap05', 'cap06', 'cap07', 'cap10']:
                public_inputs(task, root / task)
            word = root / 'cap06'; doc = Document(word / 'weekly.docx')
            doc.paragraphs[0].text = 'Weekly Report'
            doc.paragraphs[2].text = 'Atlas: Approved'; doc.paragraphs[3].text = 'Birch: Approved'
            doc.save(word / 'cap06.docx')
            self.assertTrue(grade('cap06', word)['strict_success'])
            doc.paragraphs[1].text = 'Owner: someone else'; doc.save(word / 'cap06.docx')
            self.assertFalse(grade('cap06', word)['strict_success'])
            slides = root / 'cap07'; deck = Presentation(slides / 'briefing.pptx')
            deck.slides[1].shapes.title.text = 'Results'
            member = deck.slides._sldIdLst[-1]; deck.part.drop_rel(member.rId); deck.slides._sldIdLst.remove(member)
            deck.save(slides / 'cap07.pptx')
            self.assertTrue(grade('cap07', slides)['strict_success'])
            for task, source, formulas, cached in [
                ('cap05', 'order.xlsx', {'D2': '=B2*C2', 'D3': '=B3*C3', 'D4': '=B4*C4', 'D5': '=SUM(D2:D4)'},
                 {'D2': 24, 'D3': 24, 'D4': 15, 'D5': 63}),
                ('cap10', 'sales.xlsx', {'C6': '=SUM(C2:C5)'}, {'C6': 900})]:
                folder = root / task; book = load_workbook(folder / source); sheet = book.active
                if task == 'cap10':
                    records = sorted(list(sheet.values)[1:5], key=lambda row: row[2], reverse=True)
                    for r, row in enumerate(records, 2):
                        for c, value in enumerate(row, 1): sheet.cell(r, c).value = value
                for cell, expression in formulas.items(): sheet[cell] = expression
                target = folder / (task + '.xlsx'); book.save(target)
                # openpyxl cannot calculate; model ordinary Office cache bytes in
                # the independent control, retaining formulas in the file.
                with zipfile.ZipFile(target) as archive: members = {n: archive.read(n) for n in archive.namelist()}
                tree = ET.fromstring(members['xl/worksheets/sheet1.xml']); ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
                for cell in tree.iter(ns + 'c'):
                    if cell.get('r') in cached:
                        value = cell.find(ns + 'v')
                        if value is None: value = ET.SubElement(cell, ns + 'v')
                        value.text = str(cached[cell.get('r')])
                members['xl/worksheets/sheet1.xml'] = ET.tostring(tree)
                with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
                    for name, content in members.items(): archive.writestr(name, content)
                self.assertTrue(grade(task, folder)['strict_success'])
                broken = load_workbook(target); broken.active['A2'] = 'wrong ID'; broken.save(target)
                self.assertFalse(grade(task, folder)['strict_success'])


if __name__ == '__main__':
    unittest.main()
