"""Partial positive controls must not become negative or business evidence."""
import unittest

from bounded_linux_observation import chrome_tree_script, partial_controls
from qw3_computer.execution import check_observation


class BoundedOfficeTests(unittest.TestCase):
    def test_partial_visible_menu_is_clickable_evidence_but_never_proves_absence(self):
        foreground={'handle':'0x1','process':'libreoffice-calc','title':'Budget - Calc'}
        xml='''<frame name="Budget - Calc" xmlns:s="state" xmlns:c="component">
          <menu name="Window" s:visible="true" s:showing="true" s:enabled="true"
                c:screencoord="(40, 30)" c:size="(60, 20)"/>
          <menu-item name="Hidden" s:visible="true" s:showing="false" s:enabled="true"
                c:screencoord="(40, 30)" c:size="(60, 20)"/>
        </frame>'''
        result=partial_controls(xml,foreground,1920,1080)
        self.assertEqual([c['name'] for c in result['controls']],['Window'])
        self.assertTrue(result['truncated'])
        observation={'foreground':foreground,'accessibility':result}
        self.assertTrue(check_observation([{'kind':'control_text','text':'Window'}],observation,{})[0]['passed'])
        self.assertFalse(check_observation([{'kind':'control_absent','text':'Saving'}],observation,{})[0]['passed'])

    def test_title_is_data_and_generated_reader_has_no_input_or_document_file_access(self):
        script=chrome_tree_script("Budget';raise RuntimeError('do not execute')#")
        compile(script,'readonly-reader','exec')
        self.assertIn('a.create_atspi_node=bounded',script)
        self.assertIn('depth=a.MAX_DEPTH',script)
        for method in ('pyautogui','SendInput','getTableCell','load_workbook','open(','subprocess.Popen'):
            self.assertNotIn(method,script)


if __name__=='__main__':unittest.main()
