"""Partial positive controls must not become negative or business evidence."""
import unittest
import json
from unittest.mock import AsyncMock

from bounded_linux_observation import BoundedOfficeLinuxDriver, chrome_tree_script, partial_controls
from qw3_computer.execution import check_observation
from qw3_computer.linux_driver import LinuxDriver


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


class BoundedOfficeDriverTests(unittest.IsolatedAsyncioTestCase):
    async def test_office_read_is_partial_once_with_a_deadline_and_unchanged_input_methods(self):
        driver=BoundedOfficeLinuxDriver('http://127.0.0.1:9','unit-office')
        driver.execute=AsyncMock(return_value={'output':json.dumps({'status':'ok','AT':
            '<frame name="Budget - Calc" />'})})
        foreground={'title':'Budget - Calc','process':'libreoffice-calc','handle':'0x1'}
        result=await driver.read_accessibility(foreground,1920,1080)
        self.assertTrue(result['truncated'])
        self.assertEqual(result['scope'],'foreground_office_chrome')
        driver.execute.assert_awaited_once_with(chrome_tree_script(foreground['title']),read_timeout=4)
        self.assertIs(BoundedOfficeLinuxDriver.dispatch,LinuxDriver.dispatch)
        self.assertIs(BoundedOfficeLinuxDriver.preflight,LinuxDriver.preflight)

    async def test_unavailable_office_does_not_retry_or_prove_absence(self):
        driver=BoundedOfficeLinuxDriver('http://127.0.0.1:9','unit-office')
        driver.execute=AsyncMock(side_effect=RuntimeError('deadline exceeded'))
        foreground={'title':'Budget - Calc','process':'libreoffice-calc','handle':'0x1'}
        result=await driver.read_accessibility(foreground,1920,1080)
        self.assertEqual(result['status'],'unavailable')
        self.assertTrue(result['truncated'])
        driver.execute.assert_awaited_once()
        self.assertFalse(check_observation([{'kind':'control_absent','text':'Saving'}],
            {'foreground':foreground,'accessibility':result},{})[0]['passed'])

    async def test_other_apps_retain_official_full_reader(self):
        driver=BoundedOfficeLinuxDriver('http://127.0.0.1:9','unit-office')
        driver.execute=AsyncMock(return_value={'output':json.dumps({'status':'ok','AT':
            '<frame name="Notes" />'})})
        foreground={'title':'Notes','process':'Gedit','handle':'0x1'}
        result=await driver.read_accessibility(foreground,1920,1080)
        self.assertEqual(result['scope'],'foreground_window')
        self.assertFalse(result['truncated'])
        script=driver.execute.call_args.args[0]
        self.assertNotIn('a.create_atspi_node=bounded',script)


if __name__=='__main__':unittest.main()
