from pathlib import Path
import tempfile
import unittest
from lxml import etree

from vm.clean_office_preferences import clean


class OfficePreferenceTests(unittest.TestCase):
    def test_qname_types_and_unrelated_preferences_survive(self):
        source = b'''<oor:items xmlns:oor="http://openoffice.org/2001/registry" xmlns:xs="http://www.w3.org/2001/XMLSchema"><item oor:path="/Misc"><prop oor:name="FirstRun" oor:type="xs:boolean"><value>false</value></prop></item><item oor:path="/Histories/PickList/ItemList"><node oor:name="file:///C:/Qw3Computer/OfficeHandlerQualification/probe-xlsx.xlsx"/></item><item oor:path="/Histories/PickList/ItemList"><node oor:name="file:///C:/Users/Qw3Test/Desktop/keep.docx"/></item></oor:items>'''
        with tempfile.TemporaryDirectory() as temp:
            original = Path(temp) / 'original.xcu'; result = Path(temp) / 'result.xcu'
            original.write_bytes(source); report = clean(original, result)
            self.assertEqual(report['removed_neutral_history_items'], 1)
            tree = etree.parse(str(result))
            self.assertEqual(tree.getroot().nsmap['xs'], 'http://www.w3.org/2001/XMLSchema')
            self.assertIn(b'keep.docx', result.read_bytes())
            self.assertIn(b'FirstRun', result.read_bytes())
            self.assertEqual(original.read_bytes(), source)

    def test_mixed_history_is_rejected_before_output(self):
        source = b'''<oor:items xmlns:oor="http://openoffice.org/2001/registry" xmlns:xs="http://www.w3.org/2001/XMLSchema"><item oor:path="/PickList"><node oor:name="file:///C:/Qw3Computer/OfficeHandlerQualification/probe.xlsx"/><node oor:name="file:///C:/keep.xlsx"/></item></oor:items>'''
        with tempfile.TemporaryDirectory() as temp:
            original = Path(temp) / 'original.xcu'; result = Path(temp) / 'result.xcu'; original.write_bytes(source)
            with self.assertRaises(ValueError): clean(original, result)
            self.assertFalse(result.exists())


if __name__ == '__main__':
    unittest.main()
