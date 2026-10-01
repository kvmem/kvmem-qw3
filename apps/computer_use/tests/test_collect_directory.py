import importlib.util,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import collect_directory as m
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.desktop=self.root/'Desktop';self.desktop.mkdir();self.source=self.desktop/'interview';self.output=self.root/'export'
 def run_collect(self):return m.collect_directory(self.source,self.output,allowed_roots=[self.desktop])
 def test_missing_is_task_failure_not_infrastructure(self):
  r=self.run_collect();self.assertTrue(r['infrastructure_ok']);self.assertEqual(r['directory_status'],'missing')
 def test_empty_folder_is_valid_snapshot(self):
  self.source.mkdir();r=self.run_collect();self.assertTrue(r['infrastructure_ok']);self.assertEqual(r['entries'],[])
 def test_preserves_names_and_copies_bytes_not_recursive(self):
  self.source.mkdir();(self.source/'张 三.pdf').write_bytes(b'%PDF-test');(self.source/'notes.txt').write_bytes(b'notes');(self.source/'subdirectory').mkdir();(self.source/'subdirectory'/'ignored.pdf').write_bytes(b'not exported')
  r=self.run_collect();self.assertTrue(r['infrastructure_ok']);self.assertEqual({i['name'] for i in r['entries']},{'张 三.pdf','notes.txt','subdirectory'});files=json.loads((self.output/'files/receipt.json').read_text());self.assertEqual(len(files['files']),2)
  for e in r['entries']:
   if e['type']=='file':self.assertEqual((self.output/'files'/e['export_name']).read_bytes(),(self.source/e['name']).read_bytes())
 def test_reparse_member_is_not_followed(self):
  self.source.mkdir();secret=self.root/'outside';secret.write_bytes(b'outside');(self.source/'escape.pdf').symlink_to(secret);r=self.run_collect();self.assertFalse(r['infrastructure_ok']);self.assertFalse((self.output/'files').exists())
 def test_output_under_source_rejected(self):
  self.source.mkdir();self.output=self.source/'bad'
  with self.assertRaises(ValueError):self.run_collect()
 def test_changed_folder_is_infrastructure_error(self):
  self.source.mkdir();(self.source/'a.pdf').write_bytes(b'a');real=m.collect
  def altered(*args,**kwargs):
   r=real(*args,**kwargs);(self.source/'new.pdf').write_bytes(b'late');return r
  with patch.object(m,'collect',side_effect=altered):r=self.run_collect()
  self.assertFalse(r['infrastructure_ok']);self.assertIn('changed',r['error'])
 def test_limits_do_not_silently_truncate(self):
  self.source.mkdir()
  for i in range(65):(self.source/str(i)).write_bytes(b'')
  r=self.run_collect();self.assertFalse(r['infrastructure_ok']);self.assertFalse((self.output/'files').exists())
if __name__=='__main__':unittest.main()
