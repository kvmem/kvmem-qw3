import importlib.util,json,hashlib,tempfile,unittest
from pathlib import Path
from osworld_windows import task006 as m
@unittest.skipUnless(importlib.util.find_spec('fitz') and m.ORIGINALS.exists(), 'requires qualified evaluator Python and frozen original CVs')
class Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.expected=m.expected_rules()[0]['interview'];self.manifest=json.loads((m.ORIGINALS/'manifest.json').read_text())
 def fixture(self,files,subfolder=False):
  # Synthetic host-only collector receipts. Never staged into the Windows VM.
  d=self.root/'artifacts';d.mkdir();entries=[];binary=[];source='C:\\Users\\Qw3Test\\Desktop\\interview'
  if files:(d/'files').mkdir()
  for i,(name,data) in enumerate(files.items()):
   export=f'item-{i:03d}.bin';(d/'files'/export).write_bytes(data);entries.append({'name':name,'type':'file','export_name':export});binary.append({'name':export,'source':source+'\\'+name,'status':'copied','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
  if subfolder:entries.append({'name':'nested','type':'directory'})
  receipt={'kind':'native_directory_artifact_collection','source':source,'complete':True,'infrastructure_ok':True,'recursive':False,'directory_status':'present','entries':entries}
  if files:
   p=d/'files/receipt.json';p.write_text(json.dumps({'complete':True,'infrastructure_ok':True,'files':binary}));receipt['files_receipt_sha256']=m.sha256(p)
  (d/'directory-receipt.json').write_text(json.dumps(receipt));return d
 def original(self,name):
  a=next(a for a in self.manifest['applicants'] if a['name'].strip().replace(' ','_')+'.pdf'==name);return (m.ORIGINALS/a['frozen_source']).read_bytes()
 def test_all_correct_names_with_empty_bytes_not_success(self):
  r=m.grade(self.fixture({n:b'' for n in self.expected}));self.assertEqual(r['original_compatible_score'],1);self.assertFalse(r['strict_success']);self.assertTrue(all(c['content_identity'] is False for c in r['content_checks']))
 def test_eight_available_originals_score_eight_ninths(self):
  r=m.grade(self.fixture({n:self.original(n) for n in self.expected if n!='Yiren_Lin.pdf'}));self.assertAlmostEqual(r['original_compatible_score'],8/9);self.assertEqual(r['missing_filenames'],['Yiren_Lin.pdf']);self.assertTrue(all(c['content_identity'] is True for c in r['content_checks']));self.assertFalse(r['original_materials_complete']);self.assertFalse(r['strict_success'])
 def test_another_applicants_original_is_wrong_content(self):
  r=m.grade(self.fixture({'Fu_Tuna.pdf':self.original('Yao_Tu.pdf')}));self.assertEqual(r['content_checks'][0]['status'],'another_applicants_original');self.assertFalse(r['content_checks'][0]['content_identity'])
 def test_extra_names_penalized_by_original_score(self):
  r=m.grade(self.fixture({'Fu_Tuna.pdf':self.original('Fu_Tuna.pdf'),'extra.pdf':b'bad'}));self.assertAlmostEqual(r['original_compatible_score'],.5/9);self.assertEqual(r['extra_filenames'],['extra.pdf'])
 def test_unknown_transformed_pdf_not_accepted_by_filename(self):
  import fitz
  doc=fitz.open();page=doc.new_page();page.insert_text((40,40),'Unverified CV content');data=doc.tobytes();doc.close();r=m.grade(self.fixture({'Fu_Tuna.pdf':data}));self.assertIsNone(r['content_checks'][0]['content_identity']);self.assertFalse(r['strict_success'])
 def test_changed_export_rejected(self):
  d=self.fixture({'Fu_Tuna.pdf':self.original('Fu_Tuna.pdf')});(d/'files/item-000.bin').write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'bytes changed'):m.grade(d)
 def test_nested_directory_not_silently_accepted(self):
  r=m.grade(self.fixture({},subfolder=True));self.assertFalse(r['no_nested_directories']);self.assertEqual(r['original_compatible_score'],0)
if __name__=='__main__':unittest.main()
