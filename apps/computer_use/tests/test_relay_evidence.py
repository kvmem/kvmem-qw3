import importlib.util,json,tempfile,unittest,hashlib
from pathlib import Path
from osworld_windows.events035 import schedule
from osworld_windows import analyze_relay as m
class Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);(self.root/'relay').mkdir();_,plan=schedule();self.message=plan['batches'][2][0]
 def setup_inputs(self,*,role='tool',duplicate=False,terminal=True,partial=False):
  content=self.message['content'];content=content[:20] if partial else content
  body={'messages':[{'role':role,'content':[{'type':'text','text':'Output:\n'+json.dumps({'observation':{'observation_id':'observation-1','sha256':'image-hash','accessibility':{'status':'ok','controls':[{'name':content}]}}})}]}]}
  for tid in ['one','two'] if duplicate else ['one']:
   p=self.root/'guest/app/inference'/tid;p.mkdir(parents=True);(p/'request.json').write_text(json.dumps(body))
  key=hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
  rows=[{'event':'request_forwarding','request_index':1,'host_received_monotonic_ns':2000000000,'canonical_request_sha256':key,'latest_image_sha256':'image-hash'}]
  if terminal:rows.append({'event':'request_terminal','request_index':1,'status':'complete'})
  (self.root/'relay/requests.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
  p=self.root/'reviews/3/delivery';p.mkdir(parents=True);(p/'receipt.json').write_text(json.dumps({'batch':3,'message_ids':[self.message['messageId']],'delivery_confirmed':True,'host_confirmed_monotonic_ns':1000000000}))
 def test_exact_observation_text_is_input_not_understanding(self):
  self.setup_inputs();r=m.analyze(self.root);self.assertTrue(r['receipt_integrity_passed']);self.assertTrue(r['all_forwarded_inputs_uniquely_bound']);e=r['batches'][0]['message_exposures'][0];self.assertTrue(e['exact_full_message_in_accessibility_input']);self.assertEqual(e['first_exposure']['delivery_to_first_input_seconds'],1);self.assertFalse(e['model_reading_or_understanding_proven']);self.assertFalse(e['pixel_visibility_reviewed']);self.assertNotIn(self.message['content'],json.dumps(r))
 def test_assistant_claims_not_counted_as_observation(self):
  self.setup_inputs(role='assistant');r=m.analyze(self.root);self.assertFalse(r['batches'][0]['message_exposures'][0]['exact_full_message_in_accessibility_input'])
 def test_partial_message_remains_unknown(self):
  self.setup_inputs(partial=True);r=m.analyze(self.root);self.assertFalse(r['batches'][0]['message_exposures'][0]['exact_full_message_in_accessibility_input'])
 def test_ambiguous_inputs_not_arbitrarily_bound(self):
  self.setup_inputs(duplicate=True);r=m.analyze(self.root);self.assertFalse(r['all_forwarded_inputs_uniquely_bound']);self.assertEqual(r['binding_counts'],{'ambiguous':1});self.assertFalse(r['batches'][0]['message_exposures'][0]['exact_full_message_in_accessibility_input'])
 def test_missing_terminal_is_incomplete_receipt(self):
  self.setup_inputs(terminal=False);r=m.analyze(self.root);self.assertFalse(r['receipt_integrity_passed'])
if __name__=='__main__':unittest.main()
