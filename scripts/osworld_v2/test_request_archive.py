import gzip
import json
from pathlib import Path
import random
import shutil
import tempfile
import threading
import unittest
from unittest import mock
import http.client
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from request_archive import GroupedRequestWriter,iter_group,read_request

@unittest.skipUnless(shutil.which('zstd'),'zstd is required')
class RequestArchiveTest(unittest.TestCase):
    def test_groups_and_partial_final_group_preserve_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);writer=GroupedRequestWriter(directory,group_size=3)
            randomizer=random.Random(0);blob=randomizer.randbytes(262144)
            requests=[b'',b'{"image":"'+blob+b'","step":1}',b'\x00\xff\n',blob+blob,b'last request']
            for index,raw in enumerate(requests,1):writer.write(f'request_{index:04d}.json.gz',raw)
            writer.close();writer.close()
            self.assertEqual(len(list(directory.glob('requests_*.zst'))),2)
            self.assertFalse(list(directory.glob('*.partial')))
            self.assertEqual([raw for group in sorted(directory.glob('requests_*.zst')) for _,raw in iter_group(group)],requests)
            for index,raw in enumerate(requests,1):self.assertEqual(read_request(directory,index),raw)
            with self.assertRaises(FileExistsError):GroupedRequestWriter(directory)

    def test_corruption_and_index_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);writer=GroupedRequestWriter(directory)
            writer.write('request_0001.json.gz',b'first');writer.close()
            index=json.loads((directory/'request_index.json').read_text())
            index['records']['request_0001.json.gz']['sha256']='wrong'
            (directory/'request_index.json').write_text(json.dumps(index))
            with self.assertRaises(RuntimeError):read_request(directory,1)
            archive=next(directory.glob('requests_*.zst'));archive.write_bytes(archive.read_bytes()[:-3])
            with self.assertRaises(RuntimeError):list(iter_group(archive))

    def test_existing_gzip_recordings_remain_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=b'original complete request\x00'
            (Path(tmp)/'request_0042.json.gz').write_bytes(gzip.compress(raw))
            self.assertEqual(read_request(tmp,42),raw)

    def test_native_relay_and_recording_preserve_request_and_response_bytes(self):
        from run_claude_pair import Trial,make_handler
        raw=b'{"messages":[], "literal":"\\u0000", "number":1.00e+2}'
        events=[{'type':'message_start','message':{'id':'test','usage':{'input_tokens':12,'output_tokens':0}}},
            {'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':'OK 中文'}},
            {'type':'message_delta','delta':{'stop_reason':'end_turn'},'usage':{'output_tokens':3}},
            {'type':'message_stop'}]
        wire=b''.join(b'data: '+json.dumps(e,ensure_ascii=False).encode()+b'\n\n' for e in events)
        received=[]
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                received.append(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200);self.send_header('Content-Type','text/event-stream')
                self.send_header('Content-Length',str(len(wire)));self.end_headers();self.wfile.write(wire)
        with tempfile.TemporaryDirectory() as tmp,mock.patch.dict('os.environ',{'OSWORLD_REQUEST_RECORDING':'zstd-groups'}):
            upstream=ThreadingHTTPServer(('127.0.0.1',0),Upstream)
            threading.Thread(target=upstream.serve_forever,daemon=True).start()
            trial=Trial(Path(tmp),'kvmem',upstream.server_port,2)
            relay=ThreadingHTTPServer(('127.0.0.1',0),make_handler(trial))
            threading.Thread(target=relay.serve_forever,daemon=True).start()
            try:
                client=http.client.HTTPConnection('127.0.0.1',relay.server_port,timeout=10)
                client.request('POST','/v1/messages?beta=true',body=raw,headers={'Content-Type':'application/json'})
                response=client.getresponse();self.assertEqual(response.status,200);self.assertEqual(response.read(),wire);client.close()
            finally:
                relay.shutdown();relay.server_close();upstream.shutdown();upstream.server_close()
                trial.artifacts.put(None);trial.writer.join(timeout=10)
            self.assertFalse(trial.writer.is_alive());self.assertIsNone(trial.writer_error)
            self.assertEqual(received,[raw]);self.assertEqual(read_request(tmp,1),raw)
            self.assertEqual(gzip.decompress((Path(tmp)/'response_0001.sse.gz').read_bytes()),wire)

    def test_recording_failure_drains_queue_instead_of_hanging(self):
        from run_claude_pair import Trial
        with tempfile.TemporaryDirectory() as tmp,mock.patch.dict('os.environ',{'OSWORLD_REQUEST_RECORDING':'gzip'}),\
             mock.patch('run_claude_pair.gzip.open',side_effect=OSError('simulated recording failure')):
            trial=Trial(Path(tmp),'qw3',1,1)
            trial.artifacts.put((Path(tmp)/'request_0001.json.gz',b'first'))
            trial.artifacts.put((Path(tmp)/'request_0002.json.gz',b'second'))
            trial.artifacts.put(None);trial.writer.join(timeout=5)
            self.assertFalse(trial.writer.is_alive());self.assertEqual(trial.artifacts.unfinished_tasks,0)
            self.assertIn('simulated recording failure',trial.writer_error)

if __name__=='__main__':unittest.main()
