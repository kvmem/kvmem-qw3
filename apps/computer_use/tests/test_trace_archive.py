import json
from pathlib import Path
import tempfile
import unittest
from osworld_linux.archive import archive_completed, read_request


class ArchiveTests(unittest.TestCase):
    def test_complete_requests_roundtrip_and_reuse_images_without_touching_live_request(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);image='data:image/png;base64,'+'abcd'*10000
            for name in ['first','second','live']:
                trace=root/name;trace.mkdir();body={'messages':[{'url':image,'text':'中文'}],'collision':{'blob':'literal-user-data'},'primitive':[None,True,1.0]}
                (trace/'request.json').write_text(json.dumps(body,ensure_ascii=False))
                if name!='live':(trace/'metrics.json').write_text('{}')
            original=(root/'first/request.json').read_bytes()
            self.assertEqual(archive_completed(root),2)
            self.assertEqual(read_request(root/'first'),original)
            self.assertEqual(read_request(root/'second'),original)
            self.assertTrue((root/'live/request.json').exists())
            self.assertEqual(len(list((root/'blobs').iterdir())),1)
            blob=next((root/'blobs').iterdir());blob.write_bytes(b'corrupt')
            with self.assertRaises(Exception):read_request(root/'first')
