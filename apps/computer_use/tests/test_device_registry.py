import json
from pathlib import Path
import tempfile
import unittest

from qw3_computer.device_registry import load_registry


class RegistryTests(unittest.TestCase):
    def test_registry_rejects_duplicate_endpoint_and_path_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'registry.json'
            item = {'id':'linux','label':'Linux VM','platform':'linux','endpoint':'http://127.0.0.1:18130','token_file':'token.txt'}
            for second in (item | {'id':'android','platform':'android'}, item | {'id':'../escape','endpoint':'http://127.0.0.1:18131'}):
                path.write_text(json.dumps({'devices':[item,second]}))
                with self.assertRaises(ValueError): load_registry(path)
            path.write_text(json.dumps({'devices':[item]}))
            result = load_registry(path)
            self.assertEqual(result['linux']['token_file'], Path(directory)/'token.txt')

    def test_model_cannot_expand_registry_to_external_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'registry.json'
            for endpoint in ['https://example.com:123','http://user:secret@127.0.0.1:123','http://127.0.0.1:123/act','http://127.0.0.1:123?override=1']:
                path.write_text(json.dumps({'devices':[{'id':'dev','label':'Device','platform':'android','endpoint':endpoint,'token_file':'token.txt'}]}))
                with self.assertRaises(ValueError): load_registry(path)
