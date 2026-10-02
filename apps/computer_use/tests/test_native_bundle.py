import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from osworld_windows.bundle import checked_path, prepare, verify_bundle


class NativeBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle = self.root / 'client'
        (self.bundle / 'app').mkdir(parents=True)
        (self.bundle / 'app/runtime.py').write_text('neutral source')
        manifest = dict(source_files={'runtime.py': hashlib.sha256(b'neutral source').hexdigest()})
        (self.bundle / 'manifest.json').write_text(json.dumps(manifest))
        self.archive = self.root / 'client.zip'
        with zipfile.ZipFile(self.archive, 'w') as writer:
            writer.write(self.bundle / 'manifest.json', 'manifest.json')
            writer.write(self.bundle / 'app/runtime.py', 'app/runtime.py')
        self.bind_archive()

    def tearDown(self):
        self.temporary.cleanup()

    def bind_archive(self):
        digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        (self.bundle / 'archive.json').write_text(json.dumps(dict(path=str(self.archive),
            sha256=digest, bytes=self.archive.stat().st_size)))
        (self.bundle / 'qualification.json').write_text(json.dumps(dict(passed=True, bundle_sha256=digest,
            native_result=dict(passed=True, process_exit_code=0, bundle_sha256=digest))))

    def test_same_snapshot_install_stages_exact_qualified_archive(self):
        share = self.root / 'share'; share.mkdir()
        proof = prepare(self.bundle, 'bundle-control-01', share, self.root / 'install')
        self.assertEqual(proof['source_files'], 1)
        self.assertEqual((share / 'bundle-control-01-bundle/client.zip').read_bytes(), self.archive.read_bytes())
        script = (self.root / 'install/install.ps1').read_text()
        self.assertIn(proof['manifest_sha256'], script)
        self.assertIn('C:\\Qw3Computer\\clients\\' + proof['archive']['sha256'][:12], script)
        self.assertIn('Installed source changed', script)

    def test_failed_or_mismatched_native_qualification_rejected(self):
        path = self.bundle / 'qualification.json'; original = json.loads(path.read_text())
        for update in ({'passed': False}, {'bundle_sha256': '0' * 64},
                       {'native_result': dict(passed=True, process_exit_code=1)}, {'native_result': {}}):
            with self.subTest(update=update):
                path.write_text(json.dumps(original | update))
                with self.assertRaises(ValueError): verify_bundle(self.bundle)

    def test_archive_local_source_and_manifest_tampering_rejected(self):
        for target in (self.archive, self.bundle / 'app/runtime.py', self.bundle / 'manifest.json'):
            before = target.read_bytes()
            target.write_bytes(before + b' ')
            with self.subTest(target=target), self.assertRaises(ValueError): verify_bundle(self.bundle)
            target.write_bytes(before)

    def test_unsafe_and_duplicate_zip_members_rejected_even_with_matching_receipt(self):
        before = self.archive.read_bytes()
        for name in ('../outside', 'C:/outside', 'app\\outside', 'APP/runtime.py'):
            self.archive.write_bytes(before)
            with zipfile.ZipFile(self.archive, 'a') as writer: writer.writestr(name, 'neutral')
            self.bind_archive()
            with self.subTest(name=name), self.assertRaises(ValueError): verify_bundle(self.bundle)

    def test_paths_and_attempt_identity_are_bounded(self):
        for name in ('', '../outside', '/outside', 'app//runtime.py', 'a:b', 'a\\b', None):
            with self.subTest(name=name), self.assertRaises(ValueError): checked_path(name)
        with self.assertRaises(ValueError): prepare(self.bundle, '../outside', self.root, self.root / 'install')


if __name__ == '__main__':
    unittest.main()
