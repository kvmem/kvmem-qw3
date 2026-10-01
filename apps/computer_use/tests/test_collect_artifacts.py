import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from collect_artifacts import CHUNK, collect


class BinaryCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()

    def run_collection(self, source, **options):
        return collect({"files": [{"name": "result.bin", "source": str(source), **options}]},
                       self.root / "export", allowed_roots=[self.inputs])

    def test_binary_larger_than_legacy_text_limit(self):
        source = self.inputs / "binary.xcf"
        payload = bytes(range(256)) * 8193
        source.write_bytes(payload)
        before = source.stat()
        report = self.run_collection(source)
        self.assertTrue(report["infrastructure_ok"])
        self.assertEqual((self.root / "export/result.bin").read_bytes(), payload)
        self.assertEqual(report["files"][0]["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(source.stat().st_mtime_ns, before.st_mtime_ns)

    def test_missing_deliverable_is_evidence_not_collection_failure(self):
        report = self.run_collection(self.inputs / "absent.xlsx")
        self.assertTrue(report["infrastructure_ok"])
        self.assertEqual(report["files"][0]["status"], "missing")
        self.assertFalse((self.root / "export/result.bin").exists())

    def test_outside_path_is_not_copied(self):
        source = self.root / "outside"
        source.write_bytes(b"outside")
        report = self.run_collection(source)
        self.assertFalse(report["infrastructure_ok"])
        self.assertEqual(report["files"][0]["status"], "rejected_path")

    def test_symlink_cannot_escape_task_root(self):
        source = self.root / "outside"
        source.write_bytes(b"outside")
        link = self.inputs / "link"
        try:
            link.symlink_to(source)
        except OSError as exc:
            self.skipTest(str(exc))
        report = self.run_collection(link)
        self.assertEqual(report["files"][0]["status"], "rejected_path")

    def test_oversized_file_is_not_a_partial_deliverable(self):
        source = self.inputs / "large"
        source.write_bytes(b"abc")
        report = self.run_collection(source, max_bytes=2)
        self.assertEqual(report["files"][0]["status"], "size_limit")
        self.assertFalse((self.root / "export/result.bin").exists())

    def test_concurrent_source_change_is_not_graded_as_saved_output(self):
        source = self.inputs / "changing"
        source.write_bytes(b"a" * (CHUNK * 2))
        original_open = Path.open

        class ChangingReader:
            def __init__(self, raw):
                self.raw = raw
                self.changed = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.raw.close()

            def fileno(self):
                return self.raw.fileno()

            def read(self, count):
                block = self.raw.read(count)
                if not self.changed:
                    self.changed = True
                    source.write_bytes(b"changed")
                return block

        def opened(path, mode="r", *args, **kwargs):
            raw = original_open(path, mode, *args, **kwargs)
            return ChangingReader(raw) if path == source and mode == "rb" else raw

        with patch.object(Path, "open", opened):
            report = self.run_collection(source)
        self.assertEqual(report["files"][0]["status"], "changed_during_copy")
        self.assertFalse(report["infrastructure_ok"])
        self.assertFalse((self.root / "export/result.bin").exists())
        self.assertTrue((self.root / "export/result.bin.part").is_file())

    def test_export_names_and_receipt_cannot_collide(self):
        for names in [("one", "ONE"), ("receipt.json",), ("one.part",), ("NUL",), ("one.",)]:
            with self.subTest(names=names):
                with self.assertRaises(ValueError):
                    collect({"files": [{"name": n, "source": str(self.inputs / "missing")} for n in names]},
                            self.root / "export", allowed_roots=[self.inputs])


if __name__ == "__main__":
    unittest.main()
