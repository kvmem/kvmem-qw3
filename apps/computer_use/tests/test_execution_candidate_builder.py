import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from build_execution_candidate import build
from osworld_windows.bundle import verify_bundle


SOURCE = Path(__file__).resolve().parents[1] / "qw3_computer"


class IsolatedBundleTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "qualified"
        package = source / "app" / "qw3_computer"
        package.mkdir(parents=True)
        for name in ("app.py", "desktop.py", "linux_driver.py"):
            (package / name).write_bytes((SOURCE / name).read_bytes())
        tests = source / "app" / "tests"
        tests.mkdir()
        (tests / "test_desktop.py").write_bytes((SOURCE.parent / "tests" / "test_desktop.py").read_bytes())
        (source / "runtime.bin").write_bytes(b"immutable test runtime")
        manifest = {"source_commit": "test-fixture", "source_files": {
            str(p.relative_to(source / "app")): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source / "app").rglob("*") if p.is_file()}}
        (source / "manifest.json").write_text(json.dumps(manifest))
        archive = root / "qualified.zip"
        with zipfile.ZipFile(archive, "w") as writer:
            for path in source.rglob("*"):
                if path.is_file():writer.write(path, str(path.relative_to(source)))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (source / "archive.json").write_text(json.dumps({"path": str(archive),
            "sha256": digest, "bytes": archive.stat().st_size}))
        (source / "qualification.json").write_text(json.dumps({"passed": True,
            "bundle_sha256": digest, "native_result": {"passed": True,
                "process_exit_code": 0, "bundle_sha256": digest}}))
        return source

    def test_patch_unlinks_shared_files_and_preserves_qualified_source_and_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);source = self.fixture(root)
            before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
            output, archive = root / "candidate", root / "candidate.zip"
            result = build(source, output, archive)
            self.assertFalse(result["qualified"])
            self.assertEqual(result["real_model_trials"], 0)
            self.assertEqual({str(p.relative_to(source)): p.read_bytes()
                for p in source.rglob("*") if p.is_file()}, before)
            verify_bundle(source)
            manifest = json.loads((output / "manifest.json").read_text())
            with zipfile.ZipFile(archive) as reader:
                self.assertEqual(reader.read("manifest.json"), (output / "manifest.json").read_bytes())
                for path, expected in manifest["source_files"].items():
                    self.assertEqual(hashlib.sha256(reader.read("app/" + path)).hexdigest(), expected)
            self.assertFalse((output / "qualification.json").exists())
            with self.assertRaises(FileNotFoundError):verify_bundle(output)

    def test_existing_or_nested_output_never_replaces_qualified_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);source = self.fixture(root)
            with self.assertRaises(ValueError):build(source, source / "nested", root / "candidate.zip")
            output = root / "existing";output.mkdir()
            with self.assertRaises(FileExistsError):build(source, output, root / "candidate.zip")
            archive = root / "existing.zip";archive.write_bytes(b"prior attempt")
            with self.assertRaises(FileExistsError):build(source, root / "candidate", archive)
            self.assertEqual(archive.read_bytes(), b"prior attempt")


if __name__ == "__main__":
    unittest.main()
