"""Initial broken input and post-success resets have distinct provenance."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from campaign_interlude import pilot_spec
from software_product_run import validate_repair_fixture
from osworld_windows.source import sha256


class RepairInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.path=self.root/'fixture.json'
        self.fixture=dict(status='ready_fault_injected',workspace='C:\\Qw3Computer\\fixture-one',fixture_source_sha256='source-hash')
        self.definition=dict(workspace=self.fixture['workspace'],reset_source_sha256='source-hash')

    def tearDown(self):self.temp.cleanup()

    def write(self,value):self.path.write_text(json.dumps(value))

    def test_original_broken_input_uses_exact_bytes_without_claiming_reset(self):
        self.write(self.fixture)
        definition=self.definition|dict(input_mode='initial',initial_sha256=sha256(self.path))
        self.assertEqual(validate_repair_fixture(self.path,definition),self.fixture)
        self.write(self.fixture|dict(extra='changed'))
        with self.assertRaises(ValueError):validate_repair_fixture(self.path,definition)

    def test_repetition_requires_the_declared_original_reset_marker(self):
        definition=self.definition|dict(input_mode='reset',sha256='original-hash')
        self.write(self.fixture)
        with self.assertRaises(ValueError):validate_repair_fixture(self.path,definition)
        self.write(self.fixture|dict(reset_from_sha256='original-hash'))
        self.assertEqual(validate_repair_fixture(self.path,definition)['reset_from_sha256'],'original-hash')

    def test_invalid_fixture_is_rejected_by_interlude_before_vm_or_scheduler_work(self):
        self.write(self.fixture|dict(status='failed'))
        args=SimpleNamespace(case='windows_python_repair',bundle=self.root/'bundle',engine=self.root/'engine',
            ffmpeg=self.root/'ffmpeg',repair_fixture=self.path,acceptance_manifest=None,
            seconds=900,output=self.root/'repair-attempt-01')
        with self.assertRaisesRegex(ValueError,'fixture is not ready'):pilot_spec(args)

    def test_other_workspace_or_fixture_source_cannot_pass(self):
        for delta in [dict(workspace='C:\\Qw3Computer\\other-project'),dict(fixture_source_sha256='other-source')]:
            self.write(self.fixture|delta)
            with self.assertRaises(ValueError):validate_repair_fixture(self.path,self.definition|dict(sha256=None))


if __name__=='__main__':unittest.main()
