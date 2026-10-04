import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from osworld_office_candidate_run import verify_auxiliary


class OfficeCandidateAdmissionTests(unittest.TestCase):
    def test_frozen_auxiliary_bytes_are_checked_before_model_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            framework=Path(directory)/'framework';framework.mkdir()
            files={}
            for name in ('bounded_linux_observation.py','linux_app_semantics.py'):
                p=framework/name;p.write_text('read-only helper '+name)
                files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
            manifest=Path(directory)/'acceptance.json'
            manifest.write_text(json.dumps({'framework_auxiliary_files':files}))
            args=SimpleNamespace(framework=framework,acceptance_manifest=manifest)
            verify_auxiliary(args)
            (framework/'bounded_linux_observation.py').write_text('changed after freeze')
            with self.assertRaisesRegex(ValueError,'changed'):verify_auxiliary(args)

    def test_missing_or_undeclared_helpers_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest=Path(directory)/'acceptance.json'
            args=SimpleNamespace(framework=Path(directory),acceptance_manifest=manifest)
            for files in ({},{'../outside.py':'anything'}):
                manifest.write_text(json.dumps({'framework_auxiliary_files':files}))
                with self.assertRaises(ValueError):verify_auxiliary(args)


if __name__=='__main__':unittest.main()
