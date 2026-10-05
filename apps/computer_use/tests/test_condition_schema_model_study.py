import copy
import json
from pathlib import Path
import tempfile
import unittest

import condition_schema_model_study as study


class SeparateStudyTests(unittest.TestCase):
    def setUp(self):
        self.path=study.candidate.RUNTIME/'condition-schema-model-study-20261006-01/protocol.json'
        self.data=json.loads(self.path.read_text())

    def changed(self, mutation):
        data=copy.deepcopy(self.data);mutation(data)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'changed.json';path.write_text(json.dumps(data))
            with self.assertRaises(study.adapter.owned.SafetyHalt):study.validate_study(path,data['trials'][0]['id'])

    def test_two_arms_use_the_same_native_execution_and_original_profile(self):
        for arm in study.ARMS:
            row=next(r for r in self.data['trials'] if r['arm']==arm)
            data,actual,framework,original=study.validate_study(self.path,row['id'])
            self.assertEqual(actual,row)
            self.assertEqual(data['model_profile']['model_context_tokens'],524288)
            self.assertFalse(data['model_profile']['advisor'])
            self.assertEqual(data['model_profile']['kvmem'],'off')
            build=json.loads(study.adapter.checked_ref(data['control_build' if arm=='wire-control' else 'candidate_build']).read_text())
            self.assertEqual(framework,Path(data['control_build' if arm=='wire-control' else 'candidate_build']['path']).parent/'framework')
            self.assertEqual(build['candidate_files_sha256']['qw3_computer/desktop.py'],
                json.loads(study.adapter.checked_ref(data['control_build']).read_text())['candidate_files_sha256']['qw3_computer/desktop.py'])
            self.assertIn('platform_pilots.py',original['helper_sha256'])

    def test_changed_budget_missing_condition_or_arm_order_is_rejected(self):
        for mutation in [lambda d:d['trials'][0].update(seconds=361),lambda d:d['trials'].pop(),
                         lambda d:d['trials'].reverse(),lambda d:d['model_profile'].update(model_output_tokens=8192)]:
            with self.subTest(mutation=mutation):self.changed(mutation)

    def test_using_the_same_package_for_both_arms_or_skipping_parent_is_rejected(self):
        self.changed(lambda d:d.update(candidate_build=d['control_build']))
        self.changed(lambda d:d.update(required_original_native_queue_declaration=d['original_sdk_protocol']))

    def test_unqualified_transport_audit_cannot_replace_actual_candidate_evidence(self):
        self.changed(lambda d:d.update(actual_wire_transport_audit=d['wire_cpu_qualification']))

    def test_scope_restores_original_lookup_and_digest_after_exception(self):
        import native_control_model_stack as original
        before_validate=study.adapter.validate_study;before_sha=original.STUDY_SHA
        with self.assertRaisesRegex(RuntimeError,'scope marker'):
            with study.scoped_stack(self.path,self.data['trials'][0]['id']):
                self.assertIs(study.adapter.validate_study,study.validate_study)
                self.assertEqual(original.STUDY_SHA,study.candidate.digest(self.path))
                raise RuntimeError('scope marker')
        self.assertIs(study.adapter.validate_study,before_validate)
        self.assertEqual(original.STUDY_SHA,before_sha)


if __name__=='__main__':unittest.main()
