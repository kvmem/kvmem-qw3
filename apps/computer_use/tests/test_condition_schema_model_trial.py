import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import condition_schema_model_trial as trial


class SeparateTrialTests(unittest.TestCase):
    def write_wire(self,directory,repair=True,status='completed'):
        source=trial.study.candidate.RUNTIME/'condition-schema-transport-20261006-03/linux-verified/fixture-backend-request.json'
        body=json.loads(source.read_text())
        act=next(t for t in body['tools'] if t['function']['name']=='mcp__computer__act')
        if not repair:act['function']['parameters']['properties']['step']['properties']['conditions']['items']={}
        trace=directory/'one';trace.mkdir();(trace/'request.json').write_text(json.dumps(body))
        (trace/'metrics.json').write_text(json.dumps(dict(status=status)))
        return body

    def test_actual_preserved_wire_schema_recognized_for_both_arms(self):
        for repair,arm,shape in [(False,'wire-control','opaque'),(True,'condition-repair','trusted-full-definition')]:
            with tempfile.TemporaryDirectory() as directory:
                self.write_wire(Path(directory),repair)
                rows=trial.wire_conditions(directory,arm)
                self.assertEqual(len(rows),1);self.assertEqual(rows[0]['condition_shape'],shape)

    def test_repair_missing_definition_and_nonterminal_request_are_rejected(self):
        for repair,status in [(False,'completed'),(True,'in_progress')]:
            with tempfile.TemporaryDirectory() as directory:
                self.write_wire(Path(directory),repair,status)
                with self.assertRaises(trial.study.adapter.owned.SafetyHalt):trial.wire_conditions(directory,'condition-repair')

    def test_absent_model_requests_do_not_count_as_a_trial(self):
        with tempfile.TemporaryDirectory() as directory,self.assertRaises(trial.study.adapter.owned.SafetyHalt):
            trial.wire_conditions(directory,'condition-repair')

    def test_original_profile_and_permissions_still_apply(self):
        for mutation in [lambda b:b.update(max_tokens=8192),lambda b:b.update(enable_thinking=False),lambda b:b['tools'].pop()]:
            with tempfile.TemporaryDirectory() as directory:
                self.write_wire(Path(directory));p=Path(directory)/'one/request.json';body=json.loads(p.read_text());mutation(body);p.write_text(json.dumps(body))
                with self.assertRaises(trial.study.adapter.owned.SafetyHalt):trial.wire_conditions(directory,'condition-repair')

    def test_existing_attempt_is_never_modified_even_on_error(self):
        study_path=trial.study.candidate.RUNTIME/'condition-schema-model-study-20261006-01/protocol.json'
        protocol=json.loads(study_path.read_text());row=protocol['trials'][0]
        with tempfile.TemporaryDirectory(dir=trial.study.candidate.RUNTIME) as directory:
            output=Path(directory);marker=output/'marker';marker.write_text('original')
            args=argparse.Namespace(study=study_path,trial_id=row['id'],output=output)
            with patch.object(trial.original,'run') as run,self.assertRaises(trial.study.adapter.owned.SafetyHalt):trial.run(args)
            run.assert_not_called();self.assertEqual(list(output.iterdir()),[marker]);self.assertEqual(marker.read_text(),'original')

    def test_original_failed_grade_is_preserved_by_wrapper(self):
        study_path=trial.study.candidate.RUNTIME/'condition-schema-model-study-20261006-01/protocol.json'
        protocol=json.loads(study_path.read_text());row=protocol['trials'][0]
        _,_,framework,_=trial.study.validate_study(study_path,row['id'])
        with tempfile.TemporaryDirectory(dir=trial.study.candidate.RUNTIME) as directory:
            output=Path(directory)/'fresh';args=argparse.Namespace(study=study_path,trial_id=row['id'],output=output)
            grade={'accepted':False,'artifact_passed':False}
            def delegate(args):
                args.output.mkdir();(args.output/'stack').mkdir()
                result={'outcome':grade,'status':'completed','termination_confirmed':True}
                (args.output/'result.json').write_text(json.dumps(result))
                (args.output/'stack/stack.json').write_text(json.dumps(dict(framework=str(framework),services={'client':{'runtime':str(output/'client')}})))
                return result
            with patch.object(trial.original,'run',side_effect=delegate),patch.object(trial,'wire_conditions',return_value=[{'fixture':True}]):
                result=trial.run(args)
            receipt=json.loads((output/'condition-trial-receipt.json').read_text())
            self.assertEqual(result['outcome'],grade);self.assertEqual(receipt['original_business_outcome'],grade)
            self.assertFalse(receipt['business_improvement_proven'])


if __name__=='__main__':unittest.main()
