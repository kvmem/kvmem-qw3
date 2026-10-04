import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import test_field_identity_candidate as seed
from completion_feedback import explain_condition
from completion_feedback_candidate import patch_execution


helper=seed.SOURCE.with_name("completion_feedback.py")
(seed.package/helper.name).write_bytes(helper.read_bytes())
path=seed.package/"feedback_execution.py"
path.write_bytes(patch_execution((seed.package/"execution.py").read_bytes()))
spec=importlib.util.spec_from_file_location("qw3_field_test.feedback_execution",path)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)


class CompletionFeedbackTests(unittest.TestCase):
    def check(self,condition,observed):
        old=seed.execution.check_observation([condition],observed,{})[0]
        original=copy.deepcopy((condition,observed,old))
        current=module.check_observation([condition],observed,{})[0]
        self.assertEqual((condition,observed,old),original)
        for key,value in old.items():self.assertEqual(current[key],value)
        self.assertFalse(current['comparison_evidence']['input_delivery_inferred'])
        self.assertFalse(current['comparison_evidence']['independent_business_result_verified'])
        return current

    def test_guessed_process_is_still_false_with_positive_literal_process_and_alias_evidence(self):
        obs=seed.observation();obs['foreground'].update(process='Org.gnome.Nautilus',title='Downloads')
        obs['capabilities']['app_processes']={'explorer':['Org.gnome.Nautilus']}
        result=self.check({'kind':'foreground_app','text':'nautilus'},obs)
        self.assertFalse(result['passed'])
        detail=result['comparison_evidence']
        self.assertEqual(detail['foreground']['process'],'Org.gnome.Nautilus')
        self.assertEqual(detail['observed_process_capability_aliases'],['explorer'])

    def test_advertised_alias_and_observed_process_still_verify_exactly(self):
        obs=seed.observation();obs['foreground']['process']='Org.gnome.Nautilus'
        obs['capabilities']['app_processes']={'explorer':['Org.gnome.Nautilus']}
        for text in ('explorer','Org.gnome.Nautilus'):
            self.assertTrue(self.check({'kind':'foreground_app','text':text},obs)['passed'])

    def test_foreign_and_malformed_aliases_do_not_enter_positive_alias_feedback(self):
        obs=seed.observation();obs['capabilities']['app_processes']={'calc':['not-calc'],'guess':'calc','empty':[]}
        result=self.check({'kind':'foreground_app','text':'calc'},obs)
        self.assertEqual(result['comparison_evidence']['observed_process_capability_aliases'],[])
        # The original predicate gives the declared alias precedence over an
        # identical process literal. Feedback must not silently change that.
        self.assertFalse(result['passed'])

    def test_window_title_still_uses_original_literal_predicate_and_only_reports_title(self):
        obs=seed.observation();result=self.check({'kind':'foreground_title','text':'unseen'},obs)
        self.assertFalse(result['passed'])
        self.assertEqual(result['comparison_evidence']['foreground']['title'],'Book')

    def test_missing_control_on_partial_read_is_not_converted_to_absence(self):
        obs=seed.observation();result=self.check({'kind':'control_absent','text':'Low Disk Space'},obs)
        self.assertFalse(result['passed'])
        self.assertEqual(result['comparison_evidence']['reason'],'complete_read_not_established')
        obs['accessibility']['truncated']=False
        self.assertTrue(self.check({'kind':'control_absent','text':'Low Disk Space'},obs)['passed'])

    def test_positive_present_control_blocks_absence_even_on_incomplete_tree(self):
        result=self.check({'kind':'control_absent','text':'H1'},seed.observation())
        self.assertFalse(result['passed'])
        self.assertEqual(result['comparison_evidence']['reason'],'matching_control_is_observed')

    def test_visible_save_text_is_not_marked_as_business_or_persistence_proof(self):
        result=self.check({'kind':'control_text','text':'H1'},seed.observation())
        self.assertTrue(result['passed'])
        self.assertEqual(result['comparison_evidence']['scope'],'positive_visible_control_name_only')

    def test_changed_field_name_retains_failed_exact_condition_and_native_reference_hint(self):
        obs=seed.identify_fields(seed.observation('F1','F1'),'task')
        result=self.check({'kind':'control_value','text':'H1','value':'F1','control_kind':'text'},obs)
        self.assertFalse(result['passed'])
        ref=obs['accessibility']['controls'][0]['field_ref']
        self.assertTrue(self.check({'kind':'field_value','field_ref':ref,'value':'F1'},obs)['passed'])

    def test_ambiguous_exact_labels_keep_existing_original_hint_and_failed_result(self):
        obs=seed.observation();obs['accessibility']['controls'].append(copy.deepcopy(obs['accessibility']['controls'][0]))
        result=self.check({'kind':'control_value','text':'H1','value':'H1'},obs)
        self.assertFalse(result['passed'])
        self.assertEqual(result['matching_controls'],2)

    def test_transition_is_never_advertised_as_focus_or_save_evidence(self):
        result=self.check({'kind':'screen_changed'},seed.observation())
        self.assertEqual(result['comparison_evidence']['scope'],'visual_transition_only_no_focus_or_save_proof')

    def test_unavailable_tree_does_not_prove_text_or_absence(self):
        obs=seed.observation();obs['accessibility']['status']='unavailable'
        for kind in ('control_text','control_absent'):
            self.assertFalse(self.check({'kind':kind,'text':'unseen'},obs)['passed'])

    def test_native_titles_and_alias_output_are_bounded_with_explicit_truncation(self):
        obs=seed.observation();obs['foreground']['title']='X'*10000
        obs['capabilities']['app_processes']={f'alias{i}':['calc'] for i in range(50)}
        result=self.check({'kind':'foreground_app','text':'unseen'},obs)['comparison_evidence']
        self.assertEqual(len(result['foreground']['title']),512)
        self.assertEqual(result['foreground']['truncated_fields'],['title'])
        self.assertEqual(len(result['observed_process_capability_aliases']),16)


if __name__=='__main__':unittest.main()
