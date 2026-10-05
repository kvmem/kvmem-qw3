"""Contract restoration only: no model, desktop, grader or live bridge."""
import copy
import unittest

from jsonschema import Draft202012Validator

from condition_schema_repair import repair_condition_parameters as repair
from qw3_computer.tool_schema import WORKFLOW_TOOLS


class ConditionRepairTests(unittest.TestCase):
    def setUp(self):
        self.declared=copy.deepcopy(WORKFLOW_TOOLS[0]['inputSchema']['properties']['conditions'])
        self.opaque={'type':'object','properties':{'step':{'type':'object','properties':{'conditions':{'type':'array','items':{}}}}},
                     'required':['unrelated_original_field']}

    def test_inline_contract_is_restored_without_mutating_the_input(self):
        before=copy.deepcopy(self.opaque);actual,repaired=repair('mcp__computer__act',self.opaque,self.declared)
        self.assertTrue(repaired);self.assertEqual(self.opaque,before)
        self.assertEqual(actual['required'],before['required'])
        self.assertEqual(actual['properties']['step']['properties']['conditions'],self.declared)

    def test_missing_value_and_extra_screen_changed_fields_are_rejected_after_repair(self):
        actual,_=repair('mcp__computer__act',self.opaque,self.declared)
        validator=Draft202012Validator(actual['properties']['step']['properties']['conditions'])
        self.assertTrue(list(validator.iter_errors([{'kind':'control_value','text':'Name'}])))
        self.assertTrue(list(validator.iter_errors([{'kind':'screen_changed','text':'anything'}])))
        self.assertFalse(list(validator.iter_errors([{'kind':'control_value','text':'Name','value':'literal'}])))

    def test_all_original_positive_condition_variants_remain_representable(self):
        actual,_=repair('mcp__computer__act',self.opaque,self.declared)
        validator=Draft202012Validator(actual['properties']['step']['properties']['conditions'])
        for kind in ('control_text','control_absent','foreground_title','foreground_app'):
            with self.subTest(kind=kind):self.assertFalse(list(validator.iter_errors([{'kind':kind,'text':'literal'}])))
        self.assertFalse(list(validator.iter_errors([{'kind':'screen_changed'}])))

    def test_other_tools_or_absent_step_support_are_not_injected(self):
        for name,parameters in [('unrelated_tool',self.opaque),('mcp__computer__act',{'type':'object','properties':{}})]:
            with self.subTest(name=name):self.assertEqual(repair(name,parameters,self.declared),(parameters,False))

    def test_different_nonempty_schema_or_bounds_are_never_silently_overwritten(self):
        for mutation in ({'items':{'type':'object'}},{'minItems':2}):
            parameters=copy.deepcopy(self.opaque);parameters['properties']['step']['properties']['conditions'].update(mutation)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):repair('mcp__computer__act',parameters,self.declared)

    def test_workflow_and_already_declared_contracts_are_idempotent(self):
        parameters={'type':'object','properties':{'conditions':{'type':'array','items':{}}}}
        actual,changed=repair('mcp__computer__workflow',parameters,self.declared);self.assertTrue(changed)
        self.assertEqual(repair('mcp__computer__workflow',actual,self.declared),(actual,False))


if __name__=='__main__':unittest.main()
