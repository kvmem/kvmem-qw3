import copy
import unittest
from jsonschema import Draft202012Validator
from condition_schema_wire import restore_inline_conditions
from qw3_computer.tool_schema import WORKFLOW_TOOLS


class InlineWireTests(unittest.TestCase):
    def setUp(self):
        self.declared = copy.deepcopy(WORKFLOW_TOOLS[0]['inputSchema']['properties']['conditions'])
        self.parameters = {'type': 'object', 'properties': {'step': {'type': 'object',
            'properties': {'conditions': {'type': 'array', 'items': {}}}}}}

    def repair(self, parameters):
        return restore_inline_conditions('mcp__computer__act', parameters, self.declared)

    def test_restores_all_six_variants_and_missing_bounds_without_mutation(self):
        before = copy.deepcopy(self.parameters)
        restored = self.repair(self.parameters)
        conditions = restored['properties']['step']['properties']['conditions']
        self.assertEqual(conditions, self.declared)
        self.assertEqual(self.parameters, before)
        self.assertEqual(len(conditions['items']['oneOf']), 6)
        self.assertEqual(self.repair(restored), restored)

    def test_rejects_missing_value_and_retains_exact_condition_semantics(self):
        condition = self.repair(self.parameters)['properties']['step']['properties']['conditions']
        validator = Draft202012Validator(condition)
        for value in [[], [{'kind': 'control_value', 'text': 'Name'}], [{'kind': 'screen_changed', 'text': 'bad'}]]:
            self.assertTrue(list(validator.iter_errors(value)))
        for value in [[{'kind': 'control_value', 'text': 'Name', 'value': 'Alice'}], [{'kind': 'screen_changed'}]]:
            self.assertFalse(list(validator.iter_errors(value)))

    def test_workflow_and_unverified_mode_are_left_as_received(self):
        different = {'type': 'object', 'properties': {'conditions': {'items': {'type': 'object'}}}}
        self.assertEqual(restore_inline_conditions('mcp__computer__workflow', different, {}), different)
        plain = {'type': 'object', 'properties': {}}
        self.assertEqual(self.repair(plain), plain)

    def test_refuses_unknown_nonempty_definition_or_different_bounds(self):
        for mutation in [{'items': {'type': 'object'}}, {'maxItems': 100}, {'items': None}]:
            p = copy.deepcopy(self.parameters);p['properties']['step']['properties']['conditions'].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):self.repair(p)

    def test_lost_step_fields_and_changed_trusted_source_are_not_guessed(self):
        for p in [{'properties': {'step': {}}}, {'properties': {'step': {'properties': {}}}}]:
            with self.subTest(parameters=p), self.assertRaises(ValueError):self.repair(p)
        changed = copy.deepcopy(self.declared);changed['maxItems'] = 100
        with self.assertRaises(ValueError):restore_inline_conditions('mcp__computer__act', self.parameters, changed)


if __name__ == '__main__':unittest.main()
