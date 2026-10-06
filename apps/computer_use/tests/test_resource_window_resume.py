"""The new route must not relax terminal checks or change any task budget."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import resource_window_resume_scope as scope
import resource_window_deferred as deferred
import resource_window_native as native
import resource_window_condition as condition


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('registration.json', 'protocol.json', 'declaration.json'):
            (self.root / name).write_text('{}')
        self.parent = dict(registration_path=str(self.root/'registration.json'),
                           protocol=str(self.root/'protocol.json'), queue='new-queue', identity={'pid':123})
        self.old = copy.deepcopy(deferred.read(deferred.SOURCE_PROTOCOLS['bounded']))

    def routed(self):
        return deferred.routed_protocol('bounded', self.old, self.parent, self.root/'declaration.json')

    def test_actual_route_can_change_while_all_tasks_and_deadlines_are_retained(self):
        result = self.routed()
        deferred.route_proof('bounded', self.old, result)
        self.assertEqual(result['trials'], self.old['trials'])
        self.assertEqual(result['engine_off_identity'], self.old['engine_off_identity'])
        self.assertNotEqual(result['parent_controller'], self.old['parent_controller'])

    def test_task_budget_changes_cannot_hide_in_a_new_parent_route(self):
        result = self.routed()
        result['trials'][0]['seconds'] += 1
        with self.assertRaises(scope.SafetyHalt):
            deferred.route_proof('bounded', self.old, result)

    def test_another_unregistered_parent_cannot_hide_in_the_file_inventory(self):
        result = self.routed()
        result['files'].pop()
        with self.assertRaises(scope.SafetyHalt):
            deferred.route_proof('bounded', self.old, result)

    def test_missing_or_duplicate_source_anchor_never_compiles_a_new_controller(self):
        for source in ('different', 'anchor anchor'):
            with self.assertRaises(scope.SafetyHalt):
                scope.replace_once(source, 'anchor', 'new')

    def test_live_original_producer_blocks_fresh_routing_admission(self):
        with patch.object(scope.stable.old.base, 'process_retired', return_value=False):
            with self.assertRaisesRegex(scope.SafetyHalt, 'still live'):
                scope.check_audit(actual=True)

    def test_real_completed_prefix_and_original_budgets_remain_identical(self):
        audit = scope.check_audit()
        self.assertEqual(len(audit['retained_prefix']), 1)
        self.assertEqual(native.BOUNDS, native.original.BOUNDS)
        self.assertEqual(condition.BOUNDS, condition.original.BOUNDS)
        self.assertEqual(deferred.FOUR_DECLARATION, scope.DECLARATION)
        self.assertEqual(condition.NAMESPACE.ORIGINAL_NATIVE_PARENT,
                         scope.ref(scope.ORIGINAL_NATIVE_DECLARATION))
        self.assertNotEqual(deferred.original.FOUR_DECLARATION, deferred.FOUR_DECLARATION)
