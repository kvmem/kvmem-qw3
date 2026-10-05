"""Retain full terminal inventories and saved query inputs across recovery."""
import importlib.util
import inspect
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import resource_handoff_deferred as adapter

spec = importlib.util.spec_from_file_location('resource_retained_port_deferred',
                                            Path(__file__).with_name('test_shared_quota_port_deferred.py'))
retained = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retained)
retained.adapter = adapter
retained.bounded_tests.queue = adapter.BOUNDED
retained.query_tests.queue = adapter.QUERY


class BoundedQueueTests(retained.BoundedQueueTests):pass
class QueryQueueTests(retained.QueryQueueTests):pass
class FullHandoffTests(retained.FullHandoffTests):pass
class RoutingTests(retained.RoutingTests):pass


class RecoveryTests(unittest.TestCase):
    def test_actual_resource_parent_is_required(self):
        self.assertEqual(adapter.FOUR_DECLARATION,
                         adapter.RUNTIME / 'resource-handoff-successors-20261005-01/successors.json')
        self.assertIs(adapter.q, adapter.successors.q)

    def test_query_memory_supervisor_keeps_mode_specific_profile_and_readiness(self):
        source = adapter.query_startup.source(adapter.original_query.Controller)
        for text in ('row["mode"]', '"--kvmem", "on"', '"--lock-timeout", "60"]',
                     'read(self.p["engine_profile"])["engine_identities"][row["mode"]]',
                     'context_window") == 524288, 600'):
            self.assertIn(text, source)
        self.assertIn('Path(self.state["engine_evidence_root"])', source)

    def test_saved_request_dispatch_and_final_off_restoration_are_original(self):
        self.assertEqual(inspect.getsource(adapter.QUERY.Controller.dispatch),
                         inspect.getsource(adapter.original_query.Controller.dispatch))
        self.assertIsNot(adapter.QUERY.Controller.fresh_engine, adapter.original_query.Controller.fresh_engine)

    def test_port_admission_remains_before_launch(self):
        controller = Mock(); spawn = Mock(return_value='same-child')
        command = [str(adapter.q.CONTROLLER_PYTHON), str(adapter.ROOT / 'apps/computer_use/run_engine.py'),
                   '--run-dir', 'new-runtime']
        with patch.object(adapter.previous.admission.successors.previous.port,
                          'wait_for_engine_port', return_value={'port':18080}) as wait:
            self.assertEqual(adapter.previous.admission.admitted_spawn(controller, command, 'engine', None, spawn), 'same-child')
        wait.assert_called_once(); spawn.assert_called_once_with(command, 'engine', None)


if __name__ == '__main__':unittest.main()
