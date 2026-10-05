"""Retain frozen full inventories and strict deferred routing boundaries."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock,patch

import static_handoff_deferred as adapter

spec=importlib.util.spec_from_file_location('static_handoff_retained_port_deferred',Path(__file__).with_name('test_shared_quota_port_deferred.py'))
retained=importlib.util.module_from_spec(spec);spec.loader.exec_module(retained)
retained.adapter=adapter
retained.bounded_tests.queue=adapter.BOUNDED
retained.query_tests.queue=adapter.QUERY


class BoundedQueueTests(retained.BoundedQueueTests):pass
class QueryQueueTests(retained.QueryQueueTests):pass
class FullHandoffTests(retained.FullHandoffTests):pass
class RoutingTests(retained.RoutingTests):pass


class RecoveredParentTests(unittest.TestCase):
    def test_actual_static_handoff_declaration_is_the_only_bounded_parent(self):
        self.assertEqual(adapter.FOUR_DECLARATION,adapter.RUNTIME/'static-handoff-successors-20261005-01/successors.json')
        self.assertIs(adapter.q,adapter.successors.q)

    def test_original_task_and_request_source_bytes_are_retained(self):
        self.assertEqual(adapter.owned.digest(adapter.original_bounded.__file__),adapter.SOURCE_SHA['bounded'])
        self.assertEqual(adapter.owned.digest(adapter.original_query.__file__),adapter.SOURCE_SHA['query'])

    def test_engine_port_wait_precedes_exact_original_spawn(self):
        controller=Mock();spawn=Mock(return_value='same-child')
        command=[str(adapter.q.CONTROLLER_PYTHON),str(adapter.ROOT/'apps/computer_use/run_engine.py'),'--run-dir','new-runtime']
        with patch.object(adapter.admission.successors.previous.port,'wait_for_engine_port',return_value={'port':18080}) as wait:
            self.assertEqual(adapter.admission.admitted_spawn(controller,command,'engine',None,spawn),'same-child')
        wait.assert_called_once();spawn.assert_called_once_with(command,'engine',None)


if __name__=='__main__':unittest.main()
