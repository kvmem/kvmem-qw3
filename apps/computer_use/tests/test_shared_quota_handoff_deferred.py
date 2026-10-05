"""Full retained deferred inventories and bounded/query pre-spawn admission."""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

import shared_quota_handoff_deferred as adapter

spec = importlib.util.spec_from_file_location("retained_handoff_deferred_tests", Path(__file__).with_name("test_shared_quota_port_deferred.py"))
retained = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retained)
retained.adapter = adapter
retained.bounded_tests.queue = adapter.BOUNDED
retained.query_tests.queue = adapter.QUERY


class BoundedQueueTests(retained.BoundedQueueTests):
    pass


class QueryQueueTests(retained.QueryQueueTests):
    pass


class FullHandoffTests(retained.FullHandoffTests):
    pass


class RoutingTests(retained.RoutingTests):
    pass


class SpawnTests(unittest.TestCase):
    def test_original_and_refresh_engine_wait_before_one_spawn(self):
        for command in (["python", str(adapter.ROOT / "apps/computer_use/run_engine.py")],
                        ["python", "-m", "query_refresh_engine"]):
            events=[]
            obj=types.SimpleNamespace(checkpoint=Mock())
            spawn=Mock(side_effect=lambda *args:events.append("spawn"))
            with patch.object(adapter.successors.previous.port,"wait_for_engine_port",side_effect=lambda:events.append("wait") or {}):
                adapter.admitted_spawn(obj,command,"engine",None,spawn)
            self.assertEqual(events,["wait","spawn"])
            spawn.assert_called_once_with(command,"engine",None)

    def test_busy_engine_port_prevents_child(self):
        obj=types.SimpleNamespace(checkpoint=Mock());spawn=Mock()
        with patch.object(adapter.successors.previous.port,"wait_for_engine_port",side_effect=adapter.SafetyHalt("busy")):
            with self.assertRaises(adapter.SafetyHalt):
                adapter.admitted_spawn(obj,["python","-m","query_refresh_engine"],"engine",None,spawn)
        spawn.assert_not_called()

    def test_other_children_do_not_probe_engine_port(self):
        obj=types.SimpleNamespace(checkpoint=Mock());spawn=Mock()
        with patch.object(adapter.successors.previous.port,"wait_for_engine_port") as wait:
            adapter.admitted_spawn(obj,["python","-m","task"],"task","framework",spawn)
        wait.assert_not_called()
        spawn.assert_called_once_with(["python","-m","task"],"task","framework")

    def test_real_inherited_spawn_is_called_once_and_only_after_admission(self):
        calls=[]
        class Base:
            def checkpoint(self,**values):calls.append("checkpoint")
            def spawn(self,*values):calls.append("spawn");return values
        candidate=adapter.controller(Base)()
        with patch.object(adapter.successors.previous.port,"wait_for_engine_port",side_effect=lambda:calls.append("port") or {}):
            result=candidate.spawn(["python","-m","query_refresh_engine"],"engine")
        self.assertEqual(calls,["port","checkpoint","spawn"])
        self.assertEqual(result,(["python","-m","query_refresh_engine"],"engine",None))


if __name__ == "__main__":
    unittest.main()
