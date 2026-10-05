"""Admit sustained capacity, reject transient positive GPU samples and retain history."""
import importlib.util
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import stable_resource_handoff as adapter
import stable_resource_startup as startup

spec = importlib.util.spec_from_file_location('stable_retained_resource_tests',
                                            Path(__file__).with_name('test_resource_handoff_successors.py'))
retained = importlib.util.module_from_spec(spec);spec.loader.exec_module(retained)
retained.a = adapter.NAMESPACE;retained.startup = startup


class PrefixTests(retained.PrefixTests):pass
class SupervisorTests(retained.SupervisorTests):pass
class LauncherTests(retained.LauncherTests):pass
class RetiredHandoffTests(retained.RetiredHandoffTests):pass


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.window = startup.HeadroomWindow(clock=lambda:self.now)

    def sample(self,at,free):
        self.now = at
        snapshot={'free_mib':free,'gpu_percent':99,'compute_pids':['user']}
        result=self.window.update(snapshot,56000)
        self.assertEqual(snapshot['free_mib'],free)
        self.assertEqual(result['actual_free_mib'],free)
        self.assertFalse(result['snapshot_values_changed'])
        return result

    def test_one_positive_sample_is_not_capacity(self):
        self.assertFalse(self.sample(0,56115)['ready'])

    def test_subthreshold_sample_resets_positive_window(self):
        self.sample(0,56115);self.sample(25,55711)
        self.assertEqual(self.sample(30,56115)['positive_seconds'],0)
        self.assertFalse(self.sample(55,56115)['ready'])

    def test_continuous_positive_thirty_seconds_is_admitted(self):
        self.sample(0,56115)
        self.assertFalse(self.sample(29,56115)['ready'])
        self.assertTrue(self.sample(30,56115)['ready'])

    def test_capacity_lost_after_admission_resets_gate(self):
        self.sample(0,56115);self.sample(30,56115)
        self.assertFalse(self.sample(31,55011)['ready'])

    def test_low_capacity_over_time_is_never_admitted(self):
        for t in range(0,70,5):self.assertFalse(self.sample(t,55711)['ready'])

    def test_clock_reversal_does_not_create_false_stability(self):
        self.sample(10,56115)
        with self.assertRaises(startup.image.SafetyHalt):self.sample(9,56115)

    def test_unknown_capacity_is_not_admitted(self):
        with self.assertRaises(startup.image.SafetyHalt):self.sample(0,None)


class RealSupervisorLoopTests(unittest.TestCase):
    def exercise(self, samples, budget):
        now=[0.];fake_time=SimpleNamespace(monotonic=lambda:now[0],sleep=lambda t:now.__setitem__(0,now[0]+t))
        cls=startup.guarded_controller(adapter.q.image.Controller)
        # Execute the compiled original supervisor with bounded synthetic GPU
        # observations. All OS/process/network/device boundaries are mocks.
        cls.fresh_engine.__globals__['time']=fake_time
        with tempfile.TemporaryDirectory() as temp:
            state={'parent_services':{'registered_engine':{'pid':123}},'engine_evidence_root':temp}
            controller=SimpleNamespace(args=SimpleNamespace(protocol=Path(temp)/'original/protocol.json',resource_wait_seconds=budget,min_free_mib=56000),
                                       state=state,p={'engine_off_identity':{'exact':True}},parent_idle=Mock(),checkpoint=Mock(),
                                       spawn=Mock(return_value=(SimpleNamespace(pid=456),{})),ready=Mock())
            values=iter(samples)
            def snapshot():return {'at':now[0],'free_mib':next(values)}
            with patch.object(startup,'time',fake_time),patch.object(startup.image.owned,'model_lock',side_effect=nullcontext), \
                    patch.object(startup.image.owned,'assert_no_services'),patch.object(startup.image.owned,'retire_registered',return_value={'retired':True}), \
                    patch.object(startup.image.owned,'gpu_snapshot',side_effect=snapshot), \
                    patch.object(startup.image.owned,'register_engine',return_value={'pid':456}):
                try:cls.fresh_engine(controller,{'id':'new-trial'})
                except startup.image.SafetyHalt:
                    controller.spawn.assert_not_called()
                    raise
            return controller,now[0]

    def test_volatile_sample_then_thirty_stable_seconds_causes_one_launch(self):
        controller,seconds=self.exercise([56115,55711,56115,56115,56115,56115,56115,56115,56115],60)
        self.assertEqual(seconds,40)
        controller.spawn.assert_called_once()
        self.assertEqual(controller.spawn.call_args.args[0][-2:],['--lock-timeout','60'])
        self.assertEqual(controller.ready.call_args.args[-1],600)

    def test_original_resource_deadline_refuses_without_launch(self):
        with self.assertRaisesRegex(startup.image.SafetyHalt,'headroom unavailable'):
            self.exercise([55711]*13,60)


if __name__ == '__main__':unittest.main()
