"""Resource fluctuations, exact prefix retention and scoped retired provenance."""
from contextlib import ExitStack
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import resource_handoff_successors as a
import resource_startup as startup
import run_engine as launcher


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.d = a.old.read(a.ORIGINAL)
        self.evidence = a.old.read(a.TERMINAL)
        self.documents = {str(a.ORIGINAL):self.d, str(a.TERMINAL):self.evidence,
                          str(a.DIAGNOSTIC):a.old.read(a.DIAGNOSTIC)}
        for stage, row in self.evidence['stages'].items():
            for path in (row['queue']['path'], str(a.REGISTRATIONS[stage])):
                self.documents[path] = a.old.read(path)
        self.image_state = self.documents[self.evidence['stages']['image']['queue']['path']]
        self.protocol = a.old.read(self.d['image_protocol'])
        reg = a.old.read(self.d['sdk_registration'])
        self.sdk = a.old.read(reg['queue'])

    def execute(self, actual=False, retired=True):
        real_read = a.old.read
        with ExitStack() as stack:
            stack.enter_context(patch.object(a, 'pinned'))
            stack.enter_context(patch.object(a.old, 'read', side_effect=lambda p:copy.deepcopy(self.documents[str(p)]) if str(p) in self.documents else real_read(p)))
            stack.enter_context(patch.object(a.previous, 'historical_image', return_value=self.protocol))
            stack.enter_context(patch.object(a.previous, 'terminal_scope', return_value={'sdk':self.sdk}))
            stack.enter_context(patch.object(a.old.base, 'process_retired', return_value=retired))
            stack.enter_context(patch.object(a.owned, 'assert_no_services'))
            stack.enter_context(patch.object(a.owned, 'assert_vm_retired'))
            return a.terminal_scope(actual=actual)

    def test_exact_single_prefix_is_retained_without_rewriting_result(self):
        self.assertEqual(self.execute()['prefix'],self.image_state['trials'])

    def test_empty_prefix_is_not_admitted_as_a_new_study(self):
        self.image_state['trials']=[]
        with self.assertRaises(a.SafetyHalt):self.execute()

    def test_two_completed_rows_cannot_use_one_row_recovery(self):
        self.image_state['trials']*=2
        with self.assertRaises(a.SafetyHalt):self.execute()

    def test_changed_trial_order_is_rejected(self):
        self.image_state['trials'][0]['id']='other'
        with self.assertRaises(a.SafetyHalt):self.execute()

    def test_changed_result_is_rejected(self):
        self.image_state['trials'][0]['result']['passed']=True
        with self.assertRaisesRegex(a.SafetyHalt,'Retained result'):self.execute()

    def test_missing_terminal_count_is_rejected(self):
        self.image_state['real_model_trials_completed']=0
        with self.assertRaises(a.SafetyHalt):self.execute()

    def test_live_old_producer_cannot_be_restarted(self):
        with self.assertRaisesRegex(a.SafetyHalt,'still live'):self.execute(actual=True,retired=False)

    def test_dispatched_downstream_work_cannot_be_overwritten(self):
        state=self.documents[self.evidence['stages']['feedback']['queue']['path']]
        state['trials']=[{'id':'already-started'}]
        with self.assertRaises(a.SafetyHalt):self.execute()

    def test_running_downstream_producer_is_not_terminal(self):
        state=self.documents[self.evidence['stages']['thinking']['queue']['path']]
        state['status']='running'
        with self.assertRaises(a.SafetyHalt):self.execute()

    def test_changed_registration_generation_routes_to_hash_guard(self):
        real=a.pinned
        with patch.object(a,'pinned',side_effect=lambda p,sha:(_ for _ in ()).throw(a.SafetyHalt('registration changed')) if p==a.REGISTRATIONS['image'] else real(p,sha)):
            with self.assertRaisesRegex(a.SafetyHalt,'registration changed'):a.terminal_scope()


class SupervisorTests(unittest.TestCase):
    def test_original_task_readiness_and_shared_gpu_guards_are_retained(self):
        source=inspect.getsource(startup.ORIGINAL_FRESH_ENGINE)
        fixed=startup.engine_source(source)
        for text in ('--min-free-mib','self.args.resource_wait_seconds','time.sleep(5)','owned.model_lock()',
                     'context_window") == 524288, 600','owned.register_engine(runtime, self.p["engine_off_identity"])'):
            self.assertIn(text,fixed)
        self.assertNotIn('"--lock-timeout", "5"]',fixed)

    def test_changed_supervisor_source_is_rejected(self):
        with self.assertRaises(a.SafetyHalt):startup.engine_source('def fresh_engine(): pass')

    def test_original_loop_budget_and_failure_path_are_retained(self):
        source=a.resume_source(inspect.getsource(a.ORIGINAL_IMAGE_RUN))
        self.assertIn('p["trials"][len(retained_prefix):]',source)
        self.assertIn('planned_trials=12',source)
        self.assertIn('except BaseException',source)
        self.assertIn('halted-no-further-dispatch',source)

    def test_changed_loop_source_is_rejected(self):
        with self.assertRaises(a.SafetyHalt):a.resume_source('def run(): pass')

    def test_only_remaining_trials_are_dispatched_and_prefix_copied(self):
        prefix=[{'id':'done','result':{'passed':False}}]
        with tempfile.TemporaryDirectory() as temp:
            state_path=Path(temp)/'new'; controller=Mock()
            def make(args,p,state):controller.state=state;return controller
            ns=dict(a.ORIGINAL_IMAGE_RUN.__globals__, retained_prefix=prefix, copy=copy,
                    validate=lambda p:{'trials':[{'id':'done'},{'id':'next'},{'id':'last'}]},
                    Controller=make,predecessors_ready=lambda p:{'verified':True})
            exec(compile(a.resume_source(inspect.getsource(a.ORIGINAL_IMAGE_RUN)),'prefix-qualified','exec'),ns)
            ns['run'](SimpleNamespace(protocol=Path(__file__),state_dir=state_path,wait_seconds=60))
            self.assertEqual([c.args[0]['id'] for c in controller.dispatch.call_args_list],['next','last'])
            self.assertEqual(controller.state['trials'],prefix)
            self.assertIsNot(controller.state['trials'],prefix)
            self.assertEqual(controller.state['real_model_trials_completed'],1)

    def test_remaining_trial_failure_is_preserved_without_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            controller=Mock();controller.dispatch.side_effect=a.SafetyHalt('unknown input')
            ns=dict(a.ORIGINAL_IMAGE_RUN.__globals__,retained_prefix=[{'id':'done'}],copy=copy,
                    validate=lambda p:{'trials':[{'id':'done'},{'id':'next'},{'id':'last'}]},
                    Controller=lambda *args:controller,predecessors_ready=lambda p:{'verified':True})
            exec(compile(a.resume_source(inspect.getsource(a.ORIGINAL_IMAGE_RUN)),'prefix-qualified','exec'),ns)
            with self.assertRaisesRegex(a.SafetyHalt,'unknown input'):
                ns['run'](SimpleNamespace(protocol=Path(__file__),state_dir=Path(temp)/'new',wait_seconds=60))
            controller.dispatch.assert_called_once_with({'id':'next'})
            self.assertEqual(controller.checkpoint.call_args.kwargs['status'],'halted-no-further-dispatch')


class LauncherTests(unittest.TestCase):
    """Execute the actual frozen launcher through admission, never loading a model."""
    class Admitted(Exception):pass

    def gate(self,wait,release_at):
        clock=[0.];samples=[];real_open=Path.open
        def query(command,**kwargs):
            if '--query-compute-apps=pid' in command:return '3906950\n'
            if '--query-gpu=memory.free' in command:
                samples.append(clock[0]);return '56115' if clock[0]>=release_at else '55711'
            if '--query-gpu=uuid,name,memory.total,memory.used' in command:raise self.Admitted()
            self.fail('Unexpected command in admission-only qualification')
        def opened(path,*args,**kwargs):
            if path==Path('/tmp/qw3-computer-gpu.lock'):
                return Mock(fileno=lambda:123)
            return real_open(path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(patch.object(sys,'argv',
                ['run_engine.py','--run-dir',str(Path(temp)/'run'),'--gpu-sharing','spare-capacity',
                 '--min-free-mib','56000','--lock-timeout',str(wait),'--context-tokens','524288','--kvmem','off']))
            stack.enter_context(patch.object(launcher,'time',SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda t:clock.__setitem__(0,clock[0]+t))))
            stack.enter_context(patch.object(launcher,'socket',SimpleNamespace(socket=Mock(return_value=MagicMock()))))
            stack.enter_context(patch.object(Path,'open',opened))
            stack.enter_context(patch.object(launcher.fcntl,'flock'))
            stack.enter_context(patch.object(launcher.os,'set_inheritable'))
            stack.enter_context(patch.object(launcher.subprocess,'check_output',side_effect=query))
            executed=stack.enter_context(patch.object(launcher.os,'execve'))
            signalled=stack.enter_context(patch.object(launcher.os,'kill'))
            try:launcher.main()
            finally:executed.assert_not_called();signalled.assert_not_called()
        return samples

    def test_original_five_seconds_reproduces_transient_admission_failure(self):
        with self.assertRaisesRegex(SystemExit,'Insufficient free GPU memory'):self.gate(5,6)

    def test_sixty_seconds_waits_in_same_launcher_until_capacity_returns(self):
        with self.assertRaises(self.Admitted):self.gate(60,6)

    def test_persistent_resource_shortage_still_has_bounded_refusal(self):
        with self.assertRaisesRegex(SystemExit,'Insufficient free GPU memory'):self.gate(60,120)


class RetiredHandoffTests(unittest.TestCase):
    def setUp(self):
        d=a.old.read(a.ORIGINAL);reg=a.old.read(d['sdk_registration'])
        self.sdk=a.old.read(reg['queue']);self.protocol=a.old.read(d['image_protocol'])
        self.protocol['sdk_queue']=reg['queue']
        self.scope={'sdk':self.sdk,'declaration':d,'prefix':[{'id':'004-full-retrievable-r1'}]}

    def execute(self,ready,retired=True):
        with patch.object(a,'terminal_scope',return_value=self.scope),patch.object(a.old.base,'process_retired',return_value=retired):
            return a.retired_sdk_ready(ready,self.protocol)

    def test_exact_retired_engine_receipt_is_marked_and_hook_restored(self):
        original=a.owned.register_engine
        result=self.execute(lambda p:{'registered_engine':a.owned.register_engine(Path(self.sdk['engine_runtime']),p['engine_off_identity'])})
        self.assertEqual(result['registered_engine'],self.sdk['registered_engine'])
        self.assertTrue(result['sdk_engine_retired_before_prefix_continuation'])
        self.assertIs(a.owned.register_engine,original)

    def test_live_sdk_engine_is_not_substituted(self):
        with self.assertRaisesRegex(a.SafetyHalt,'has not retired'):
            self.execute(lambda p:a.owned.register_engine(Path(self.sdk['engine_runtime']),p['engine_off_identity']),retired=False)

    def test_other_runtime_uses_real_registration(self):
        with patch.object(a.owned,'register_engine',return_value={'actual':True}) as real:
            result=self.execute(lambda p:a.owned.register_engine(Path('/tmp/another-runtime'),p['engine_off_identity']))
            self.assertEqual(result,{'actual':True,'sdk_engine_retired_before_prefix_continuation':True,'retained_image_prefix':['004-full-retrievable-r1']})
            real.assert_called_once()

    def test_other_profile_is_rejected_and_hook_restored(self):
        original=a.owned.register_engine
        with self.assertRaisesRegex(a.SafetyHalt,'another engine profile'):
            self.execute(lambda p:a.owned.register_engine(Path(self.sdk['engine_runtime']),{}))
        self.assertIs(a.owned.register_engine,original)

    def test_original_receipt_failure_remains_failure(self):
        original=a.owned.register_engine
        with self.assertRaisesRegex(a.SafetyHalt,'original ledger unknown'):
            self.execute(Mock(side_effect=a.SafetyHalt('original ledger unknown')))
        self.assertIs(a.owned.register_engine,original)


if __name__=='__main__':unittest.main()
