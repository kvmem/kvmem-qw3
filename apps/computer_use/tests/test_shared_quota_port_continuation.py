import argparse
import copy
import errno
import inspect
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import shared_quota_port_continuation as q


class PortContinuationTests(unittest.TestCase):
    def fixture(self):
        ids = q.PREFIX_IDS + [q.FAILED_ID, 'quota1-080-kvmem-off-r2', 'quota1-080-kvmem-on-r2']
        modes = ['off', 'on', 'on', 'off', 'on', 'off', 'off', 'on']
        protocol = dict(trials=[dict(id=name, kvmem=mode) for name, mode in zip(ids, modes)])
        rows = [dict(id=spec['id'], kvmem=spec['kvmem'], exit_code=0, result_sha256='retained',
                     result=dict(finished_at=1, uncertain_actions=0, score=None if '035' in spec['id'] else 0,
                                 accepted=False)) for spec in protocol['trials'][:5]]
        state = dict(status='halted-no-further-dispatch', error=q.ERROR, current_trial=q.FAILED_ID, trials=rows)
        return [state, protocol, q.PORT_ERROR, []]

    def test_five_matched_results_retained_without_mutation(self):
        data = self.fixture(); retained = copy.deepcopy(data)
        prefix = q.failure_scope(*data)
        self.assertEqual(data, retained)
        self.assertEqual(prefix, retained[0]['trials'])
        prefix[0]['result']['accepted'] = True
        self.assertFalse(data[0]['trials'][0]['result']['accepted'])

    def test_live_waiting_process_is_not_a_failure(self):
        data = self.fixture(); data[0]['status'] = 'starting-fresh-engine'
        with self.assertRaises(q.SafetyHalt): q.failure_scope(*data)

    def test_other_engine_failure_not_admitted(self):
        data = self.fixture(); data[2] = 'CUDA error'
        with self.assertRaises(q.SafetyHalt): q.failure_scope(*data)

    def test_started_task_or_runtime_is_not_replayed(self):
        for path in ('attempts/' + q.FAILED_ID, 'engines/' + q.FAILED_ID):
            data = self.fixture(); data[3] = [path]
            with self.subTest(path=path), self.assertRaises(q.SafetyHalt): q.failure_scope(*data)

    def test_unknown_input_is_preserved_as_rejection(self):
        data = self.fixture(); data[0]['trials'][2]['result']['uncertain_actions'] = 1
        with self.assertRaises(q.SafetyHalt): q.failure_scope(*data)

    def test_boolean_zero_is_not_zero_unknowns(self):
        data = self.fixture(); data[0]['trials'][2]['result']['uncertain_actions'] = False
        with self.assertRaises(q.SafetyHalt): q.failure_scope(*data)

    def test_unconfirmed_task_stop_rejects_admission(self):
        data = self.fixture(); data[0]['trials'][2]['result']['owned_task_stop_unconfirmed'] = True
        with self.assertRaises(q.SafetyHalt): q.failure_scope(*data)

    def test_missing_or_reordered_prefix_rejected(self):
        for operation in ('missing', 'reverse'):
            data = self.fixture()
            if operation == 'missing': data[0]['trials'].pop()
            else: data[0]['trials'].reverse()
            with self.subTest(operation=operation), self.assertRaises(q.SafetyHalt): q.failure_scope(*data)

    def test_changed_arm_or_runner_exit_rejected(self):
        for field, value in (('kvmem', 'wrong'), ('exit_code', 1)):
            data = self.fixture(); data[0]['trials'][0][field] = value
            with self.subTest(field=field), self.assertRaises(q.SafetyHalt): q.failure_scope(*data)

    def test_other_failed_trial_not_admitted(self):
        data = self.fixture(); data[0]['current_trial'] = 'quota1-080-kvmem-on-r2'
        with self.assertRaises(q.SafetyHalt): q.failure_scope(*data)

    def test_busy_port_waits_same_probe_scope_until_free(self):
        with patch.object(q.socket, 'socket') as socket:
            bind = socket.return_value.__enter__.return_value.bind
            bind.side_effect = [OSError(errno.EADDRINUSE, 'busy'), None]
            with patch.object(q.time, 'sleep') as sleep:
                result = q.wait_for_engine_port()
            self.assertEqual(result['probes'], 2); sleep.assert_called_once_with(.2)
            self.assertEqual(bind.call_args.args[0], ('127.0.0.1', 18080))
            socket.return_value.__enter__.return_value.setsockopt.assert_not_called()

    def test_permanently_busy_port_stops_without_new_launcher(self):
        with patch.object(q.socket, 'socket') as socket, patch.object(q.time, 'monotonic', side_effect=[1, 122]):
            socket.return_value.__enter__.return_value.bind.side_effect = OSError(errno.EADDRINUSE, 'busy')
            with self.assertRaises(q.SafetyHalt): q.wait_for_engine_port()

    def test_permission_error_not_retried(self):
        with patch.object(q.socket, 'socket') as socket, patch.object(q.time, 'sleep') as sleep:
            socket.return_value.__enter__.return_value.bind.side_effect = OSError(errno.EACCES, 'denied')
            with self.assertRaises(q.SafetyHalt): q.wait_for_engine_port()
            sleep.assert_not_called()

    def test_original_port_and_deadline_cannot_be_changed(self):
        with self.assertRaises(q.SafetyHalt): q.wait_for_engine_port(18081)
        with self.assertRaises(q.SafetyHalt): q.wait_for_engine_port(seconds=3600)

    def test_changed_frozen_loop_anchor_rejected(self):
        source = inspect.getsource(q.owned.run).replace('for row in p["trials"]:', 'changed_loop:', 1)
        with self.assertRaises(q.SafetyHalt): q.patch_loop(source)

    def test_omitting_retired_successors_rejects_before_any_host_probe(self):
        declaration = dict(kind='exact_shared_quota_pre_task_port_continuation', python=str(q.PYTHON),
                           completed_prefix_replayed=False, engine_profiles_changed=False, old_failures_overwritten=False,
                           planned_trials=8, remaining_model_trials=3, port_wait_seconds=120, retired_successors=[])
        with patch.object(q, 'read', return_value=declaration), patch.object(q.owned, 'process_identity') as process:
            with self.assertRaises(q.SafetyHalt): q.validate(Path('fixture-only'))
            process.assert_not_called()

    def test_full_original_loop_executes_only_three_remaining_trials(self):
        self.execute_loop(False)

    def test_port_wait_failure_prevents_first_launcher_and_preserves_prefix(self):
        self.execute_loop(True)

    def execute_loop(self, fail_port):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); protocol_path = root / 'protocol.json'; parent_path = root / 'parent.json'
            parent_path.write_text(json.dumps(dict(engine_runtime='original')))
            data = self.fixture(); specs = data[1]['trials']
            for row in specs: row['task'] = '035' if '035' in row['id'] else '080'
            protocol = data[1] | dict(parent_candidate_protocol=str(parent_path), engine_identities={'on':{}, 'off':{}},
                                     pilot_app='http://127.0.0.1:18213', framework='pinned', formal_campaign='pinned',
                                     engine_endpoint='http://127.0.0.1:18080')
            protocol_path.write_text(json.dumps(protocol)); launches=[]; saved=[]
            namespace = dict(vars(q.owned))
            def popen(command, **kwargs):
                launches.append(command)
                if '--output' in command:
                    directory = Path(command[command.index('--output') + 1]);directory.mkdir(parents=True)
                    (directory/'result.json').write_text(json.dumps(dict(finished_at=1, uncertain_actions=0, accepted=False)))
                return types.SimpleNamespace(pid=100 + len(launches), poll=lambda:None, wait=lambda timeout:0)
            class Lock:
                def __enter__(self): return self
                def __exit__(self, *values): return False
            declaration = dict(profile='unchanged', original_followup=str(parent_path))
            namespace.update(old=types.SimpleNamespace(continued_parent=lambda *values:dict(status='completed',pid=1,trials=[])),
                owned=q.owned, canonical={}, declaration=declaration, declaration_sha='decl', wrapper_sha='wrapper', PYTHON=q.PYTHON,
                completed_prefix=data[0]['trials'], copy=copy, validate_protocol=lambda path:protocol,
                digest=lambda path:'original', save=lambda path,state:saved.append(copy.deepcopy(state)),
                terminal_parent=lambda *values:[], process_identity=lambda pid:None,
                assert_apps_idle=lambda *values:None, assert_no_services=lambda *values:None, assert_vm_retired=lambda *values:None,
                model_lock=Lock, gpu_snapshot=lambda:dict(free_mib=60000), retire_registered=lambda item:dict(retired=True),
                register_engine=lambda *values:dict(pid=100 + len(launches)), register_child=lambda pid,cmd:dict(pid=pid),
                request=lambda *values,**kwargs:dict(data=[dict(context_window=524288)]),
                subprocess=types.SimpleNamespace(Popen=popen, STDOUT=-2), wait_for_engine_port=lambda:dict(passed=True))
            if fail_port:
                def stop(): raise q.SafetyHalt('Engine port remains busy')
                namespace['wait_for_engine_port'] = stop
            exec(compile(q.patch_loop(inspect.getsource(q.owned.run)), 'cpu-full-frozen-port-loop', 'exec'), namespace)
            args=argparse.Namespace(protocol=protocol_path,state_dir=root/'state',wait_seconds=60,resource_wait_seconds=60,min_free_mib=56000)
            if fail_port:
                with self.assertRaises(q.SafetyHalt): namespace['run'](args)
                self.assertEqual(launches,[]);self.assertEqual(saved[-1]['trials'],data[0]['trials'])
            else:
                state=namespace['run'](args)
                self.assertEqual(state['status'],'completed');self.assertEqual(len(state['trials']),8)
                self.assertEqual(state['trials'][:5],data[0]['trials'])
                children=[command for command in launches if '--output' in command]
                self.assertEqual([Path(command[command.index('--output')+1]).name for command in children],[row['id'] for row in specs[5:]])
                for command in launches:
                    if '--run-dir' in command:
                        self.assertIn('shared_quota_engine.py', ' '.join(command))
                        self.assertEqual(command[command.index('--context-tokens')+1],'524288')
                        self.assertEqual(command[command.index('--min-free-mib')+1],'56000')
                self.assertEqual(len(children),3)


if __name__ == '__main__': unittest.main()
