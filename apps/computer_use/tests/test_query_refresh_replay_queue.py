"""Real queue/receipt checks; static admission, historical40 and host probes are fixtures."""
import argparse
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import query_refresh_replay_queue as queue


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = argparse.Namespace(protocol=self.root / 'protocol.json', state_dir=self.root / 'state',
                                       wait_seconds=60, resource_wait_seconds=60, min_free_mib=56000)
        queue.save(self.args.protocol, {})
        self.protocol = dict(trials=queue.trials())

    def loop(self, predecessor, send):
        with patch.object(queue, 'validate', return_value=self.protocol), \
                patch.object(queue, 'predecessor_ready', side_effect=predecessor), \
                patch.object(queue.Controller, 'dispatch', side_effect=send), \
                patch.object(queue.image.Controller, 'fresh_engine') as restore:
            return queue.run(self.args), restore

    def test_four_trial_loop_preserves_order_and_restores_original_off(self):
        seen = []
        with patch.object(queue, 'validate', return_value=self.protocol), patch.object(queue, 'predecessor_ready', return_value={}), \
                patch.object(queue.image.Controller, 'fresh_engine') as restore:
            def dispatch(controller, row):
                seen.append(copy.deepcopy(row)); controller.state['trials'].append(dict(row, mechanism_qualified=True))
            with patch.object(queue.Controller, 'dispatch', dispatch): result = queue.run(self.args)
        self.assertEqual(seen, queue.trials())
        self.assertTrue(result['termination_confirmed'])
        self.assertEqual(result['final_engine_policy'], 'off')
        self.assertEqual(restore.call_count, 1)
        self.assertEqual(restore.call_args.args[1], dict(id='restore-original-off'))

    def test_missing_predecessor_stops_without_launch_or_restore(self):
        with patch.object(queue, 'validate', return_value=self.protocol), \
                patch.object(queue, 'predecessor_ready', side_effect=queue.SafetyHalt('Missing actual parent')), \
                patch.object(queue.Controller, 'dispatch') as send, patch.object(queue.image.Controller, 'fresh_engine') as restore:
            with self.assertRaises(queue.SafetyHalt): queue.run(self.args)
            send.assert_not_called(); restore.assert_not_called()
        self.assertEqual(queue.read(self.args.state_dir / 'queue.json')['status'], 'halted-no-further-dispatch')

    def test_second_failure_stops_later_replays_and_preserves_first(self):
        seen = []
        def send(controller, row):
            seen.append(row['id'])
            if len(seen) == 2: raise queue.SafetyHalt('Request retirement unconfirmed')
            controller.state['trials'].append(dict(row, mechanism_qualified=False, result_sha256='preserved-fixture'))
        with patch.object(queue, 'validate', return_value=self.protocol), patch.object(queue, 'predecessor_ready', return_value={}), \
                patch.object(queue.Controller, 'dispatch', send), self.assertRaises(queue.SafetyHalt):
            queue.run(self.args)
        self.assertEqual(seen, [r['id'] for r in queue.trials()[:2]])
        state = queue.read(self.args.state_dir / 'queue.json')
        self.assertTrue(state['live_evidence_preserved'])
        self.assertEqual(state['trials'][0]['id'], queue.trials()[0]['id'])
        self.assertEqual(state['trials'][0]['result_sha256'], 'preserved-fixture')

    def test_existing_attempt_is_rejected_before_engine_change(self):
        row = queue.trials()[0]
        (self.root / 'attempts' / row['id']).mkdir(parents=True)
        controller = queue.Controller(self.args, self.protocol, dict(protocol_sha256=queue.owned.digest(self.args.protocol)))
        with patch.object(queue, 'validate', return_value=self.protocol), patch.object(controller, 'fresh_refresh_engine') as launch:
            with self.assertRaises(queue.SafetyHalt): controller.dispatch(row)
            launch.assert_not_called()

    def test_unfinished_actual_handle_is_verified_and_waited(self):
        reg = dict(identity={'pid': 100}, queue='/fixture')
        with patch.object(queue, 'registration', return_value=(reg, dict(status='waiting'))), \
                patch.object(queue.base, 'require_identity', return_value=reg['identity']) as verify:
            self.assertIsNone(queue.predecessor_ready(dict(parent_registration='fixture', parent_controller=reg['identity'], parent_queue=reg['queue'])))
        verify.assert_called_once_with(reg['identity'])

    def test_completed_but_live_producer_cannot_handoff(self):
        reg = dict(identity={'pid': 100}, queue='/fixture')
        with patch.object(queue, 'registration', return_value=(reg, dict(status='completed'))), \
                patch.object(queue.base, 'process_retired', return_value=False), patch.object(queue.bounded, 'validate') as admit:
            self.assertIsNone(queue.predecessor_ready(dict(parent_registration='fixture', parent_controller=reg['identity'], parent_queue=reg['queue'])))
            admit.assert_not_called()

    def test_scoped_full_handoff_source_is_immutable(self):
        with patch.object(queue.inspect, 'getsource', return_value='def terminal_predecessor(): pass'), self.assertRaises(queue.SafetyHalt):
            queue.full_prior_handoff({}, {})


class FullHandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        actual = queue.read(queue.RUNTIME / 'shared-quota-successors-20261005/successors.json')
        self.d = dict(memory_continuation='fixture-memory', memory_protocol=str(self.root / 'memory/protocol.json'),
                      original_followup=str(self.root / 'followup.json'))
        queue.save(Path(self.d['original_followup']), {})
        self.regs, self.protocols, self.states = {}, {}, {}
        self.engine = self.identity(999); self.feedback_engine = self.identity(998)
        self.client, self.gateway = self.identity(997), self.identity(996)
        self.registry = self.root / 'registry.json'; queue.save(self.registry, {})
        self.services = dict(model_client=self.client, registry=str(self.registry), registry_sha256=queue.owned.digest(self.registry),
                             new_gateways=[dict(process_identity=self.gateway, command=['fixture', '--runtime', str(self.root / 'gateway')])])
        for number, stage in enumerate(queue.q.STAGES, 200):
            frozen = queue.q.frozen_protocol(stage, Path(actual[stage + '_protocol']))
            specs = frozen[0]['trials'] if stage == 'sdk' else frozen['trials']
            path = self.root / stage / 'protocol.json'; path.parent.mkdir(); queue.save(path, {'trials': specs})
            self.d[stage + '_registration'] = stage
            self.d[stage + '_protocol'] = str(path)
            self.regs[stage] = dict(identity=self.identity(number), protocol=str(path))
            self.protocols[stage] = dict(trials=specs)
            rows = []
            for spec in specs:
                out = path.parent / 'attempts' / spec['id']; out.mkdir(parents=True)
                result = dict(finished_at=1, uncertain_actions=0, owned_task_retired=True,
                              administrative_recovery=True, unassisted_recovery_claimed=False, accepted=False)
                if stage == 'thinking': result.update(termination_confirmed=True, tool_calls_executed=0, business_success_claimed=False)
                name = 'integration-result.json' if spec.get('kind') in ('normal-pilot', 'receipt-loss') else 'result.json'
                queue.save(out / name, result)
                rows.append(dict(id=spec['id'], result=result, result_sha256=queue.owned.digest(out / name)))
            self.states[stage] = dict(status='completed', trials=rows)
        self.states['feedback'].update(termination_confirmed=True, parent_services_preserved=True,
                                      final_engine_policy='off', parent_services=self.services,
                                      engine_runtime='old-feedback', registered_engine=self.feedback_engine)
        self.parent_queue = self.root / 'feedback-queue.json'; queue.save(self.parent_queue, self.states['feedback'])
        memory = queue.q.quota.validate_protocol(Path(actual['memory_protocol']))
        self.kp = dict(trials=memory['trials'])
        mp = Path(self.d['memory_protocol']); mp.parent.mkdir(); queue.save(mp, self.kp)
        rows = []
        for spec in self.kp['trials']:
            out = mp.parent / 'attempts' / spec['id']; out.mkdir(parents=True)
            result = dict(finished_at=1, uncertain_actions=0)
            queue.save(out / 'result.json', result)
            rows.append(dict(id=spec['id'], result=result, result_sha256=queue.owned.digest(out / 'result.json')))
        self.memory = dict(status='completed', protocol_sha256=queue.owned.digest(mp), trials=rows)
        self.parent = dict(parent_declaration='fixture', parent_registration='feedback', parent_queue=str(self.parent_queue),
                           parent_app='fixture-app', engine_off_identity={'profile': 'fixture-off'})
        self.last = dict(engine_runtime='last-bounded', registered_engine=self.engine)

    @staticmethod
    def identity(pid):
        return dict(pid=pid, uid=1001, start_ticks=pid * 10, command=['fixture', str(pid)], cwd=str(queue.ROOT), state='S')

    def invoke(self, retired=lambda identity: True, registered=None):
        q = queue.q
        with ExitStack() as stack:
            patches = [(q, 'validate', lambda path: (self.d, dict(continuation_controller_identity=self.identity(150)), dict(identity=self.identity(151)))),
                       (q, 'successor_registration', lambda path, stage, *args: self.regs[stage]),
                       (q, 'successor_queue', lambda reg, path: next(self.states[s] for s in q.STAGES if reg == self.regs[s])),
                       (q, 'frozen_protocol', lambda stage, path: (self.protocols[stage], {}) if stage == 'sdk' else self.protocols[stage]),
                       (q, 'memory_queue', lambda *args: self.memory), (q.quota, 'validate_protocol', lambda path: self.kp),
                       (q.old, 'continued_parent', lambda *args: dict(status='completed')),
                       (q.old, 'terminal_parent', lambda *args: [dict(id='explicit-historical40-fixture')]),
                       (queue.base, 'process_retired', retired), (queue.base, 'require_identity', lambda identity: identity),
                       (queue.base, 'ledger_terminal', lambda runtime: {}), (queue.owned, 'assert_no_services', lambda path: None),
                       (queue.owned, 'assert_vm_retired', lambda path: None), (queue.owned, 'assert_apps_idle', lambda app: None),
                       (queue.owned, 'register_engine', lambda runtime, expected: registered or self.engine)]
            for owner, name, value in patches: stack.enter_context(patch.object(owner, name, side_effect=value))
            return queue.full_prior_handoff(self.parent, self.last)

    def test_success_checks_all_actual_frozen_stage_inventories_and_moves_only_final_engine(self):
        result = self.invoke()
        self.assertEqual(result['registered_engine'], self.engine)
        self.assertEqual(len(result['bounded_predecessor_receipts']), 10 + 12 + 4 + 8 + 8 + 1)

    def test_prior_business_result_tamper_is_rejected(self):
        stage = 'image'; row = self.states[stage]['trials'][0]
        path = Path(self.regs[stage]['protocol']).parent / 'attempts' / row['id'] / 'result.json'
        queue.save(path, dict(row['result'], score=1))
        with self.assertRaises(queue.SafetyHalt): self.invoke()

    def test_missing_or_reordered_prior_inventory_is_rejected(self):
        self.states['sdk']['trials'].reverse()
        with self.assertRaises(queue.SafetyHalt): self.invoke()

    def test_thinking_replay_cannot_execute_actions(self):
        row = self.states['thinking']['trials'][0]; row['result']['tool_calls_executed'] = 1
        path = Path(self.regs['thinking']['protocol']).parent / 'attempts' / row['id'] / 'result.json'
        queue.save(path, row['result']); row['result_sha256'] = queue.owned.digest(path)
        with self.assertRaises(queue.SafetyHalt): self.invoke()

    def test_old_feedback_engine_must_be_retired(self):
        with self.assertRaises(queue.SafetyHalt): self.invoke(retired=lambda identity: identity != self.feedback_engine)

    def test_latest_bounded_engine_identity_must_match(self):
        with self.assertRaises(queue.SafetyHalt): self.invoke(registered=self.identity(1000))


if __name__ == '__main__':
    unittest.main()
