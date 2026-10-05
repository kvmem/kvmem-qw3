"""Historical admission boundaries and exact terminal/zero-trial recovery."""
from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import static_handoff_successors as a


class HistoricalAdmissionTests(unittest.TestCase):
    def test_nested_static_validator_uses_static_sdk_and_restores_effective_validator(self):
        static=Mock(return_value=('static','historical-memory'));effective=Mock(return_value=('static','actual-memory'))
        def image(path):return a.old.sdk.validate(path)
        with patch.object(a,'STATIC_SDK',static),patch.object(a,'STATIC_IMAGE',image),patch.object(a.old.sdk,'validate',effective):
            self.assertEqual(a.historical_image(Path('image')),('static','historical-memory'))
            self.assertEqual(a.old.sdk.validate(Path('sdk')),('static','actual-memory'))
        static.assert_called_once_with(Path('image'));effective.assert_called_once_with(Path('sdk'))

    def test_validator_failure_does_not_leave_a_replaced_global(self):
        effective=Mock();error=a.SafetyHalt('source changed')
        with patch.object(a,'STATIC_IMAGE',side_effect=error),patch.object(a.old.sdk,'validate',effective):
            with self.assertRaises(a.SafetyHalt):a.historical_image(Path('image'))
            self.assertIs(a.old.sdk.validate,effective)

    def test_actual_frozen_image_admission_rejects_mixed_profile_but_scoped_admission_passes(self):
        d=a.old.read(a.ORIGINAL)
        memory=a.old.quota.validate_protocol(Path(d['memory_protocol']))
        effective=lambda path:(a.STATIC_SDK(Path(path))[0],memory)
        with patch.object(a.old.sdk,'validate',effective):
            with self.assertRaisesRegex(a.SafetyHalt,'Original OSWorld source admission or budget changed'):
                a.STATIC_IMAGE(Path(d['image_protocol']))
            historical=a.historical_image(Path(d['image_protocol']))
            self.assertIs(a.old.sdk.validate,effective)
            self.assertNotEqual(historical['engine_off_identity'],memory['engine_identities']['off'])

    def test_recovered_namespace_does_not_replace_the_original_source_validator(self):
        self.assertIs(a.q.FROZEN_VALIDATORS['image'],a.historical_image)
        self.assertIs(a.old.FROZEN_VALIDATORS['image'],a.STATIC_IMAGE)


class TerminalScopeTests(unittest.TestCase):
    def setUp(self):
        self.documents={str(a.ORIGINAL):a.old.read(a.ORIGINAL),str(a.TERMINAL):a.old.read(a.TERMINAL),str(a.DIAGNOSTIC):a.old.read(a.DIAGNOSTIC)}
        self.evidence=self.documents[str(a.TERMINAL)]
        for row in self.evidence['stages'].values():
            for key in ['registration','queue']:
                path=row[key]['path'];self.documents[path]=a.old.read(path)
        self.sdk=self.documents[self.evidence['stages']['sdk']['queue']['path']]
        self.declaration=self.documents[str(a.ORIGINAL)]
        self.inventory=Mock(return_value=[])

    def execute(self,actual=False,retired=True):
        with ExitStack() as stack:
            for item in [patch.object(a.old.image,'pinned'),patch.object(a.old,'read',side_effect=lambda path:copy.deepcopy(self.documents[str(path)])),
                         patch.object(a.old,'frozen_protocol',return_value=({'trials':[]},{})),patch.object(a.old.image,'mixed_terminal_rows',self.inventory),
                         patch.object(a.old.base,'process_retired',return_value=retired)]:stack.enter_context(item)
            return a.terminal_scope(actual=actual)

    def test_exact_completed_sdk_and_five_zero_trial_halts_are_accepted(self):
        self.assertEqual(self.execute()['sdk'],self.sdk)
        self.inventory.assert_called_once()

    def test_nine_sdk_trials_are_not_a_completed_inventory(self):
        self.sdk['trials'].pop()
        with self.assertRaises(a.SafetyHalt):self.execute()

    def test_replaying_sdk_setup_is_not_admitted(self):
        self.sdk['status']='running'
        with self.assertRaises(a.SafetyHalt):self.execute()

    def test_changed_terminal_sdk_result_must_reach_original_inventory_rejection(self):
        self.inventory.side_effect=a.SafetyHalt('terminal result hash changed')
        with self.assertRaisesRegex(a.SafetyHalt,'terminal result hash'):self.execute()

    def test_missing_sdk_service_handoff_is_rejected(self):
        self.sdk['new_services_preserved']=False
        with self.assertRaises(a.SafetyHalt):self.execute()

    def test_unknown_terminal_sdk_state_is_rejected(self):
        self.sdk['termination_confirmed']=False
        with self.assertRaises(a.SafetyHalt):self.execute()

    def test_dispatched_successor_is_not_recovered_as_zero_trial(self):
        state=self.documents[self.evidence['stages']['image']['queue']['path']];state['trials']=[{'id':'already-dispatched'}]
        with self.assertRaises(a.SafetyHalt):self.execute()

    def test_waiting_successor_is_not_a_terminal_halt(self):
        self.documents[self.evidence['stages']['thinking']['queue']['path']]['status']='waiting'
        with self.assertRaises(a.SafetyHalt):self.execute()

    def test_changed_registered_generation_is_rejected(self):
        self.documents[self.evidence['stages']['image']['registration']['path']]['identity']['start_ticks']=-1
        with self.assertRaises(a.SafetyHalt):self.execute()

    def test_live_former_producer_is_not_restarted(self):
        with self.assertRaises(a.SafetyHalt):self.execute(actual=True,retired=False)

    def test_retired_former_producers_are_accepted(self):
        self.execute(actual=True,retired=True)

    def test_different_failure_cannot_use_this_recovery(self):
        self.documents[str(a.DIAGNOSTIC)]['exact_error']='unrelated failure'
        with self.assertRaises(a.SafetyHalt):self.execute()


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.declaration=a.old.read(a.ORIGINAL)
        temporary=tempfile.TemporaryDirectory(prefix='qw3-static-handoff-')
        self.addCleanup(temporary.cleanup)
        self.output=Path(temporary.name)/'recovery';self.cpu=Path(temporary.name)/'static-cpu.json'
        self.cpu.write_text('{}')

    def test_derivation_preserves_original_sdk_and_all_protocol_paths(self):
        before=copy.deepcopy(self.declaration);d=a.derive(before,self.output,self.cpu)
        self.assertEqual(before,self.declaration)
        for stage in ['sdk','image','thinking','feedback']:
            self.assertEqual(d[stage+'_protocol'],before[stage+'_protocol'])
        for key in ['sdk_registration','sdk_state_dir','memory_protocol','memory_queue','counts','controller_python','runner_python']:
            self.assertEqual(d[key],before[key])
        self.assertFalse(d['new_sdk_dispatch_allowed'])
        for stage in a.STAGES:self.assertEqual(Path(d[stage+'_state_dir']),self.output/(stage+'-queue'))

    def test_derived_scope_modification_is_rejected_before_original_validation(self):
        d=a.derive(self.declaration,self.output,self.cpu);d['counts']['image']=13
        with patch.object(a.old,'read',return_value=d),patch.object(a,'terminal_scope',return_value={'declaration':self.declaration}),patch.object(a.old,'validate') as original:
            with self.assertRaises(a.SafetyHalt):a.validate(self.output/'successors.json')
            original.assert_not_called()

    def test_retained_sdk_registration_routes_to_its_original_declaration(self):
        path=Path(self.declaration['sdk_registration'])
        with patch.object(a.old,'successor_registration',return_value={'stage':'sdk'}) as register:
            self.assertEqual(a.successor_registration(path,'sdk',self.output/'successors.json',self.declaration),{'stage':'sdk'})
            self.assertEqual(register.call_args.args[2],a.ORIGINAL)

    def test_substituted_sdk_registration_is_rejected(self):
        d=dict(self.declaration,sdk_registration='/tmp/foreign-sdk.json')
        with self.assertRaises(a.SafetyHalt):a.successor_registration(Path(d['sdk_registration']),'sdk',self.output/'successors.json',d)

    def test_retained_sdk_queue_uses_original_receipt_guards(self):
        registration=a.old.read(self.declaration['sdk_registration'])
        with patch.object(a.old,'successor_queue',return_value={'status':'completed'}) as queue:
            self.assertEqual(a.successor_queue(registration,self.output/'successors.json'),{'status':'completed'})
            self.assertEqual(queue.call_args.args[1],a.ORIGINAL)

    def test_changed_sdk_generation_cannot_reach_receipt_guard(self):
        registration=copy.deepcopy(a.old.read(self.declaration['sdk_registration']));registration['identity']['start_ticks']=-1
        with patch.object(a.old,'successor_queue') as queue:
            with self.assertRaises(a.SafetyHalt):a.successor_queue(registration,self.output/'successors.json')
            queue.assert_not_called()

    def test_new_producer_uses_original_registration_checks(self):
        with patch.object(a,'BASE_REGISTRATION',return_value={'stage':'image'}) as register:
            self.assertEqual(a.successor_registration(Path('image'),'image',Path('declaration'),self.declaration),{'stage':'image'})
            register.assert_called_once()

    def test_sdk_is_not_an_available_new_run_stage(self):
        self.assertNotIn('sdk',a.STAGES)


if __name__=='__main__':unittest.main()
