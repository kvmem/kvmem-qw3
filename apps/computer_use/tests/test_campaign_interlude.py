"""Never admit a second workload based on the preceding formal result."""
import json
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from campaign_interlude import pilot_spec, pilot_watchdog, run, terminal_pilot


def arguments(root, **overrides):
    values = dict(campaign=root/'formal', output=root/'windows-python-new-01',
                  app_runtime=root/'ui', app='fixture-app', case='windows_python_new',
                  bundle=root/'bundle', engine=root/'engine', ffmpeg=root/'ffmpeg',
                  repair_fixture=None, acceptance_manifest=root/'acceptance.json', seconds=900)
    return SimpleNamespace(**(values | overrides))


class InterludeAdmissionTests(unittest.TestCase):
    def test_windows_requires_the_correct_repair_fixture_kind(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            with self.assertRaises(ValueError):
                pilot_spec(arguments(root, case='windows_python_repair'))
            with self.assertRaises(ValueError):
                pilot_spec(arguments(root, repair_fixture=root/'fixture.json'))
            command,result=pilot_spec(arguments(root))
            self.assertIn('--acceptance-manifest',command)
            self.assertEqual(result.parent.name,'windows-python-new-01-trial')
            self.assertEqual(pilot_watchdog(arguments(root)),2100)

    def test_windows_unsafe_or_overlong_identity_is_rejected_before_setup(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for identity in ('unsafe name','a'*40):
                with self.assertRaises(ValueError):
                    pilot_spec(arguments(root,output=root/identity))

    def test_business_failure_can_be_terminal_but_missing_stop_receipt_cannot(self):
        args=SimpleNamespace(case='windows_latex_new')
        ended={'status':'complete','passed':False,'guest_result':{'desktop_stopped':True}}
        self.assertTrue(terminal_pilot(args,ended))
        self.assertFalse(terminal_pilot(args,ended | {'guest_result':{}}))
        self.assertFalse(terminal_pilot(args,ended | {'guest_result':{'desktop_stopped':True,'needs_reconciliation':True}}))
        self.assertFalse(terminal_pilot(args,ended | {'status':'preparing'}))

    def test_paused_platform_with_unknown_delivery_cannot_resume_queue(self):
        args=SimpleNamespace(case='linux_browser',app='fixture-app')
        with patch('campaign_interlude.request',return_value={'status':'paused','uncertain_actions':['a1']}):
            self.assertFalse(terminal_pilot(args,{}))

    def orchestrate(self, root, outcome=None, watchdog=False):
        args=arguments(root)
        args.campaign.mkdir()
        formal_result=args.campaign/'043-legacy-r1/result.json'
        formal_result.parent.mkdir()
        formal_result.write_text(json.dumps({'status':'scored','score':0}))
        plan={'status':'running','engine_runtime':str(args.engine),
              'jobs':[{'status':'running','id':'043-legacy-r1','runner_pid':999998}]}
        (args.campaign/'plan.json').write_text(json.dumps(plan))
        command,result=pilot_spec(args)
        class Pilot:
            pid=999997
            def wait(self,timeout):
                if watchdog:raise subprocess.TimeoutExpired(command,timeout)
                result.parent.mkdir()
                result.write_text(json.dumps(outcome))
                return 0
        def owned_command(pid):
            if pid==999999:return ['owned-scheduler']
            raise ValueError('Unrelated process')
        with patch('campaign_interlude.verify_frozen'), \
             patch('osworld_windows.bundle.verify_bundle'), \
             patch('campaign_interlude.scheduler',return_value=(999999,['owned-scheduler'])), \
             patch('campaign_interlude.command',side_effect=owned_command), \
             patch('campaign_interlude.suspended',side_effect=[False,True,True,True,True]), \
             patch('campaign_interlude.request',return_value={'status':'idle','uncertain_actions':[]}), \
             patch('campaign_interlude.time.sleep'), \
             patch('campaign_interlude.os.kill') as signals, \
             patch('campaign_interlude.subprocess.Popen',return_value=Pilot()):
            receipt=run(args)
            return receipt,[call.args[1] for call in signals.call_args_list]

    def test_terminal_pilot_is_read_from_its_own_result_not_formal_result(self):
        with tempfile.TemporaryDirectory() as name:
            outcome={'status':'complete','passed':False,'guest_result':{'desktop_stopped':True}}
            receipt,signals=self.orchestrate(Path(name),outcome)
            self.assertEqual(receipt['pilot_result'],outcome)
            self.assertEqual(receipt['preceding_formal_result']['status'],'scored')
            self.assertEqual(signals,[signal.SIGSTOP,signal.SIGCONT])

    def test_watchdog_does_not_resume_formal_queue_or_kill_native_worker(self):
        with tempfile.TemporaryDirectory() as name:
            receipt,signals=self.orchestrate(Path(name),watchdog=True)
            self.assertEqual(receipt['status'],'needs-reconciliation')
            self.assertEqual(signals,[signal.SIGSTOP])


if __name__=='__main__':
    unittest.main()
