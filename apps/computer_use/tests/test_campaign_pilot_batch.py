"""A batch never trades reliable task retirement for apparent business success."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import campaign_pilot_batch as batch


class Child:
    pid=99
    def wait(self,timeout):return 0


class BatchSafetyTests(unittest.TestCase):
    def test_correct_artifact_without_stop_is_not_terminal(self):
        self.assertFalse(batch.child_terminal({'accepted':True,'artifact_passed':True}))
        self.assertFalse(batch.child_terminal({'owned_task_retired':True,'owned_task_stop_unconfirmed':True}))

    def exercise(self,root,results):
        calls=[]
        protocol={'app':'http://127.0.0.1:18205','app_runtime':str(root/'runtime'),
                  'trials':[{'id':'trial-one','case':'linux_browser','seconds':600},
                            {'id':'trial-two','case':'android_browser','seconds':600}]}
        def launch(command,**kwargs):
            destination=Path(command[command.index('--output')+1]);destination.mkdir()
            (destination/'integration-result.json').write_text(json.dumps(results[len(calls)]))
            calls.append(command);return Child()
        protocol_path=root/'protocol.json';protocol_path.write_text('{}')
        with patch.object(batch,'load_protocol',return_value=protocol),patch.object(batch,'admitted_app'),\
                patch.object(batch,'idle',return_value=True),patch.object(batch.subprocess,'Popen',side_effect=launch):
            result=batch.worker(protocol_path,root/'output')
        return calls,result

    def test_business_failure_with_acknowledged_retirement_can_continue(self):
        with tempfile.TemporaryDirectory() as name:
            calls,result=self.exercise(Path(name),[{'accepted':False,'owned_task_retired':True},
                                                  {'accepted':True,'owned_task_retired':True}])
            self.assertEqual(len(calls),2)
            self.assertTrue(result['termination_confirmed'])
            self.assertFalse(result['trials'][0]['result']['accepted'])
            self.assertIn('--retire-task',calls[0])

    def test_missing_stop_even_with_idle_does_not_start_the_next_trial(self):
        with tempfile.TemporaryDirectory() as name:
            calls,result=self.exercise(Path(name),[{'accepted':True}])
            self.assertEqual(len(calls),1)
            self.assertEqual(result['status'],'needs-reconciliation')
            self.assertFalse(result['termination_confirmed'])

    def test_unknown_input_is_never_accepted_for_admission(self):
        protocol={'app':'http://127.0.0.1:18205','model_output_tokens':16384,'tool_call_limit':100}
        state={'status':'idle','uncertain_actions':[{'action_id':'one'}],
               'model_output_tokens':16384,'thinking':True,'tool_call_limit':100}
        with patch.object(batch,'request',return_value=state):
            with self.assertRaises(RuntimeError):batch.admitted_app(protocol)

    def test_changed_model_budget_is_rejected_before_input(self):
        protocol={'app':'http://127.0.0.1:18205','model_output_tokens':16384,'tool_call_limit':100}
        with patch.object(batch,'request',return_value={'status':'idle','model_output_tokens':4096,
                         'thinking':True,'tool_call_limit':100}):
            with self.assertRaises(RuntimeError):batch.admitted_app(protocol)

    def test_completed_campaign_uses_standalone_admission_and_total_batch_budget(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);campaign=root/'formal';campaign.mkdir()
            (campaign/'plan.json').write_text(json.dumps({'status':'finished'}))
            args=SimpleNamespace(protocol=root/'protocol.json',output=root/'batch',campaign=campaign)
            protocol={'app':'http://127.0.0.1:18208','app_runtime':str(root/'runtime'),
                      'trials':[{'seconds':600},{'seconds':900}]}
            with patch.object(batch,'load_protocol',return_value=protocol),\
                 patch('completed_campaign_pilot.run_completed',return_value={'status':'pilot-finished'}) as standalone,\
                 patch.object(batch.interlude,'run') as original:
                self.assertEqual(batch.run(args)['status'],'pilot-finished')
                self.assertEqual(standalone.call_args.kwargs['watchdog_seconds'],2100)
                self.assertIn('campaign_pilot_batch',standalone.call_args.args[1])
                original.assert_not_called()


if __name__=='__main__':unittest.main()
