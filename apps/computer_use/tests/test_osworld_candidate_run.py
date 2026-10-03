"""Candidate hooks preserve original evaluation and refuse live campaigns."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import requests
import osworld_candidate_run as candidate


class CandidateRunnerTests(unittest.TestCase):
    def test_recorder_child_receives_helper_path_without_changing_agent_environment(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'vm.json').write_text(json.dumps({'container_id':'owned','name':'owned-vm'}))
            observed=[]
            def factory(*args,**kwargs):
                return SimpleNamespace(start=lambda:observed.append(os.environ['PYTHONPATH']))
            media=candidate.CandidateMedia(root,recorder_factory=factory)
            with patch.dict(os.environ,{'PYTHONPATH':'/private/frozen/core'}):
                media.start()
                self.assertEqual(os.environ['PYTHONPATH'],'/private/frozen/core')
            self.assertEqual(len(observed),1)
            self.assertTrue(observed[0].startswith('/private/frozen/core'+os.pathsep))
            self.assertIn('/apps/computer_use/.runtime/media-qualification',observed[0])
            self.assertFalse(media.errors)

    def test_only_owned_services_are_rewritten(self):
        gateway=['python','-m','qw3_computer.device_gateway','--port','18304','--runtime','private']
        converted=candidate.candidate_command(gateway,16384)
        self.assertEqual(gateway[2],'qw3_computer.device_gateway')
        self.assertEqual(converted[2],'device_gateway_entry')
        self.assertEqual(converted[converted.index('--port')+1],'18504')
        app=['python','-m','qw3_computer.app','--port','18305','--desktop','http://127.0.0.1:18304']
        converted=candidate.candidate_command(app,16384)
        self.assertEqual(converted[-2:],['--model-output-tokens','16384'])
        unrelated=['python','-m','task_loader','--port','1234']
        self.assertEqual(candidate.candidate_command(unrelated,16384),unrelated)

    def test_running_formal_campaign_is_rejected_before_reading_other_inputs(self):
        with tempfile.TemporaryDirectory() as name:
            directory=Path(name)
            (directory/'plan.json').write_text(json.dumps({'status':'running','jobs':[]}))
            with patch.object(candidate.runner,'request') as request:
                with self.assertRaisesRegex(ValueError,'all40'):
                    candidate.admit(SimpleNamespace(formal_campaign=directory))
                request.assert_not_called()

    def hooks_args(self,root):
        engine=root/'engine';engine.mkdir()
        (engine/'manifest.json').write_text(json.dumps({'kvmem':'off'}))
        framework=root/'framework';framework.mkdir()
        (framework/'device_gateway_entry.py').write_text('canonical')
        acceptance=root/'acceptance.json';acceptance.write_text('{}')
        return SimpleNamespace(output=root/'attempt',engine_runtime=engine,framework=framework,
                               acceptance_manifest=acceptance,model_output_tokens=16384)

    def test_hooks_leave_setup_and_grader_requests_untouched_and_restore_globals(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);args=self.hooks_args(root)
            original_ports=candidate.environment.PORTS
            with patch.object(requests,'post',return_value='official response') as original_post,\
                    patch.object(candidate.runner,'request',return_value={}) as original_request:
                media=SimpleNamespace(start=lambda:'start',finish=lambda:'end')
                with candidate.candidate_hooks(args,media):
                    self.assertEqual(requests.post('http://127.0.0.1:18500/start_recording'),'start')
                    self.assertEqual(requests.post('http://127.0.0.1:18500/end_recording'),'end')
                    self.assertEqual(requests.post('http://127.0.0.1:18500/execute',json={'original':'setup'}),'official response')
                    original_post.assert_called_once_with('http://127.0.0.1:18500/execute',json={'original':'setup'})
                    candidate.runner.request(candidate.ORIGINAL_APP,'/api/state')
                    original_request.assert_called_once_with(candidate.CANDIDATE_APP,'/api/state')
                self.assertIs(candidate.environment.PORTS,original_ports)

    def test_failed_media_does_not_discard_a_valid_independent_score(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);args=self.hooks_args(root)
            base={'status':'scored','score':1.0,'passed':True,'unassisted_passed':True,'finished_at':1}
            def underlying(_):
                args.output.mkdir()
                requests.post('http://127.0.0.1:18500/start_recording').raise_for_status()
                requests.post('http://127.0.0.1:18500/end_recording').raise_for_status()
                candidate.runner.save(args.output/'result.json',base)
                return dict(base)
            # No vm.json causes a real media-initialization failure. The
            # unchanged evaluator still completes and its receipt is retained.
            with patch.object(candidate,'admit'),patch.object(candidate.runner,'run',side_effect=underlying):
                result=candidate.run(args)
            self.assertEqual(result['score'],1.0)
            self.assertTrue(result['unassisted_business_passed'])
            self.assertFalse(result['accepted'])
            self.assertFalse(result['video_collection_passed'])
            self.assertEqual(json.loads((args.output/'runner-result.json').read_text()),base)
            self.assertEqual([row['stage'] for row in result['video_collection_errors']],['start','finish'])

    def test_complete_video_without_business_success_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);args=self.hooks_args(root)
            base={'status':'scored','score':0.0,'passed':False,'unassisted_passed':False,'finished_at':1}
            def underlying(_):
                args.output.mkdir();candidate.runner.save(args.output/'result.json',base)
                return dict(base)
            media=SimpleNamespace(complete=True,errors=[],close=lambda:None)
            with patch.object(candidate,'admit'),patch.object(candidate,'CandidateMedia',return_value=media),\
                    patch.object(candidate.runner,'run',side_effect=underlying):
                result=candidate.run(args)
            self.assertFalse(result['accepted'])


if __name__=='__main__':unittest.main()
