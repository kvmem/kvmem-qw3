import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from observe_window_campaign import ClientProgress, completed_arm_paths, parallel_needs_inspection, snapshot


class StreamingChecks(unittest.TestCase):
    def test_live_supervisor_does_not_hide_missing_model_or_misreport_cleanup(self):
        for parallel in [False,True]:
            with self.subTest(parallel=parallel),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);arm=root/'first/small/attempt_01'
                arm.mkdir(parents=True)
                (arm/'result.json').write_text(json.dumps({'status':'running_agent','server_pid':3}))
                (root/'state.json').write_text(json.dumps({'status':'running_arm','selected_variants':['small'],
                    'completed':[],'task':'first','variant':'small','arm':str(arm)}))
                model={'state':'S','command':f'serve_aligned_infllm.py --out {arm}/server'}
                identities={1:{'state':'S','starttime_ticks':10},3:model}
                victim=3
                if parallel:
                    arm=root/'fourth/small/attempt_01';arm.mkdir(parents=True)
                    model={'state':'S','command':f'serve_aligned_infllm.py --out {arm}/server'}
                    identities.update({2:{'state':'S','starttime_ticks':20,
                        'command':f'run_parallel_window_arm.py --campaign {root}'},4:model})
                    victim=4
                    (root/'parallel_arms').mkdir()
                    (root/'parallel_arms/fourth.json').write_text(json.dumps({'status':'running_agent',
                        'arm':str(arm),'supervisor_pid':2,'supervisor_starttime_ticks':20,'server_pid':4,
                        'server_process_verified_live':True}))
                with patch('observe_window_campaign.process_identity',side_effect=identities.get):
                    self.assertFalse(snapshot(root,1,10)['requires_inspection'])
                    # Missing/reused/zombie model PIDs must override live supervisors
                    # and the parallel supervisor's stale "server live" observation.
                    for invalid in [None,{**model,'state':'Z'},{'state':'S','command':'unrelated_process'}]:
                        identities[victim]=invalid
                        self.assertTrue(snapshot(root,1,10)['requires_inspection'])
                    (arm/'cell').mkdir()
                    (arm/'cell/agent_summary.json').write_text('{}')
                    self.assertFalse(snapshot(root,1,10)['requires_inspection'])

    def test_parallel_failure_or_disappeared_supervisor_requires_inspection(self):
        self.assertFalse(parallel_needs_inspection({'status':'running_agent','supervisor_process_verified_live':True}))
        self.assertTrue(parallel_needs_inspection({'status':'resource_failed','supervisor_process_verified_live':True}))
        self.assertTrue(parallel_needs_inspection({'status':'running_agent','supervisor_process_verified_live':False}))
        self.assertFalse(parallel_needs_inspection({'status':'completed','supervisor_process_verified_live':False}))

    def test_completed_parallel_arm_is_counted_before_serial_controller_reaches_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def result(task,variant,attempt,status):
                path=root/task/variant/attempt/'result.json'
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps({'status':status}))
                return str(path.parent)
            result('first','small','attempt_01','running_agent')
            fourth=result('fourth','small','attempt_01','completed')
            (Path(fourth).parent/'attempt_01.controller.log').write_text('controller output\n')
            result('fourth','large','attempt_01','completed')
            self.assertEqual(completed_arm_paths(root,['small']),[fourth])
            result('fourth','small','attempt_02','failed')
            self.assertEqual(completed_arm_paths(root,['small']),[])

    def test_partial_utf8_append_and_deduplicated_tool_calls(self):
        def wire(event,at):return (json.dumps({'observed_at':at,'event':event},ensure_ascii=False)+'\n').encode()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events'
            reader=ClientProgress()
            start=wire({'type':'stream_event','event':{'type':'message_start'}},1)
            delta=wire({'type':'stream_event','event':{'type':'content_block_delta','delta':{'thinking':'中文'}}},2)
            path.write_bytes(start+delta[:-5]);reader.update(path)
            self.assertEqual(reader.stream_events,1)
            self.assertEqual(reader.thinking_chars,0)
            with path.open('ab') as f:f.write(delta[-5:])
            reader.update(path);reader.update(path)
            self.assertEqual(reader.stream_events,2)
            self.assertEqual(reader.thinking_chars,2)
            tool=wire({'message':{'content':[{'type':'tool_use','id':'one'}]}},3)
            with path.open('ab') as f:f.write(tool+tool)
            reader.update(path)
            self.assertEqual(reader.tool_ids,{'one'})
            with path.open('ab') as f:f.write(start)
            reader.update(path)
            self.assertEqual(reader.thinking_chars,0)
            self.assertEqual(reader.stream_events,1)

    def test_tool_json_and_closed_stream_are_not_mistaken_for_thinking(self):
        def wire(event):return json.dumps({'observed_at':1,'event':{'type':'stream_event','event':event}})+'\n'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events'
            path.write_text(''.join(map(wire,[
                {'type':'message_start'},
                {'type':'content_block_start','content_block':{'type':'tool_use'}},
                {'type':'content_block_delta','delta':{'type':'input_json_delta','partial_json':'{"path":'}},
                {'type':'message_stop'},
            ])))
            reader=ClientProgress();reader.update(path)
            self.assertEqual(reader.last_block_type,'tool_use')
            self.assertEqual(reader.tool_json_chars,8)
            self.assertFalse(reader.stream_open)
            with path.open('a') as f:f.write(wire({'type':'message_start'}))
            reader.update(path)
            self.assertTrue(reader.stream_open)
            self.assertIsNone(reader.last_block_type)
            self.assertEqual(reader.tool_json_chars,0)


if __name__=='__main__':unittest.main()
