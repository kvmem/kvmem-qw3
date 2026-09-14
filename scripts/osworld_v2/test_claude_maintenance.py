"""Check serial phase attribution and maintenance within an action interval."""
import contextlib
import csv
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest

import analyze_claude_pair as analysis
import summarize_claude_suite as suite


def phase_record(rid, prefill=0, middecode=0):
    return (
        f'[qw3-native-accounting] post_semantic_ms={prefill} post_query_replay_ms=0\n'
        f'native generate: prompt_tokens={1000+rid} kvmem_reuse=900 prefilled={100+rid} '
        f'kvmem_middecode_refreshes={middecode} kvmem_middecode_query_ms={100*middecode} '
        f'kvmem_middecode_generated_query_ms={90*middecode} '
        f'kvmem_middecode_reselect_ms={20*middecode}\n'
        f'[qw3-server-ttft] rid={rid} engine_ttft_ms=1000\n'
    )


class MaintenanceTest(unittest.TestCase):
    def test_summary_tool_use_stop_does_not_reuse_previous_compaction(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);arm=root/'full/qw3';arm.mkdir(parents=True)
            steps=[{'step':i,'started_ns':(9+10*i)*10**9,
                    'observation_ready_ns':(10+10*i)*10**9,
                    'decision_wait_ms':9000,'action':{'action':'screenshot'}} for i in range(3)]
            apis=[]
            for rid,start,step,stop in [(1,11,0,'end_turn'),(2,15,0,'tool_use'),
                                      (3,21,1,'tool_use'),(4,25,1,'tool_use')]:
                apis.append({'id':rid,'path':'/v1/messages','after_step':step,
                    'started_ns':start*10**9,'first_content_ns':(start+1)*10**9,
                    'finished_ns':(start+2)*10**9,'stop_reason':stop,
                    'input_usage':{'input_tokens':1000+rid},'output_usage':{'output_tokens':20},
                    'ttft_ms':1000,'total_ms':2000,'status':200})
                event={'type':'message_start','message':{'id':f'msg_{rid}'}}
                (arm/f'response_{rid:04d}.sse.gz').write_bytes(
                    gzip.compress(('data: '+json.dumps(event)+'\n\n').encode()))
            events=[]
            for rid in (2,4):
                events.extend([{'type':'system','subtype':'compact_boundary',
                                'compact_metadata':{'duration_ms':2000,'trigger':'auto'}},
                               {'type':'assistant','message':{'id':f'msg_{rid}'}}])
            for name,rows in [('steps.jsonl',steps),('api.jsonl',apis),('claude.stream.jsonl',events)]:
                (arm/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (arm/'result.json').write_text(json.dumps(
                {'status':'complete','task':'028','arm':'qw3','steps':2,'score':0}))
            with contextlib.redirect_stdout(io.StringIO()):analysis.analyze(root)
            metric=json.loads((root/'analysis/metrics.json').read_text())['qw3']
            self.assertEqual(metric['native_compactions'],2)
            self.assertEqual([c['summary_api']['id'] for c in metric['compaction_details']],[1,3])
            self.assertEqual([c['next_action_api_id'] for c in metric['compaction_details']],[2,4])
            self.assertEqual(metric['step_ttft_ms']['mean'],6000)
            self.assertEqual(metric['maintenance_step_ttft_ms']['n'],2)

    def test_middecode_fields_stay_with_their_request(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'server.log'
            path.write_text(phase_record(1,middecode=2)+phase_record(2,prefill=4))
            records=analysis.server_records(path)
        self.assertEqual(records[1]['kvmem_middecode_refreshes'],2)
        self.assertEqual(records[1]['kvmem_middecode_query_ms'],200)
        self.assertEqual(records[2]['kvmem_middecode_refreshes'],0)
        both={'server':{'native_post_semantic_ms':4,'kvmem_middecode_refreshes':2}}
        self.assertTrue(analysis.selection_request(both))
        self.assertTrue(analysis.prefill_selection(both))
        self.assertEqual(analysis.middecode_refreshes(both),2)

    def test_middecode_and_earlier_non_action_response_are_maintenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);arm=root/'full/kvmem';arm.mkdir(parents=True)
            steps=[{'step':i,'started_ns':(9+10*i)*10**9,
                    'observation_ready_ns':(10+10*i)*10**9,
                    'decision_wait_ms':9000,'action':{'action':'left_click'}} for i in range(5)]
            # Each decision interval is [10,19), [20,29), [30,39), [40,49).
            # The third interval has a refresh in a non-action response before
            # the final tool call. Neither kind belongs in regular-step means.
            apis=[]
            for rid,start,step,stop in [(1,11,0,'tool_use'),(2,21,1,'tool_use'),
                                      (3,31,2,'end_turn'),(4,35,2,'tool_use'),(5,41,3,'tool_use')]:
                apis.append({'id':rid,'path':'/v1/messages','after_step':step,
                    'started_ns':start*10**9,'first_content_ns':(start+1)*10**9,
                    'finished_ns':(start+2)*10**9,'stop_reason':stop,
                    'input_usage':{'input_tokens':1000+rid},'output_usage':{'output_tokens':20},
                    'ttft_ms':1000,'total_ms':2000,'status':200})
            for name,rows in [('steps.jsonl',steps),('api.jsonl',apis),('claude.stream.jsonl',[])]:
                (arm/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result={'status':'complete','task':'052','arm':'kvmem','steps':4,'score':0}
            (arm/'result.json').write_text(json.dumps(result))
            (arm/'qw3_server.log').write_text(phase_record(1,middecode=2)+phase_record(2,prefill=4)+
                phase_record(3,middecode=1)+phase_record(4)+phase_record(5))
            with contextlib.redirect_stdout(io.StringIO()):analysis.analyze(root)
            metric=json.loads((root/'analysis/metrics.json').read_text())['kvmem']
            with (root/'analysis/step_latency.csv').open() as source:
                rows=list(csv.DictReader(source))
            self.assertEqual(metric['kvmem_selections'],3)
            self.assertEqual(metric['kvmem_prefill_selection_requests'],1)
            self.assertEqual(metric['kvmem_middecode_selection_requests'],2)
            self.assertEqual(metric['kvmem_middecode_refreshes'],3)
            self.assertEqual(metric['kvmem_selection_events'],4)
            self.assertEqual(metric['maintenance_action_wait_ms']['n'],3)
            self.assertEqual(metric['regular_step_ttft_ms']['n'],1)
            self.assertEqual(metric['regular_step_ttft_ms']['mean'],2000)
            self.assertEqual(rows[0]['kvmem_middecode_selection_before_action'],'True')
            self.assertEqual(float(rows[0]['kvmem_middecode_maintenance_ms']),240)
            self.assertEqual(rows[2]['api_id'],'4')
            self.assertEqual(rows[2]['kvmem_selection_api_ids'],'[3]')
            self.assertEqual(rows[2]['kvmem_middecode_selection_before_action'],'True')

            # Cohort aggregation must retain refresh events and preserve the
            # distinction between first-content latency and action wait.
            pair=root/'trials/052/analysis';pair.mkdir(parents=True)
            other={**metric,'result':{**result,'arm':'qw3'}}
            (pair/'metrics.json').write_text(json.dumps({'kvmem':metric,'qw3':other}))
            (root/'plan.json').write_text(json.dumps({'tasks':['052']}))
            combined,_=suite.summarize(root)
            self.assertEqual(combined['arms']['kvmem']['kvmem_middecode_refreshes'],3)
            self.assertEqual(combined['arms']['kvmem']['latency']['maintenance_action_wait_ms']['observations'],3)


if __name__=='__main__':unittest.main()
