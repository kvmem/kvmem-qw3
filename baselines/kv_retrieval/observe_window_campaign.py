"""CPU-only progress/report observer; never signals or restarts experiment jobs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from compare_agent_windows import jsonl, write_json

ROOT=Path(__file__).resolve().parent


class ClientProgress:
    """Read append-only client events once, including incomplete streamed replies."""
    def __init__(self):
        self.offset=0
        self.pending=b''
        self.identity=None
        self.last_observed=None
        self.tool_ids=set()
        self.tool_errors=0
        self.stream_events=0
        self.thinking_chars=0
        self.text_chars=0
        self.message_start=None
        self.stream_open=False
        self.last_block_type=None
        self.tool_json_chars=0

    def update(self,path):
        if not path.exists():return
        info=path.stat()
        identity=(info.st_dev,info.st_ino)
        if self.identity is not None and (identity!=self.identity or info.st_size<self.offset):
            raise RuntimeError('client event file was replaced/truncated; inspection required')
        self.identity=identity
        with path.open('rb') as f:
            f.seek(self.offset)
            while chunk:=f.read(1024*1024):
                self.offset+=len(chunk)
                lines=(self.pending+chunk).split(b'\n')
                self.pending=lines.pop()
                for line in lines:
                    if not line:continue
                    row=json.loads(line)
                    self.last_observed=row['observed_at']
                    event=row.get('event',{})
                    streamed=event.get('event',{}) if event.get('type')=='stream_event' else {}
                    if streamed.get('type')=='message_start':
                        self.message_start=self.last_observed
                        self.stream_events=0;self.thinking_chars=0;self.text_chars=0
                        self.stream_open=True;self.last_block_type=None;self.tool_json_chars=0
                    if streamed:
                        self.stream_events+=1
                        if streamed.get('type')=='content_block_start':
                            self.last_block_type=streamed.get('content_block',{}).get('type')
                        delta=streamed.get('delta',{})
                        self.thinking_chars+=len(delta.get('thinking',''))
                        self.text_chars+=len(delta.get('text',''))
                        self.tool_json_chars+=len(delta.get('partial_json',''))
                        if streamed.get('type')=='message_stop':self.stream_open=False
                    content=event.get('message',{}).get('content',[])
                    for block in content if isinstance(content,list) else []:
                        if block.get('type')=='tool_use' and block.get('id'):self.tool_ids.add(block['id'])
                        if block.get('type')=='tool_result' and block.get('is_error'):self.tool_errors+=1


CLIENT_READERS={}


def completed_arm_paths(campaign, variants):
    """Count actual terminal arms, including a reserved arm finishing early."""
    completed=[]
    for variant in variants:
        for parent in campaign.glob('*/'+variant):
            attempts=sorted(p for p in parent.glob('attempt_*') if p.is_dir())
            if not attempts:
                continue
            path=attempts[-1]/'result.json'
            if path.exists() and json.loads(path.read_text()).get('status')=='completed':
                completed.append(str(attempts[-1]))
    return completed


def parallel_needs_inspection(row):
    if row.get('status')=='completed':
        return False
    return (row.get('status') in {'failed','agent_failed','verifier_failed',
             'resource_failed','infrastructure_failed','supervisor_failed'}
            or not row.get('supervisor_process_verified_live')
            or row.get('model_process_missing_during_agent',False))


def model_missing_during_agent(arm,status,model_live):
    # The harness writes agent_summary before judging/cleanup, but only updates
    # result.json after shutting down the model. That normal gap is not a fault.
    return (status=='running_agent' and not model_live
            and not (arm/'cell/agent_summary.json').exists())


def process_identity(pid):
    root=Path('/proc')/str(pid)
    try:
        raw=(root/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'state':raw[0],'starttime_ticks':int(raw[19]),
                'command':(root/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
    except (OSError,ValueError):return None


def snapshot(campaign,pid,starttime):
    state=json.loads((campaign/'state.json').read_text())
    process=process_identity(pid)
    alive=bool(process and process['state']!='Z' and process['starttime_ticks']==starttime)
    result={'observed_at':time.time(),'campaign_status':state['status'],'campaign_pid':pid,
            'campaign_process_verified_live':alive,
            'completed_trials':len(completed_arm_paths(campaign,state.get('selected_variants',['large','small']))),
            'serial_controller_completed_trials':len(state.get('completed',[])),
            'planned_trials':state.get('planned_trials',8),'task':state.get('task'),'variant':state.get('variant')}
    parallel=[]
    for path in (campaign/'parallel_arms').glob('*.json'):
        row=json.loads(path.read_text())
        owner=process_identity(row.get('supervisor_pid',0))
        row['supervisor_process_verified_live']=bool(owner and owner['state']!='Z'
            and owner['starttime_ticks']==row.get('supervisor_starttime_ticks')
            and 'run_parallel_window_arm.py' in owner['command'] and str(campaign) in owner['command'])
        arm=Path(row['arm'])
        model=process_identity(row.get('server_pid',0))
        row['server_process_verified_live']=bool(model and model['state']!='Z'
            and 'serve_aligned_infllm.py' in model['command'] and str(arm/'server') in model['command'])
        row['model_process_missing_during_agent']=model_missing_during_agent(
            arm,row.get('status'),row['server_process_verified_live'])
        parallel.append(row)
    result['parallel_arms']=parallel
    if state.get('arm'):
        arm=Path(state['arm'])
        if (arm/'result.json').exists():
            run=json.loads((arm/'result.json').read_text())
            result['arm_status']=run['status']
            server=process_identity(run.get('server_pid',0))
            result['server_process_verified_live']=bool(server and server['state']!='Z'
                and str(arm/'server') in server['command'] and 'serve_aligned_infllm.py' in server['command'])
            result['model_process_missing_during_agent']=model_missing_during_agent(
                arm,run['status'],result['server_process_verified_live'])
            requests=jsonl(arm/'server/requests.jsonl')
            complete=[r for r in requests if r.get('status')=='completed']
            result.update(completed_model_requests=len(complete),failed_model_requests=len(requests)-len(complete),
                completed_output_tokens=sum(r.get('output_tokens',0) for r in complete),
                max_prompt_tokens=max((r.get('prompt_tokens',0) for r in requests),default=0),
                unexpected_cold_prefills=sum(r.get('unexpected_cold_prefill',False) for r in complete))
            reader=CLIENT_READERS.setdefault(str(arm),ClientProgress())
            reader.update(arm/'client_events.jsonl')
            result.update(latest_client_event_age_s=time.time()-reader.last_observed if reader.last_observed else None,
                unique_tool_calls=len(reader.tool_ids),tool_result_errors=reader.tool_errors,
                current_response_stream_events=reader.stream_events,
                current_response_thinking_characters=reader.thinking_chars,
                current_response_text_characters=reader.text_chars,
                current_response_tool_json_characters=reader.tool_json_chars,
                current_response_stream_open=reader.stream_open,
                current_response_last_block_type=reader.last_block_type,
                current_response_since_stream_start_s=time.time()-reader.message_start if reader.message_start else None,
                streaming_counts_are_token_counts=False)
    result['requires_inspection']=((not alive and state['status'] not in ['completed','failed'])
        or state['status']=='failed' or result.get('model_process_missing_during_agent',False)
        or any(parallel_needs_inspection(row) for row in parallel))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--pid',type=int,required=True)
    a=p.parse_args();a.campaign=a.campaign.resolve()
    identity=process_identity(a.pid)
    if not identity or str(a.campaign) not in identity['command'] or 'run_window_campaign.py' not in identity['command']:
        raise ValueError('must attach to a currently verified campaign process')
    starttime=identity['starttime_ticks']
    while True:
        progress=snapshot(a.campaign,a.pid,starttime)
        write_json(a.campaign/'progress.json',progress)
        with (a.campaign/'observer.progress.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
        # Artifact generation occurs on CPU after reading complete lines, outside
        # either model's request path; frozen experiment scripts remain intact.
        completed=subprocess.run([sys.executable,str(ROOT/'compare_agent_windows.py'),
            '--campaign',str(a.campaign)],capture_output=True,text=True)
        if completed.returncode:
            write_json(a.campaign/'comparison_observer_error.json',{'at':time.time(),
                'error':completed.stderr,'exit_code':completed.returncode})
        status_path=ROOT/'status.json'
        status=json.loads(status_path.read_text())
        if status.get('current_run',{}).get('campaign_pid')==a.pid:
            status.update(updated_at=time.time(),agent_tasks_completed=progress['completed_trials'],
                          latest_campaign_observation=progress)
            status['current_run'].update(task=progress['task'],variant=progress['variant'])
            write_json(status_path,status)
        if progress['campaign_status'] in ['completed','failed'] or not progress['campaign_process_verified_live']:return
        time.sleep(60)


if __name__=='__main__':main()
