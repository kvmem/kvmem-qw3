"""Independent actual VM/receipt/video audit of the separate condition study."""
import argparse
import json
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import time

import av

import condition_schema_model_study as study


def require(value,reason):
    if not value:raise study.adapter.owned.SafetyHalt(reason)


def retired(identity):
    current=study.adapter.owned.process_identity(identity['pid'])
    return current is None or current['state']=='Z' or current['start_ticks']!=identity['start_ticks']


def audit(study_path,stack_paths,cpu_path,output):
    output=Path(output).resolve()
    require(output.is_relative_to(study.candidate.RUNTIME),'Use a new goal-owned audit directory')
    protocol=json.loads(Path(study_path).read_text())
    study.validate_study(study_path,protocol['trials'][0]['id'])
    cpu=json.loads(Path(cpu_path).read_text())
    require(cpu['passed'] and cpu['tests_run']==5 and not any(cpu[k] for k in ['errors','failures','skipped','model_requests','device_inputs']),
        'Study qualification incomplete')
    for name,sha in cpu['source_sha256'].items():
        require(study.candidate.digest(study.candidate.ROOT/'apps/computer_use'/name)==sha,'Qualified study source changed')
    output.mkdir(parents=True,exist_ok=False,mode=0o700);shutil.copyfile(__file__,output/'audit-source.py')
    report=dict(kind='independent_separate_condition_schema_actual_vm_stack_audit',checked_at=time.time(),passed=False,
        study=study.ref(study_path),cpu_qualification=study.ref(cpu_path),audit_source=study.ref(output/'audit-source.py'),
        records=[],model_requests=0,device_inputs=0,engine_binding_qualified=False,business_improvement_proven=False)
    try:
        arms=[]
        for directory in map(lambda path:Path(path).resolve(),stack_paths):
            result_path=directory/'result.json';result=json.loads(result_path.read_text())
            stack_path=directory/'stack.json';stack=json.loads(stack_path.read_text())
            receipt_path=directory/'condition-stack-qualification.json';receipt=json.loads(receipt_path.read_text())
            row=next(r for r in protocol['trials'] if r['id']==result['trial_id']);arms.append(row['arm'])
            _,_,framework,_=study.validate_study(study_path,row['id'])
            require(result['passed'] and len(result['checks'])==13 and all(result['checks'].values()),'Actual original stack checks incomplete')
            require(receipt['passed'] and receipt['study']==study.ref(study_path) and receipt['original_result']==study.ref(result_path),
                'Separate wrapper receipt changed')
            require(stack['study']==study.ref(study_path) and stack['framework']==str(framework),'Actual private source selection changed')
            require(study.candidate.digest(stack_path)==result['stack_sha256'],'Stack receipt changed')
            require(stack['status']=='retired' and stack['vm_retired'] and stack['seed_metadata_unchanged'],'Stack retirement incomplete')
            require(result['engine_binding_qualified'] is False and receipt['actual_engine_bound'] is False,'Do not claim engine binding')
            require(result['model_tasks_started']==result['model_inference_requests']==result['engine_starts']==result['gui_inputs']==0,
                'Readonly qualification dispatched model/input')
            for source_map in [stack['source_sha256'],result['source_sha256']]:
                for name,sha in source_map.items():
                    require(study.candidate.digest(directory/'source'/name)==sha==study.candidate.digest(study.candidate.ROOT/'apps/computer_use'/name),
                        'Original archived stack dependency changed')
            for name,reference in receipt['new_helper_sources'].items():
                require(study.adapter.checked_ref(reference)==directory/'source'/name,'New wrapper archive changed')
                require(study.candidate.digest(study.candidate.ROOT/'apps/computer_use'/name)==reference['sha256'],'Current wrapper changed')
            processes=[result['producer'],result['recorder_identity']]+[v['identity'] for v in stack['services'].values()]+stack['client_descendants']
            require(all(retired(i) for i in processes),'Same actual process generation remains live')
            probe=subprocess.run(['docker','inspect','--type','container',stack['vm']['container_id']],capture_output=True,text=True,timeout=10)
            require(probe.returncode!=0 and any(s in probe.stderr for s in ['No such object','No such container']),
                'Actual owned VM still exists or read failed')
            counts=[]
            client=Path(stack['services']['client']['runtime']);gateway=Path(stack['services']['gateway']['runtime'])
            for path in [gateway/'receipts.sqlite',client/'devices/linux-vm/desktop.sqlite']:
                with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:counts.append(db.execute('SELECT COUNT(*) FROM actions').fetchone()[0])
            require(counts==[0,0],'Actual GUI input receipts found')
            model_files=list((client/'inference').glob('**/request.json'))+list((client/'inference').glob('**/request.archive.json'))
            require(model_files==[],'Actual model request files found')
            first=json.loads((directory/'initial-client-state.json').read_text());last=json.loads((directory/'refreshed-client-state.json').read_text())
            import native_control_model_stack as original
            require(original.idle_profile(first) and original.idle_profile(last),'Client readiness/pause differs')
            require(first['observation']['desktop_id']==last['observation']['desktop_id'] and first['observation']['observation_id']!=last['observation']['observation_id'],
                'Actual observation identities differ')
            require(study.candidate.digest(directory/'observed-screen.png')==last['observation']['sha256'],'Actual screenshot changed')
            manifest=json.loads((directory/'vm.json').read_text())
            require(manifest['image']==study.adapter.environment.IMAGE and manifest['disk_read_only'] is True,'VM source/seed changed')
            movie=Path(result['video']['path']);require(study.candidate.digest(movie)==result['video']['sha256'],'Actual movie changed')
            with av.open(str(movie)) as media:frames=sum(1 for _ in media.decode(video=0))
            require(frames>0 and frames==result['video']['decoded_frames'],'Independent video decode differs')
            video_receipt_path=directory/'recording/video/recording.json';video_receipt=json.loads(video_receipt_path.read_text())
            require(video_receipt['complete'] is True,'Original recording receipt incomplete')
            ports=[]
            for port in stack['ports'].values():
                with socket.socket() as sock:
                    sock.settimeout(.5);code=sock.connect_ex(('127.0.0.1',port))
                require(code==111,'Private loopback port is still reachable');ports.append(dict(port=port,connect_ex=code))
            report['records'].append(dict(arm=row['arm'],result=study.ref(result_path),stack=study.ref(stack_path),wrapper_receipt=study.ref(receipt_path),
                actual_selected_framework=str(framework),actual_model_context_without_engine_binding=result['actual_model_context_without_engine_binding'],
                actual_generations_retired=True,vm_absent=True,sqlite_action_counts=counts,model_request_files=0,
                movie=study.ref(movie),original_video_receipt=study.ref(video_receipt_path),decoded_frames=frames,ports=ports))
        require(arms==list(study.ARMS),'Both separate study arms required exactly once')
        report['passed']=True
    except BaseException as error:report.update(error_kind=type(error).__name__,error=str(error));raise
    finally:(output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','cpu','output'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--stacks',required=True,nargs=2,type=Path)
    args=parser.parse_args();r=audit(args.study,args.stacks,args.cpu,args.output)
    print(json.dumps(dict(passed=r['passed'],actual_stacks=len(r['records']),model_requests=0,device_inputs=0)))
