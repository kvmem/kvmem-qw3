"""Scripted Python repair inputs, never credited as an agent repair success."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import time

from qw3_computer.software import Software,process_environment


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def probe(project):
    python=project/'.venv/Scripts/python.exe'
    result=subprocess.run([str(python),'-I','-c','import requests,pytest;print(requests.__version__);print(pytest.__version__)'],cwd=project,env=process_environment(),capture_output=True,text=True,timeout=30)
    return {'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}


async def run(args):
    if args.reset_from:
        return await reset(args)
    args.output.mkdir(parents=True,exist_ok=False)
    if args.workspace.exists():raise ValueError('Repair fixture workspace already exists')
    software=Software(args.workspace)
    report={'kind':'scripted_python_repair_fixture','model_used':False,'agent_success_claimed':False,'status':'preparing','started_at':time.time(),'jobs':[]}
    try:
        for project in ('repair-target','unrelated-project'):
            path=software.path('projects',project);path.mkdir()
            (path/'main.py').write_text('import requests\nprint(sum([7, 14, 21]))\n',encoding='utf-8')
            (path/'test_main.py').write_text('def test_total():\n    assert sum([7, 14, 21]) == 42\n',encoding='utf-8')
            job=await software.invoke('fixture-setup',{'operation':'setup_python','action_id':'fixture-'+project,'project':project,'profile':'python-web'})
            await software.tasks[job['id']]
            result=software.jobs[job['id']];report['jobs'].append(result.copy());save(args.output/'result.json',report)
            if result['status']!='completed' or not result.get('verification',{}).get('passed'):raise ValueError('Healthy prerequisite setup failed; do not inject faults')
            assert probe(path)['returncode']==0
        target=software.path('projects','repair-target');reference=software.path('projects','unrelated-project')
        report['source_hashes']={project:{name:digest(software.path('projects',project,name)) for name in ('main.py','test_main.py')} for project in ('repair-target','unrelated-project')}
        report['unrelated_project_files']={str(p.relative_to(reference)):digest(p) for p in reference.rglob('*') if p.is_file()}
        report['target_healthy_probe']=probe(target);report['reference_healthy_probe']=probe(reference)
        module=software.path('projects','repair-target','.venv','Lib','site-packages','requests')
        metadata=list(module.parent.glob('requests-*.dist-info'))
        if len(metadata)!=1:raise ValueError('Expected one pinned requests distribution')
        backup=software.path('fixture-backup');backup.mkdir()
        module.rename(backup/'requests-removed')
        launcher=software.path('projects','repair-target','open-terminal.cmd')
        (backup/'original-open-terminal.cmd').write_bytes(launcher.read_bytes())
        # A plausible broken launcher selects the other valid project. Do not run it here.
        launcher.write_text('@echo off\ncall "'+str(reference/'.venv/Scripts/activate.bat')+'"\ncmd /k\n',encoding='utf-8')
        report['target_broken_probe']=probe(target);report['reference_after_injection_probe']=probe(reference)
        report['faults']={'requests_module_absent':not module.exists(),'requests_metadata_still_present':metadata[0].is_dir(),'launcher_points_at_other_project':str(reference) in launcher.read_text(),'broken_launcher_sha256':digest(launcher)}
        if report['target_broken_probe']['returncode']==0 or report['reference_after_injection_probe']['returncode']!=0:raise ValueError('Fault injection did not isolate the target project')
        if report['unrelated_project_files']!={str(p.relative_to(reference)):digest(p) for p in reference.rglob('*') if p.is_file()}:raise ValueError('Unrelated project changed during injection')
        report.update(status='ready_fault_injected',workspace=str(args.workspace),target_project=str(target),reference_project=str(reference))
    except Exception as exc:report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        await software.close();report['finished_at']=time.time();save(args.output/'result.json',report)
    return report


async def reset(args):
    """Restore the same two faults only after the previous repair is verified."""
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads(args.reset_from.read_text(encoding='utf-8-sig'))
    if original['status'] != 'ready_fault_injected' or Path(original['workspace']) != args.workspace:
        raise ValueError('Reset belongs to a different repair workspace')
    software = Software(args.workspace)
    report = dict(original, status='resetting', started_at=time.time(), reset_from_sha256=digest(args.reset_from), jobs=[])
    try:
        if any(j['status'] not in {'completed', 'failed', 'cancelled'} for j in software.jobs.values()):
            raise ValueError('Previous software action is unresolved')
        target=software.path('projects','repair-target');reference=software.path('projects','unrelated-project')
        if probe(target)['returncode'] or probe(reference)['returncode']:
            raise ValueError('Reset requires both projects to be healthy before fault injection')
        if original['unrelated_project_files'] != {str(p.relative_to(reference)):digest(p) for p in reference.rglob('*') if p.is_file()}:
            raise ValueError('Unrelated project differs from the frozen input')
        for name, project in [('repair-target', target), ('unrelated-project', reference)]:
            if any(digest(project/file)!=value for file,value in original['source_hashes'][name].items()):
                raise ValueError('Source files differ from the frozen input')
        backup=software.path('fixture-resets', args.output.name)
        backup.mkdir(parents=True, exist_ok=False)
        launcher=software.path('projects','repair-target','open-terminal.cmd')
        (backup/'open-terminal.cmd.before').write_bytes(launcher.read_bytes())
        module=software.path('projects','repair-target','.venv','Lib','site-packages','requests')
        if len(list(module.parent.glob('requests-*.dist-info')))!=1:
            raise ValueError('Expected exactly one requests metadata directory')
        module.rename(backup/'requests-removed')
        launcher.write_text('@echo off\ncall "'+str(reference/'.venv/Scripts/activate.bat')+'"\ncmd /k\n',encoding='utf-8')
        if digest(launcher)!=original['faults']['broken_launcher_sha256']:
            raise ValueError('Reset launcher bytes differ from the original fault')
        report['target_broken_probe']=probe(target)
        if report['target_broken_probe']['returncode']==0:
            raise ValueError('Missing-module fault is not present')
        report.update(status='ready_fault_injected', reset_backup=str(backup), model_used=False, agent_success_claimed=False)
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        await software.close();report['finished_at']=time.time();save(args.output/'result.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workspace',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reset-from',type=Path)
    result=asyncio.run(run(parser.parse_args()));print(json.dumps({'status':result['status']}));raise SystemExit(0 if result['status']=='ready_fault_injected' else 1)
