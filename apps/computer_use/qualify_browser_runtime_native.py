"""Qualify the candidate's actual native reader in a separately owned VM.

Reuses the owned Chrome/source probe lifecycle. No model or calibrated GUI
input is sent. This checks the real reader, not task success or dispatch gains.
"""
import argparse
import asyncio
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import qualify_chrome_tab_source as native


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_candidate(folder):
    package=folder/'framework/qw3_computer';name='qw3_browser_runtime_native_qualification'
    spec=importlib.util.spec_from_file_location(name,package/'__init__.py',submodule_search_locations=[str(package)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return importlib.import_module(name+'.linux_driver'),importlib.import_module(name+'.browser_runtime_metadata')


def run(args):
    folder=args.candidate.resolve();manifest=json.loads((folder/'build.json').read_text())
    if manifest.get('kind')!='isolated_sdk51_native_browser_runtime_candidate':raise ValueError('Require isolated browser candidate')
    for name,expected in manifest['candidate_files_sha256'].items():
        if Path(name).name!=name or digest(folder/'framework/qw3_computer'/name)!=expected:raise ValueError('Candidate source changed')
    for name,expected in manifest.get('unchanged_auxiliary_framework_files_sha256',{}).items():
        original=Path(manifest['auxiliary_source_root'] or Path(manifest['source_package']).parent)/name
        if digest(original)!=expected or digest(folder/'source-archive/auxiliary'/name)!=expected:
            raise ValueError('Pinned original runner or grader source changed')
    qualification=json.loads(args.qualification.read_text())
    if qualification.get('passed') is not True or qualification.get('tests_run',0)<20:raise ValueError('Require completed CPU qualification')
    for name,expected in qualification['source_sha256'].items():
        if digest(Path(__file__).parent/name)!=expected:raise ValueError('CPU qualified source changed')
    if manifest['candidate_files_sha256']['browser_runtime_metadata.py']!=qualification['source_sha256']['browser_runtime_metadata.py']:
        raise ValueError('Candidate helper differs from qualification')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    state=dict(kind='actual_native_browser_runtime_reader_qualification',status='preparing',started_at=time.time(),
        producer=native.owned.process_identity(__import__('os').getpid()),recipe_sha256=digest(__file__),
        source_probe_recipe_sha256=digest(native.__file__),candidate_build_sha256=digest(folder/'build.json'),
        cpu_qualification_sha256=digest(args.qualification),model_calls=0,existing_device_inputs=0,
        calibrated_key_mouse_actions=0,administrative_browser_launch=True,real_model_retest=False,
        qualified_for_production=False,actual_resource_change_observed=False)
    def save(**values):
        state.update(values);temporary=output/'qualification.tmp';temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');temporary.replace(output/'qualification.json')
    save();driver_module,helper=load_candidate(folder);original_snapshot=native.native_snapshot;reads=[]
    async def capture(index):
        driver=driver_module.LinuxDriver('http://127.0.0.1:'+str(native.PORTS['server']),'native-browser-source-reader')
        await driver.start()
        try:
            raw=await driver.observe();png=raw.pop('png');path=output/f'candidate-source-{index:03d}.png';path.write_bytes(png)
            return raw|dict(capabilities=driver.capabilities,screenshot=str(path),sha256=digest(path))
        finally:await driver.close()
    def extended_snapshot(expected,title,*,startup=False):
        snapshot=original_snapshot(expected,title,startup=startup)
        if not startup:
            began=time.monotonic()
            try:
                observed=asyncio.run(capture(len(reads)));observed['read_elapsed_seconds']=time.monotonic()-began
                tabs=[c for c in observed.get('accessibility',{}).get('controls',[]) if helper.valid_metadata(c,observed['foreground']) is not None]
                observed['qualified_native_tabs']=len(tabs)
                observed['registered_browser_owner_matches']=bool(tabs) and all(c['browser_runtime_metadata']['owner_pid']==expected['pid'] for c in tabs)
            except Exception as error:observed=dict(error_kind=type(error).__name__,error=str(error),qualified_native_tabs=0)
            reads.append(observed);snapshot['candidate_reader_observation']=observed
            save(status='reading-native-candidate',candidate_reads=len(reads))
        return snapshot
    try:
        with patch.object(native,'native_snapshot',extended_snapshot):probe=native.run(argparse.Namespace(output=output/'probe'))
        checks=dict(original_native_source_checks_passed=probe.get('passed') is True,
            candidate_native_selected_tab_observed=bool(reads) and all(x.get('qualified_native_tabs')==1 for x in reads),
            native_owner_matches_registered_browser=bool(reads) and all(x.get('registered_browser_owner_matches') is True for x in reads),
            original_raw_label_preserved=all(c['name']==c['browser_runtime_metadata']['page_title']+' - Memory usage - '+c['browser_runtime_metadata']['resource']
                for x in reads for c in x.get('accessibility',{}).get('controls',[]) if 'browser_runtime_metadata' in c),
            dedicated_vm_retired=probe.get('dedicated_vm_retired') is True,
            no_model_or_calibrated_input=state['model_calls']==state['calibrated_key_mouse_actions']==state['existing_device_inputs']==0)
        resources=sorted({c['browser_runtime_metadata']['resource'] for x in reads for c in x.get('accessibility',{}).get('controls',[]) if 'browser_runtime_metadata' in c})
        save(status='passed' if all(checks.values()) else 'failed',passed=all(checks.values()),checks=checks,
            reads=reads,resources=resources,actual_resource_change_observed=len(resources)>1,finished_at=time.time(),
            limits=['Reader/source qualification only; no native dispatch or QW3 business gain demonstrated.',
                    'Raw task text, labels and completion predicates remain unchanged; old observations are not retroactively admitted.'])
    except BaseException as error:
        save(status='failed',passed=False,error_kind=type(error).__name__,error=str(error),reads=reads,finished_at=time.time());raise
    if not state['passed']:raise RuntimeError('Candidate native browser source qualification failed')
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    r=run(p.parse_args());print(json.dumps({k:r.get(k) for k in ('status','checks','actual_resource_change_observed')}))


if __name__=='__main__':main()
