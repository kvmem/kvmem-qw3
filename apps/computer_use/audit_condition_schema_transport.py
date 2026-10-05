"""Independent read-only audit of the actual private Codex transport captures."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import tomllib
import types
from urllib.parse import urlparse

from jsonschema import Draft202012Validator

import condition_schema_candidate as candidate
import kvmem_osworld_queue as owned


def require(condition, message):
    if not condition:raise ValueError(message)


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=candidate.digest(path))


def audit(report_path, output):
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    require(output.is_relative_to(candidate.RUNTIME), 'Use a goal-owned new audit directory')
    report = json.loads(report_path.read_text());require(report['passed'], 'Actual transport must be completed')
    require(report['fixture_backend'] and not report['real_model'] and report['model_inference_requests']==0,
        'Require an explicit fixture backend, not QW3 qualification')
    require(report['fixture_backend_requests']==6 and report['tool_dispatch_attempts']==0 and report['device_inputs']==0,
        'Actual dispatch inventory differs')
    root = report_path.parent
    build_path = Path(report['build']);require(candidate.digest(build_path)==report['build_sha256'], 'Candidate build changed')
    build = json.loads(build_path.read_text());base_path = Path(build['base_build'])
    require(candidate.digest(base_path)==candidate.BASE_BUILD_SHA==build['base_build_sha256'], 'Original base build changed')
    refs = [ref(report_path), ref(build_path), ref(base_path)]
    for name, digest in build['base_candidate_files_sha256'].items():
        path = base_path.parent/'framework'/name
        require(candidate.digest(path)==digest, 'Original base source changed: '+name);refs.append(ref(path))
    for name, digest in build['candidate_files_sha256'].items():
        path = build_path.parent/'framework'/name
        require(candidate.digest(path)==digest, 'Private candidate source changed: '+name);refs.append(ref(path))
        if name not in ['qw3_computer/protocol.py','qw3_computer/condition_schema_wire.py']:
            require(build['base_candidate_files_sha256'][name]==digest, 'Unrelated package behavior changed')
    require(build['changed_package_files']==['qw3_computer/protocol.py','qw3_computer/condition_schema_wire.py'], 'Candidate scope changed')
    for name, digest in report['source_sha256'].items():
        path = Path(__file__).with_name(name)
        require(candidate.digest(path)==digest==candidate.digest(root/name), 'Qualification source changed: '+name);refs.append(ref(root/name))
    # Load the original converter, not the repaired helper. Construct the sole
    # expected change independently from the original trusted declaration.
    package_name = '_independent_original_native_transport_audit'
    package = types.ModuleType(package_name);package.__path__=[str(base_path.parent/'framework/qw3_computer')]
    sys.modules[package_name]=package
    modules={}
    for name in ('protocol','tool_schema'):
        path=base_path.parent/'framework/qw3_computer'/f'{name}.py'
        spec=importlib.util.spec_from_file_location(package_name+'.'+name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);modules[name]=module
    declared=modules['tool_schema'].WORKFLOW_TOOLS[0]['inputSchema']['properties']['conditions']
    validator=Draft202012Validator(declared)
    observed_order=[(r['platform'],r['workflow']) for r in report['cases']]
    require(observed_order==[(p,w) for p in ('windows','linux','android') for w in (False,True)], 'Six actual catalogs changed')
    output.mkdir(parents=True,exist_ok=False,mode=0o700);shutil.copyfile(__file__,output/'audit-source.py')
    result=dict(kind='independent_actual_private_condition_transport_audit',checked_at=time.time(),passed=False,
        source_report=ref(report_path),audit_source=ref(output/'audit-source.py'),cases=[],source_refs=refs,
        model_calls=0,device_inputs=0,process_signals=0,business_improvement_proven=False)
    try:
        for index,row in enumerate(report['cases'],1):
            require(row['passed'] and row['all_registered_processes_retired'], 'Actual case not complete')
            folder=root/(row['platform']+('-verified' if row['workflow'] else '-plain'))
            raw_path=folder/'responses-request.json';body_path=folder/'fixture-backend-request.json'
            require(candidate.digest(raw_path)==row['actual_request_sha256'], 'Actual Responses bytes changed')
            require(candidate.digest(body_path)==row['forwarded_request_sha256'], 'Actual forwarded bytes changed')
            raw=json.loads(raw_path.read_text());body=json.loads(body_path.read_text())
            require(raw==json.loads((root/f'responses-raw-{index}.json').read_text()), 'Raw HTTP capture mismatch')
            require(body==json.loads((root/f'fixture-backend-{index}.json').read_text()), 'Backend HTTP capture mismatch')
            expected=modules['protocol'].convert_request(raw,'qw3',16384,allow_thinking=True).body
            allowed={'mcp__computer__observe','mcp__computer__act'}|({'mcp__computer__workflow'} if row['workflow'] else set())
            expected['tools']=[t for t in expected['tools'] if t['function']['name'] in allowed]
            require({t['function']['name'] for t in body['tools']}==allowed, 'Actual model tool permissions changed')
            if row['workflow']:
                act=next(t['function']['parameters'] for t in expected['tools'] if t['function']['name']=='mcp__computer__act')
                act['properties']['step']['properties']['conditions']=copy.deepcopy(declared)
            require(expected==body, 'Actual request changed outside inline conditions and unchanged tool policy')
            traces=[]
            for path in (root/'inference').glob('*/request.json'):
                if json.loads(path.read_text())==body:traces.append(path)
            require(len(traces)==1,'Actual bridge trace association is not unique')
            metric_path=traces[0].parent/'metrics.json';metric=json.loads(metric_path.read_text())
            require(metric['status']=='completed','Actual bridge fixture stream not terminal')
            config_path=folder/'codex/config.toml';config=tomllib.loads(config_path.read_text())
            catalog_path=Path(config['model_catalog_json']);catalog=json.loads(catalog_path.read_text())
            require(config['model_provider']=='qw3' and config['model_reasoning_effort']=='medium', 'Actual provider/thinking profile differs')
            require(catalog['models'][0]['context_window']==524288,'Actual context profile differs')
            url=urlparse(config['model_providers']['qw3']['base_url'])
            require(url.hostname=='127.0.0.1' and url.scheme=='http','Actual provider was not loopback')
            ports={url.port}
            for identity in [row['actual_codex_identity']]+row['actual_descendants']:
                require(identity['uid']==os.getuid() and identity['cwd']==str(folder/'codex'),'Actual process owner/directory differs')
                current=owned.process_identity(identity['pid'])
                require(current is None or current['state']=='Z' or current['start_ticks']!=identity['start_ticks'],
                    'Same actual qualification generation remains live')
                if identity in row['actual_descendants']:
                    argv=identity['command'];require(str(build_path.parent/'framework/qw3_computer/mcp_proxy.py') in argv,'Unqualified MCP child')
                    endpoint=urlparse(argv[argv.index('--endpoint')+1]);require(endpoint.hostname=='127.0.0.1','MCP endpoint not private loopback');ports.add(endpoint.port)
            for port in ports:
                with socket.socket() as probe:
                    probe.settimeout(.2);require(probe.connect_ex(('127.0.0.1',port))!=0,'Qualification listener remains reachable')
            events_path=folder/'codex-events.jsonl';events=[json.loads(line) for line in events_path.read_text().splitlines()]
            completions=[e for e in events if e.get('method')=='turn/completed' and e['params']['turn']['id']==row['turn']]
            require(len(completions)==1 and completions[0]['params']['turn']['status']=='completed','Actual Codex turn not terminal')
            require(not any(e.get('method')=='item/completed' and e.get('params',{}).get('item',{}).get('type')=='mcpToolCall' for e in events),'A tool was called')
            if row['workflow']:
                for condition in [[{'kind':'control_value','text':'Name'}],[{'kind':'screen_changed','text':'bad'}],[],[{'kind':'screen_changed'}]*13]:
                    require(bool(list(validator.iter_errors(condition))), 'Required field/bound no longer enforced')
                for kind in ['control_text','control_absent','foreground_title','foreground_app']:
                    require(not list(validator.iter_errors([{'kind':kind,'text':'literal'}])), 'Original positive condition missing')
                require(not list(validator.iter_errors([{'kind':'control_value','text':'Name','value':'Alice','control_kind':'entry'}])), 'Original field value condition missing')
                require(not list(validator.iter_errors([{'kind':'screen_changed'}])), 'Original transition condition missing')
            for path in [raw_path,body_path,root/f'responses-raw-{index}.json',root/f'fixture-backend-{index}.json',traces[0],metric_path,config_path,catalog_path,events_path]:
                refs.append(ref(path))
            result['cases'].append(dict(platform=row['platform'],workflow=row['workflow'],passed=True,
                actual_request_only_inline_conditions_changed=True,actual_generation_retired=True,private_listeners_unreachable=True,
                native_device_inputs=0,model_inference_requests=0))
        binary=Path(shutil.which('codex')).resolve()
        version=subprocess.run([str(binary),'--version'],capture_output=True,text=True,timeout=10)
        require(version.returncode==0 and version.stdout.strip()=='codex-cli 0.159.2','Pinned actual Codex version changed')
        result.update(passed=True,actual_codex_binary=ref(binary),actual_codex_version=version.stdout.strip(),source_refs_checked=len(refs))
    except BaseException as error:result.update(error_kind=type(error).__name__,error=str(error));raise
    finally:
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        for name in list(sys.modules):
            if name==package_name or name.startswith(package_name+'.'):sys.modules.pop(name)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=audit(args.report,args.output)
    print(json.dumps(dict(passed=result['passed'],actual_cases=len(result['cases']),source_refs_checked=result['source_refs_checked'],model_calls=0,device_inputs=0)))
