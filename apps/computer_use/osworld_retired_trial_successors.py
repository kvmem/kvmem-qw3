"""Scoped successor admission for the one retained, retired OSWorld unknown."""
import copy
import hashlib
import json
from pathlib import Path

import kvmem_osworld_queue as owned
import osworld_queue_continuation as continuation


def command_path(identity, flag, path):
    command=identity.get('command',[])
    if command.count(flag)!=1:return False
    index=command.index(flag)+1
    if index>=len(command):return False
    value=Path(command[index])
    if not value.is_absolute():value=Path(identity['cwd'])/value
    return value.resolve()==Path(path).resolve()


def validate_followup(path):
    d=continuation.read(path)
    if d.get('kind')!='retired_022_frozen_successors' or d.get('original_profiles_changed') is not False:
        raise owned.SafetyHalt('Use the declared frozen successor continuation')
    for reference in d['files']:
        if owned.digest(reference['path'])!=reference['sha256']:raise owned.SafetyHalt('Successor evidence/source changed')
    case,p,original=continuation.validate(d['continuation_declaration'],live=False)
    if Path(d['continued_osworld_queue'])!=Path(case['continued_queue']):raise owned.SafetyHalt('Wrong continued parent')
    state=continuation.read(d['continued_osworld_queue'])
    identity=d['continuation_controller_identity']
    if state.get('pid')!=identity['pid'] or state.get('continuation_declaration_sha256')!=owned.digest(d['continuation_declaration']):
        raise owned.SafetyHalt('Continued parent identity changed')
    actual=owned.process_identity(identity['pid'])
    if actual is not None and not owned.same_process(identity,actual):raise owned.SafetyHalt('Continued parent PID reused')
    for reference in d['former_waiters']:
        previous=continuation.read(reference['path'])
        if (previous.get('pid')!=reference['pid'] or not previous.get('status','').startswith('halted')
                or previous.get('trials')!=[] or previous.get('child') or previous.get('engine_launcher_pid')):
            raise owned.SafetyHalt('Former waiter is not a zero-trial halted controller')
        if owned.process_identity(reference['pid']) is not None:raise owned.SafetyHalt('Former waiter PID remains or was reused')
    return d,case,p,original


def terminal_rows(protocol, state, original, results):
    expected=continuation.ordered_trials(protocol)
    rows=state.get('trials',[])
    if (state.get('status')!='completed' or len(rows)!=40
            or [r['id'] for r in rows]!=[r['id'] for r in expected]
            or rows[:len(original)]!=original or state.get('historical_unknowns_preserved')!=1):
        raise owned.SafetyHalt('Need all forty exact terminal rows with original failures preserved')
    for row in rows:
        result=results.get(row['id'])
        if (not isinstance(result,dict) or row.get('result')!=result or not result.get('finished_at')
                or type(result.get('uncertain_actions')) is not int
                or result['uncertain_actions']!=(1 if row['id']==continuation.CASE else 0)
                or result.get('owned_task_stop_unconfirmed')):
            raise owned.SafetyHalt('Changed or newly unknown terminal result')
    return copy.deepcopy(rows)


def terminal_parent(memory_protocol,state,followup):
    d,case,p,original=validate_followup(followup)
    campaign=Path(case['campaign'])
    if (Path(memory_protocol['parent_candidate_protocol']).resolve()!=campaign/'protocol.json'
            or state.get('protocol_sha256')!=memory_protocol['parent_candidate_protocol_sha256']
            or state.get('pid')!=d['continuation_controller_identity']['pid']):
        raise owned.SafetyHalt('Parent protocol or continued process identity changed')
    results={row['id']:continuation.read(campaign/'attempts'/row['id']/'result.json') for row in state.get('trials',[])}
    rows=terminal_rows(p,state,original,results)
    # The original unknown remains unknown. Its disappeared VM and services,
    # rather than a guessed delivery result, establish isolation for new tasks.
    owned.assert_no_services(campaign)
    for row in rows:owned.assert_vm_retired(campaign/'attempts'/row['id'])
    return [{'id':row['id'],'result_sha256':owned.digest(campaign/'attempts'/row['id']/'result.json'),
             'retired_unknown_preserved':row['id']==continuation.CASE} for row in rows]


def prepare(followup, declaration, memory_protocol, sdk_protocol, memory_state, sdk_state):
    followup,declaration,memory_protocol,sdk_protocol,memory_state,sdk_state=map(
        lambda p:Path(p).resolve(),(followup,declaration,memory_protocol,sdk_protocol,memory_state,sdk_state))
    case,_,_=continuation.validate(declaration,live=False)
    queue=continuation.read(case['continued_queue']);identity=owned.process_identity(queue['pid'])
    if (identity is None or identity['state']=='Z' or identity['uid']!=1001
            or identity['cwd']!=str(owned.ROOT) or 'osworld_queue_continuation' not in identity['command']
            or not command_path(identity,'--declaration',declaration)):raise owned.SafetyHalt('Register the actual continued controller')
    paths=[declaration,memory_protocol,sdk_protocol,memory_state,sdk_state,Path(__file__),
           Path(__file__).with_name('resumed_kvmem_queue.py'),Path(__file__).with_name('resumed_execution_retest_queue.py')]
    d={'kind':'retired_022_frozen_successors','original_profiles_changed':False,
       'continuation_declaration':str(declaration),'continued_osworld_queue':case['continued_queue'],
       'continuation_controller_identity':identity,'memory_protocol':str(memory_protocol),'sdk_protocol':str(sdk_protocol),
       'memory_state_dir':str(memory_state.parent.with_name('queue-runtime-02')),
       'sdk_state_dir':str(sdk_state.parent.with_name('queue-runtime-02')),
       'former_waiters':[{'path':str(path),'pid':continuation.read(path)['pid']} for path in (memory_state,sdk_state)],
       'files':[{'path':str(path),'sha256':owned.digest(path)} for path in paths],
       'scope':'Redirect successor waits to continued parent and exact retired unknown admission only; original eight and ten profiles unchanged.'}
    if followup.exists():raise owned.SafetyHalt('Successor declaration already exists')
    followup.write_text(json.dumps(d,indent=2)+'\n');validate_followup(followup)
    return d
