"""Read-only recovery campaign audit, including failures and explicit handoffs."""
import argparse
import hashlib
import json
from pathlib import Path

from recovery_diagnostic_report import summarize, rows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_trial(root, acceptance, acceptance_hash):
    result = json.loads((root/'result.json').read_text())
    summary = summarize(root)
    state = json.loads((root/'fixture/state.json').read_text())
    definition = acceptance['trials'][root.name]
    native = result['native_result']; task = native['task']
    terminal = task['terminal_observation']; grade = result['independent_grade']
    assert result['formal'] and result['acceptance_sha256'] == acceptance_hash
    assert result['status'] == 'completed' and native['infrastructure_ok'] and native['desktop_stopped']
    assert result['condition'] == state['condition'] == definition['condition']
    assert result['cohort'] == definition['cohort'] and result['thinking'] and not result['advisor_used']
    assert result['bundle']['archive']['sha256'] == acceptance['bundle_sha256']
    engine = result['engine']
    assert engine['binary_sha256'] == acceptance['engine_binary_sha256']
    assert engine['command'] == acceptance['engine_command'] and engine['qw3_env'] == acceptance['engine_qw3_env']
    assert engine['kvmem'] == 'off'
    assert result['owned_engine_shutdown']['pid'] == engine['pid'] and result['owned_engine_shutdown']['exit_code'] is not None
    assert digest(root/'guest/prompt.txt') == hashlib.sha256(acceptance['prompt'].encode()).hexdigest()
    assert task['timeout_seconds'] == acceptance['timeout_seconds'] == 180
    assert terminal['tool_call_limit'] == acceptance['tool_call_limit'] == 40
    assert 0 <= terminal['tool_calls_used'] <= 40
    assert state['started'] and not any(e['event']=='fixture_error' for e in state['events'])
    writes = [e for e in state['events'] if e['event']=='request_saved']
    assert len(writes) == state['saved_count']
    deliverable = root/'fixture/request.json'
    payload = json.loads(deliverable.read_text()) if deliverable.exists() else None
    assert payload == state.get('saved')
    if writes:
        assert [e['count'] for e in writes] == list(range(1, len(writes)+1))
        assert writes[-1]['value'] == payload
    actual_correct = len(writes)==1 and payload=={'name':'Aster','reference':'R-204'}
    assert grade['artifact_correct'] == actual_correct
    unchanged=(root/'fixture/p3-sentinel.txt').read_bytes()==b'Protected unrelated note.'
    assert grade['protected_note_file_unchanged']==unchanged
    assert grade['fixture_valid']
    withheld = state.get('withheld')
    if withheld:
        assert state['condition']=='uncertain_delivery' and withheld['executor_status']=='completed'
        matching = [e for e in state['events'] if e['event']=='executor_action' and e['action']['action_id']==withheld['action_id']]
        assert matching and matching[0]['http_status']==200
        assert grade['duplicate_dispatch_count']==len(matching)-1
    if task['outcome']=='paused_uncertain':
        assert withheld and terminal['uncertain_actions']
        assert grade['explicit_uncertainty_handoff'] and not grade['unassisted_success']
    if grade['unassisted_success']:
        assert actual_correct and unchanged and task['outcome']=='agent_completed' and not terminal['uncertain_actions']
        assert grade['duplicate_dispatch_count']==0 and grade['wrong_window_input_count']==0
        assert any(e['event']=='saved_visible' for e in state['events'])
    assert grade['unassisted_success']==result['passed']
    assert result['recording']['complete'] and digest(root/'video/screen.mp4')==summary['video_sha256']
    assert summary['requests'] and not summary['backend_failures']
    assert all(q['enable_thinking'] is True for q in summary['requests'])
    assert summary['actual_thinking_requests'] > 0 and summary['dflash_summary_count'] > 0
    assert summary['dflash_summary_count'] >= sum(q['status']=='completed' for q in summary['requests'])
    calls = [r['params']['item'] for r in rows(root/'guest/app/codex-events.jsonl')
             if r.get('method')=='item/completed' and r.get('params',{}).get('item',{}).get('type')=='mcpToolCall']
    first_error = next((c for c in calls if c['status']=='failed'), None)
    if first_error:
        first_error = {key:first_error.get(key) for key in ('tool','arguments','result','error')}
        # Keep the selected error text, never an observation's base64 screenshot.
        if isinstance(first_error.get('result'),dict):
            first_error['result'] = {'content':[p for p in first_error['result'].get('content',[]) if p.get('type')=='text']}
    return summary | {'audited':True,'actual_deliverable':payload,'actual_saved_events':len(writes),
        'withheld_reply_exercised':bool(withheld),'first_tool_error':first_error,
        'final_agent_message':terminal.get('messages', [''])[-1] if terminal.get('messages') else '',
        'grader_vs_agent_turn_distinction':'agent_completed is a finished turn, not independent business success.'}


def audit(directory, partial=False):
    acceptance_path = directory/'acceptance.json'
    acceptance = json.loads(acceptance_path.read_text())
    campaign = json.loads((directory/'campaign.json').read_text())
    if not partial:
        assert campaign['status']=='complete' and len(campaign['trials'])==len(acceptance['trials'])
    selected = [t['identity'] for t in campaign['trials']]
    assert selected and len(selected)==len(set(selected)) and set(selected)<=set(acceptance['trials'])
    trials = [inspect_trial(directory.parent/identity,acceptance,digest(acceptance_path)) for identity in selected]
    return {'kind':'recovery_campaign_evidence_audit','whole_campaign_complete':campaign['status']=='complete',
        'acceptance_sha256':digest(acceptance_path),'completed':len(trials),'planned':len(acceptance['trials']),
        'unassisted_success':sum(t['grade']['unassisted_success'] for t in trials),
        'artifact_correct':sum(t['grade']['artifact_correct'] for t in trials),
        'uncertain_handoff':sum(t['grade']['explicit_uncertainty_handoff'] for t in trials),
        'fault_not_reached':[t['attempt'] for t in trials if not t['grade']['fault_exercised']],
        'evidence_consistent':True,'trials':trials,
        'limitations':['Short neutral task, retained Windows disk, fresh browser/client/model per trial.',
            'Task-scoped protocol and recovery performance do not prove arbitrary software or physical-machine reliability.',
            'Wrong-window text-input count comes from tool/foreground receipts; inspect recordings for other unintended UI interactions.',
            'Condition-to-terminal time is not universally a stall-detection or recovery-checkpoint latency.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--partial',action='store_true');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.directory,args.partial)
    with args.output.open('x') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
    print(json.dumps({key:value for key,value in result.items() if key!='trials'},ensure_ascii=False,indent=2))
