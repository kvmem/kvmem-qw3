"""Read-only P3 metrics; phase/event annotations never replace artifact grading."""
import argparse
import hashlib
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line] if path.exists() else []


def summarize(root):
    result=json.loads((root/'result.json').read_text())
    task=result.get('native_result',{}).get('task',{})
    calls=[r['params']['item'] for r in rows(root/'guest/app/codex-events.jsonl')
           if r.get('method')=='item/completed' and r.get('params',{}).get('item',{}).get('type')=='mcpToolCall']
    requests=[]
    for path in sorted((root/'guest/app/inference').glob('*/metrics.json')):
        metrics=json.loads(path.read_text());request=json.loads((path.parent/'request.json').read_text())
        chars=sum(len(choice.get('delta',{}).get('reasoning_content','') or '')
                  for chunk in rows(path.parent/'backend.jsonl') for choice in chunk.get('choices',[]))
        requests.append(dict(id=path.parent.name,status=metrics.get('status'),enable_thinking=request.get('enable_thinking'),
            actual_reasoning_chars=chars,elapsed_seconds=metrics.get('elapsed_seconds'),usage=metrics.get('usage')))
    resources=rows(root/'engine-resources.jsonl')
    log=(root/'engine/engine.log').read_text() if (root/'engine/engine.log').exists() else ''
    streak=longest=0
    for call in calls:
        streak=streak+1 if call.get('tool')=='observe' else 0
        longest=max(longest,streak)
    return dict(attempt=root.name,status=result['status'],formal=result.get('formal',False),condition=result['condition'],
        cohort=result['cohort'],grade=result.get('independent_grade'),result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
        source_sha256=result.get('source_sha256'),bundle_sha256=result.get('bundle',{}).get('archive',{}).get('sha256'),
        engine_binary_sha256=result.get('engine',{}).get('binary_sha256'),engine_stopped=result.get('owned_engine_shutdown'),
        tool_calls=len(calls),tool_call_budget_used=task.get('terminal_observation',{}).get('tool_calls_used'),
        observe_calls=sum(c.get('tool')=='observe' for c in calls),longest_observe_only_streak=longest,
        tool_errors=sum(c.get('status')=='failed' for c in calls),
        action_kinds=[c.get('arguments',{}).get('action',{}).get('kind') for c in calls if c.get('tool')=='act'],
        requests=requests,dflash_summary_count=log.count('dflash_spec_summary: enabled=true'),
        backend_failures=[r['id'] for r in requests if r['status'] in ('http_error','failed')],
        actual_thinking_requests=sum(r['actual_reasoning_chars']>0 for r in requests),
        device_gpu_peak_sampled_bytes=max((r.get('gpu_memory_bytes',0) for r in resources),default=None),
        host_engine_rss_peak_sampled_bytes=max((r.get('rss_bytes',0) for r in resources),default=None),
        video_complete=result.get('recording',{}).get('complete'),
        video_sha256=result.get('recording',{}).get('artifacts',{}).get('screen.mp4',{}).get('sha256'),
        evidence_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x') as handle:json.dump([summarize(p) for p in args.runs],handle,indent=2)
