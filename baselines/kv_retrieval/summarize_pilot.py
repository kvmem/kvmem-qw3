"""Export complete-task outcomes and request/observation latency without mixing summaries with actions."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    def quantile(q):
        p = (len(values) - 1) * q
        i = int(p)
        return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (p - i)
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p95": quantile(.95), "p99": quantile(.99), "max": values[-1]}


def native_phases(path):
    result, pending = [], {}
    for line in path.read_text().splitlines():
        numbers = {k: float(v) for k, v in re.findall(r'\b([A-Za-z_][A-Za-z_0-9]*)=(-?\d+(?:\.\d+)?)', line)}
        if '[qw3-native-accounting]' in line:
            pending.update(numbers)
        elif 'native generate: prompt_tokens=' in line:
            pending.update(numbers)
        elif 'native continuous_batch: request=' in line:
            pending.update(numbers)
            pending.update(prefill_ms=numbers['prefill'] * 1000,
                           decode_ms=numbers['decode'] * 1000)
        elif '[qw3-server-ttft]' in line:
            result.append({**pending, **numbers})
            pending = {}
    return result


def align_steps(requests, client_events):
    requests = sorted(requests, key=lambda row: row['received_at'])
    by_message = {r['message_id']: i for i, r in enumerate(requests) if r.get('message_id')}
    summary_indices, boundaries, observations = set(), [], []
    pending_boundary = None
    tool_ids, tool_results = set(), set()
    for wrapped in client_events:
        event = wrapped['event']
        if event.get('subtype') == 'compact_boundary':
            pending_boundary = event.get('compact_metadata', {})
            boundaries.append(pending_boundary)
        message = event.get('message', {})
        if pending_boundary is not None and event.get('type') == 'assistant':
            index = by_message.get(message.get('id'))
            if index is not None:
                # Native Claude emits the boundary before the first post-summary
                # assistant. The preceding serial generation is its summary.
                if index:
                    summary_indices.add(index - 1)
                pending_boundary = None
        content = message.get('content', [])
        if not isinstance(content, list):
            continue
        fresh_observation = False
        for block in content:
            if block.get('type') == 'tool_use' and block.get('id'):
                tool_ids.add(block['id'])
            if block.get('type') == 'tool_result' and block.get('tool_use_id') not in tool_results:
                tool_results.add(block.get('tool_use_id'))
                fresh_observation = True
        if fresh_observation:
            observations.append(wrapped['observed_at'])
    used_observations, steps = set(), []
    for index, request in enumerate(requests):
        eligible = [t for t in observations if t <= request['received_at'] and t not in used_observations]
        observation = max(eligible) if eligible else None
        summary = index in summary_indices
        row = {**request, 'request_index': index, 'is_compaction': summary,
               'observation_to_first_content_s': None, 'observation_to_first_tool_call_s': None}
        if not summary and observation is not None and request.get('api_first_content_s') is not None:
            used_observations.update(t for t in observations if t <= observation)
            row['observation_to_first_content_s'] = request['received_at'] + request['api_first_content_s'] - observation
            if request.get('first_tool_call_s') is not None:
                row['observation_to_first_tool_call_s'] = request['received_at'] + request['first_tool_call_s'] - observation
        steps.append(row)
    return steps, {'compact_boundaries': boundaries, 'identified_summary_requests': len(summary_indices),
                   'unique_tool_calls': len(tool_ids), 'unique_tool_results': len(tool_results),
                   'boundary_without_following_response': pending_boundary is not None}


def summarize(run):
    result = json.loads((run / 'result.json').read_text())
    requests = [r for r in jsonl(run / 'proxy/requests.jsonl') if r['path'].split('?')[0] == '/v1/messages']
    rows, client = align_steps(requests, jsonl(run / 'client_events.jsonl'))
    agent = run / 'cell/agent_summary.json'
    agent = json.loads(agent.read_text()) if agent.exists() else {}
    summary = {'run': str(run.resolve()), 'method': result['method'], 'task_id': result['task']['task_id'],
        'status': result['status'], 'performance_eligible': result['performance_eligible'],
        'shared_gpu': result['shared_gpu'], 'official_result': result.get('cell_summary'),
        'agent_wall_s': agent.get('elapsed_sec'), 'model_requests': len(rows),
        'failed_model_requests': sum(r['status'] != 'completed' for r in rows),
        'request_wall_sum_s': sum(r.get('request_wall_s', 0) for r in rows),
        'output_tokens': sum((r.get('output_usage') or {}).get('output_tokens', 0) for r in rows),
        'decoded_token_scope': 'API output only; native private retrieval query work reported separately in time',
        'api_first_content_s': stats([r.get('api_first_content_s') for r in rows]),
        'regular_api_first_content_s': stats([r.get('api_first_content_s') for r in rows if not r['is_compaction']]),
        'compaction_api_first_content_s': stats([r.get('api_first_content_s') for r in rows if r['is_compaction']]),
        'observation_to_first_content_s': stats([r['observation_to_first_content_s'] for r in rows]),
        'observation_to_first_tool_call_s': stats([r['observation_to_first_tool_call_s'] for r in rows]),
        'peak_process_gpu_gib': result['peak_process_gpu_gib'], 'client': client}
    summary['observation_clock'] = 'host receipt of Claude tool_result JSON event; subject to client stdout buffering'
    summary['compaction_classification_complete'] = len(client['compact_boundaries']) == client['identified_summary_requests']
    if result['method'] == 'infllm':
        phases = jsonl(run / 'server/requests.jsonl')
        summary['prefill_s'] = sum(r.get('prefill_s', 0) for r in phases)
        summary['decode_s'] = sum(r.get('decode_s', 0) for r in phases)
        summary['unexpected_cold_prefills'] = sum(r.get('unexpected_cold_prefill', False) for r in phases)
    else:
        phases = native_phases(run / 'server.log')
        summary['prefill_s'] = sum(r.get('prefill_ms', 0) for r in phases) / 1000
        summary['decode_s'] = sum(r.get('decode_ms', 0) for r in phases) / 1000
        summary['private_query_s'] = sum(r.get('post_guided_query_ms', 0) for r in phases) / 1000
        summary['middecode_query_s'] = sum(r.get('kvmem_middecode_query_ms', 0) for r in phases) / 1000
        summary['query_replay_s'] = sum(r.get('post_query_replay_ms', 0) for r in phases) / 1000
    return summary, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summaries, rows = [], []
    for run in a.run:
        summary, steps = summarize(run)
        summaries.append(summary)
        for row in steps:
            rows.append({'task_id': summary['task_id'], 'method': summary['method'],
                **{k: v for k, v in row.items() if not isinstance(v, (list, dict))}})
    (a.out / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    if rows:
        with (a.out / 'requests.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(row.keys() for row in rows))))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
