"""Read-only timing and outcome summary for an end-to-end native preflight."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3


def distribution(values):
    values = sorted(values)
    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = int(position)
        return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)
    return dict(count=len(values), p50=percentile(.5), p95=percentile(.95), maximum=values[-1],
                mean=sum(values)/len(values)) if values else dict(count=0)


def log_rows(text, prefix):
    return [dict(re.findall(r'(\w+)=("[^"]*"|[^\s]+)', line))
            for line in text.splitlines() if line.startswith(prefix)]


def summarize(directory):
    directory = Path(directory)
    result = json.loads((directory / 'result.json').read_text())
    engine = (directory / 'engine/engine.log').read_text(errors='replace')
    timing = log_rows(engine, '[qw3-server-ttft]')
    accounting = log_rows(engine, '[qw3-server-accounting]')
    dispatch = log_rows(engine, '[qw3-server-dispatch]')
    drafts = log_rows(engine, '[qw3] native dflash_spec_summary:')
    report = dict(kind='native_preflight_evidence_summary', formal_run=False,
                  task=result['task'], run=result['run'], kvmem=result['kvmem'],
                  pipeline_passed=result['pipeline_passed'],
                  independent_grade=result.get('independent_grade'),
                  task_terminal=result.get('native_result', {}).get('task'),
                  percentile_method='linear interpolation of sorted observations')
    report['model_requests'] = dict(dispatched=len(dispatch), timing_rows=len(timing),
        accounting_rows=len(accounting), native_receipts=result.get('inference_request_receipts'),
        terminal_statuses=dict(Counter(r.get('terminal_status', 'unknown') for r in accounting)),
        dispatch_routes=dict(Counter(r.get('dispatch', 'unknown') for r in dispatch)),
        dispatch_is_not_speculative_decoder=True,
        server_first_model_token_ms=distribution([float(r['server_ttft_ms']) for r in timing if r.get('model_token_seen')=='1']),
        engine_first_model_token_ms=distribution([float(r['engine_ttft_ms']) for r in timing if r.get('model_token_seen')=='1']),
        response_first_visible_output_ms=distribution([float(r['response_ttft_ms']) for r in timing if r.get('visible_output_seen')=='1']),
        total_request_ms=distribution([float(r['request_total_ms']) for r in timing]),
        timing_units='milliseconds, each measured within one host request; not end-to-end tool latency')
    drafted = sum(int(r.get('drafted',0)) for r in drafts)
    accepted = sum(int(r.get('accepted',0)) for r in drafts)
    report['dflash_actual'] = dict(summary_count=len(drafts), enabled_count=sum(r.get('enabled')=='true' for r in drafts),
        fallback_count=sum(r.get('fallback')=='true' for r in drafts), drafted=drafted, accepted=accepted,
        weighted_acceptance=accepted/drafted if drafted else None,
        scope='all native DFlash summaries, including any internal retrieval work')
    prompt_counts = [int(v) for v in re.findall(r'^\[qw3-serve\] #\d+ chat[^\n]* prompt_tokens=(\d+)', engine, re.M)]
    report['largest_reported_prompt_tokens'] = max(prompt_counts, default=None)
    samples = [json.loads(line) for line in (directory / 'engine-resources.jsonl').read_text().splitlines()]
    report['engine_resources'] = dict(samples=len(samples),
        peak_gpu_mib=max((r.get('gpu_memory_bytes',0) for r in samples), default=0)/1024**2,
        peak_host_rss_gib=max((r.get('rss_bytes',0) for r in samples), default=0)/1024**3,
        scope='engine process and device samples including model startup; excludes Windows VM and native client memory')
    capture = [json.loads(line) for line in (directory / 'video/capture.jsonl').read_text().splitlines()]
    intervals = [(b['captured_monotonic_ns']-a['captured_monotonic_ns'])/1e9 for a,b in zip(capture,capture[1:])]
    video = result['recording']
    report['recording'] = dict(complete=video['complete'], seconds=video.get('elapsed_seconds'),
        actual_captures=len(capture), encoded_frames=video['video_frames'],
        actual_capture_gap_seconds=distribution(intervals),
        sampling_note='Encoded 10 fps repeats last available image; real VNC sampling intervals are reported separately.',
        recorder_cpu_seconds=video.get('recorder_process_cpu_seconds'),
        encoder_cpu_seconds=video.get('encoder_and_remux_cpu_seconds'))
    database = directory / 'guest/app/desktop/desktop.sqlite'
    if database.exists():
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
            actions = db.execute('SELECT request,status FROM actions ORDER BY created').fetchall()
        report['desktop_actions'] = dict(count=len(actions), statuses=dict(Counter(r[1] for r in actions)),
            kinds=dict(Counter(json.loads(r[0])['action']['kind'] for r in actions)))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as handle:
        json.dump(summarize(args.directory), handle, ensure_ascii=False, indent=2)
