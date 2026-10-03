"""Compare audited recovery attempts, retaining failures and uncertainty handoffs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def load_audits(paths):
    trials = []
    identities = set()
    for path in paths:
        audit = json.loads(path.read_text())
        if not audit.get('evidence_consistent'):
            raise ValueError('Only independently audited trials may be compared')
        for trial in audit['trials']:
            if trial['attempt'] in identities:
                raise ValueError('Duplicate audited trial')
            identities.add(trial['attempt'])
            trials.append(trial)
    return trials


def aggregate(trials):
    return {
        'valid_trials': len(trials),
        'unassisted_success': sum(t['grade']['unassisted_success'] for t in trials),
        'artifact_correct': sum(t['grade']['artifact_correct'] for t in trials),
        'uncertain_handoff': sum(t['grade']['explicit_uncertainty_handoff'] for t in trials),
        'business_failure': sum(not t['grade']['artifact_correct'] for t in trials),
        'task_seconds_median': statistics.median(t['grade']['elapsed_seconds'] for t in trials),
        'tool_calls_median': statistics.median(t['grade']['tool_calls'] for t in trials),
        'tool_errors_total': sum(t['tool_errors'] for t in trials),
        'executor_rejections_total': sum(t['grade']['rejected_calls'] for t in trials),
        'wrong_window_text_inputs': sum(t['grade']['wrong_window_input_count'] for t in trials),
        'duplicate_dispatches': sum(t['grade']['duplicate_dispatch_count'] for t in trials),
        'fault_not_reached': [t['attempt'] for t in trials if not t['grade']['fault_exercised']],
        'bundle_sha256': sorted(set(t['bundle_sha256'] for t in trials)),
    }


def pair_key(trial):
    match = re.search(r'-r([12])-([0-9]{2})(?:-retry[0-9]+)?$', trial['attempt'])
    if not match:
        raise ValueError('Recovery trial has no frozen repetition identity')
    return trial['condition'], int(match[1])


def compare(baseline, candidate):
    left = {pair_key(t): t for t in baseline}
    right = {pair_key(t): t for t in candidate}
    if len(left) != len(baseline) or len(right) != len(candidate):
        raise ValueError('Repeated condition/repetition identity')
    pairs = []
    for key in sorted(left.keys() & right.keys()):
        a, b = left[key], right[key]
        if a['cohort'] != 'baseline' or b['cohort'] != 'candidate':
            raise ValueError('Mismatched cohort')
        if a['source_sha256'] != b['source_sha256'] or a['engine_binary_sha256'] != b['engine_binary_sha256']:
            raise ValueError('Common runner/executor/model changed across cohorts')
        pairs.append({'condition': key[0], 'repeat': key[1], 'baseline': a['attempt'], 'candidate': b['attempt'],
            'baseline_grade': a['grade'], 'candidate_grade': b['grade'],
            'task_seconds_difference': b['grade']['elapsed_seconds'] - a['grade']['elapsed_seconds'],
            'tool_calls_difference': b['grade']['tool_calls'] - a['grade']['tool_calls']})
    return {'kind': 'audited_local_recovery_comparison',
        'matched_pairs': len(pairs), 'expected_pairs': 16,
        'whole_comparison_complete': len(pairs) == len(left) == len(right) == 16,
        'baseline': aggregate(baseline), 'candidate': aggregate(candidate), 'pairs': pairs,
        'matched_baseline': aggregate([left[key] for key in sorted(left.keys() & right.keys())]),
        'matched_candidate': aggregate([right[key] for key in sorted(left.keys() & right.keys())]),
        'limits': ['Eight short neutral conditions repeated twice, not sixteen independent applications.',
            'Retained Windows disk and executor ledger; fresh task/browser/client/model per attempt.',
            'Latency medians include failed tasks and handoffs; no stable speedup ratio is claimed.',
            'A separately retained interrupted baseline attempt was replaced with a fresh same-condition identity.',
            'Artifact correctness with an uncertain pause is not unassisted success.',
            'Fault-to-terminal time is not universally stall-detection latency.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, nargs='+', required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare(load_audits(args.baseline), load_audits([args.candidate]))
    report['audits'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*args.baseline, args.candidate]}
    with args.output.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ('pairs', 'audits')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
