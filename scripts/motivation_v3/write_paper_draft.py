"""Build a reviewable English scheduling draft from saved, audited statistics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

DOMAINS = ['code', 'tool_state', 'dialogue', 'gui']
NAMES = ['software agents', 'tool/state workloads', 'multi-session conversations', 'GUI agents']


def interval(value, scale=1, digits=3):
    if value['mean'] is None:
        return 'unavailable'
    point = f"{value['mean'] * scale:.{digits}f}"
    lo, hi = value['ci95']
    if lo is None or hi is None:
        return point + ' (CI unavailable)'
    return point + f' [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]'


def positive(value):
    return value['ci95'][0] is not None and value['ci95'][0] > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    source = args.campaign / 'analysis'
    out = args.out or source
    out.mkdir(parents=True, exist_ok=True)
    paths = dict(analysis=source / 'analysis.json',
                 paired=source / 'paired_schedule_controls.json',
                 audit=args.campaign / 'integrity_audit.json', manifest=args.manifest)
    inputs = {k: json.loads(p.read_text()) for k, p in paths.items()}
    data, paired, audit, manifest = [inputs[k] for k in ['analysis', 'paired', 'audit', 'manifest']]
    complete = sum(s['complete_trajectories'] for s in data['summary'].values())
    calls = data['status_counts'].get('collected', 0)
    assert not audit['errors'], 'Draft requires an audit without errors'
    assert calls == audit['verified_requests'], 'Draft requires all included calls to be audited'
    assert all(r['collected_steps'] == r['planned_steps'] for r in data['trajectories'])
    manifest_rows = {r['id']: r for r in manifest['trajectories']}
    audited_counts = Counter(r['trajectory'] for r in audit['requests'])
    assert audited_counts == Counter({r['id']: r['collected_steps'] for r in data['trajectories']})
    for row in data['trajectories']:
        original = manifest_rows[row['id']]
        assert row['domain'] == original['domain']
        assert row['independent_unit'] == original['independent_unit']
        assert row['planned_steps'] == len(original['frames'])
    ready = audit['complete'] and complete == len(manifest['trajectories'])
    estimates = [data['summary'][d]['32'] for d in DOMAINS]
    eligible = sum(s['paired_rollouts'] for s in estimates)
    larger = sum(s['cross_gt_within'] for s in estimates)
    supporting = [name for name, s in zip(NAMES, estimates) if positive(s['paired_difference'])]
    interpretation = (
        'Each workload has a positive trajectory-bootstrap 95% confidence interval for the paired difference. '
        'These results support reconsidering historical relevance at step boundaries.'
        if len(supporting) == len(DOMAINS) else
        'The workload-specific confidence intervals below delimit the evidence for a boundary effect; '
        'the scheduling motivation must be qualified by workload.')
    lines = [
        '# Step-level memory scheduling: evidence-linked writing draft', '',
        f'Status: {"full primary collection audited" if ready else "interim snapshot"}; '
        'figures and interpretation still require final review.',
        f'Complete primary trajectories: {complete}/{len(manifest["trajectories"])}; '
        f'audited model calls: {calls}/{audit["planned_requests"]}.', '',
        '## Design paragraph', '',
        r'\sys adopts \emph{step-level memory scheduling} to bound historical working-set updates '
        'to one per agent invocation. During current-step prefill, it gathers model-native signals '
        'for historical selection. Once prefill completes, it refreshes the historical working set '
        'before generating the first decode token. The selected historical blocks remain fixed '
        'during decoding, while newly generated KV state is appended to the execution view. '
        'The next invocation opens a new recall epoch in which historical relevance is reconsidered.', '',
        '## Measured motivation', '',
        'We examine this scheduling choice using full-visible attention reference traces on recorded '
        'inputs from software agents, tool/state workloads, multi-session conversations, and GUI agents. '
        f'Traces are collected with {manifest["model"]}. '
        f'The analyzed cohort contains {complete} distinct source trajectories and {calls} model calls. '
        'We compare within-step and across-step attention changes using non-overlapping 32-query windows '
        'over verified common historical content, pairing comparisons within the same adjacent source steps. '
        f'Across-step divergence exceeds within-step divergence in {larger} of {eligible} eligible '
        'trajectory-level means. ' + interpretation, '',
        '| Workload | Complete trajectories | Eligible W=32 trajectories | Across > within | '
        'Paired JS difference, bits [95% CI] |',
        '| --- | ---: | ---: | ---: | --- |',
    ]
    for domain, name, stat in zip(DOMAINS, NAMES, estimates):
        lines.append(f'| {name} | {data["summary"][domain]["complete_trajectories"]} | '
                     f'{stat["paired_rollouts"]} | {stat["cross_gt_within"]} | '
                     f'{interval(stat["paired_difference"])} |')
    lines += ['',
        'Within-step stability must also be assessed directly. At a nominal historical-block budget '
        'of 10%, rounded up, we measure the change in conditional historical-attention coverage between '
        'the first and last decode quarters for fixed-prefill selection. We also compare it with a '
        'causal refresh every 128 decode queries, using the preceding 32 queries as the refresh signal. '
        'The paired effects are reported below; they measure coverage, not task accuracy.', '',
        '| Workload | Fixed selection: last minus first quarter, pp [95% CI] | '
        'Refresh every 128 queries: gain over fixed selection, pp [95% CI] |',
        '| --- | --- | --- |']
    for domain, name in zip(DOMAINS, NAMES):
        lines.append(f'| {name} | {interval(paired[domain]["fixed_quarter_change"], 100, 1)} | '
                     f'{interval(paired[domain]["periodic_128_gain"], 100, 1)} |')
    lines += ['',
        'Once-per-step scheduling therefore represents a choice between update frequency and historical '
        'coverage. These diagnostics do not establish that the selected working set remains lossless '
        'throughout decoding. Preserving end-task quality and reducing actual retrieval overhead '
        'require separate system measurements.', '',
        '## Sensitivity and scope to retain in the paper', '',
        'Window settings without a strictly positive workload-level paired-difference 95% CI:', '']
    exceptions = []
    for domain, name in zip(DOMAINS, NAMES):
        for width in ['16', '32', '64', '128']:
            value = data['summary'][domain][width]['paired_difference']
            if not positive(value):
                exceptions.append(f'- {name}, W={width}: {interval(value)} bits, n={value["n"]}.')
    lines += exceptions or ['All reported workload/window estimates have strictly positive intervals.']
    lines += ['',
        '- Windows may have different eligible trajectories; report their sample counts rather than '
        'treating all window comparisons as a fixed cohort.',
        '- Inputs are replayed from source records. Newly generated actions are neither executed nor '
        'appended to subsequent recorded inputs. These are not fresh closed-loop task evaluations.',
        '- Exact attention is a diagnostic selector, distinct from the production compact index. '
        'Coverage is conditional on eligible historical blocks; budget rounding and saturation '
        'are reported separately.',
        '- Measurements cover 16 standard attention layers with heads summed, not every head or '
        'the model\'s linear-memory components.',
        '- Absolute decode-distance bins have changing survivor populations. Common-history-length '
        'stratification was added after observing shortened GUI prefixes and is exploratory.',
        '- Refresh counts are an overhead proxy. This experiment does not measure actual index lookup '
        'or KV transport latency, million-token workspace scaling, or task-quality effects.', '',
        '## Cohort amendment and provenance', '',
        'The original collection manifest was frozen before measurement. During collection, source '
        'metadata revealed two software attempts from the same canonical task. The first attempt '
        'in original manifest order was retained in the primary cohort; the other was kept as an '
        'auxiliary comparison. The first unused eligible canonical task in lexical order was frozen '
        'as a supplement before its measurement. This amendment occurred after original measurement '
        'had begun and used task metadata, not attention outcomes or task scores.', '',
        'The legacy 49-trajectory cohort uses a different model and different history-identity evidence; '
        'it is reported separately. See `paper_report_zh.md` for source details, controls and exclusions. '
        '`paper_design_draft_evidence.json` records the exact inputs used to produce this draft.',
    ]
    (out / 'paper_design_draft.md').write_text('\n'.join(lines) + '\n')
    evidence = dict(complete_primary_trajectories=complete, audited_calls=calls,
                    eligible_w32_trajectories=eligible, across_gt_within=larger,
                    full_primary_audit_ready=ready, human_review_required=True,
                    inputs={k: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                            for k, p in paths.items()},
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'paper_design_draft_evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({k: v for k, v in evidence.items() if k not in ['inputs', 'script_sha256']}))


if __name__ == '__main__':
    main()
