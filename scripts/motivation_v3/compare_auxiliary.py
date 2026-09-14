"""Describe the retained same-task attempts without treating them as independent tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from metrics import analyze_steps, mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign
    primary = json.loads((root / 'manifest.json').read_text())
    auxiliary = json.loads((root / 'auxiliary_manifest.json').read_text())
    counterpart = auxiliary['primary_counterpart_id']
    primary_row = next(r for r in primary['trajectories'] if r['id'] == counterpart)
    rows = [(primary_row, root / 'measurements', 'primary')]
    rows += [(r, root / 'auxiliary_measurements', 'auxiliary')
             for r in auxiliary['trajectories']]
    assert len({r['independent_unit'] for r, _, _ in rows}) == 1
    results = []
    for row, measurements, role in rows:
        steps = []
        for i in range(len(row['frames'])):
            record = json.loads((measurements / row['id'] / f'step_{i:03d}.json').read_text())
            assert record['status'] == 'collected'
            with np.load(record['path']) as data:
                steps.append(record | dict(step=record['source_step'],
                                           mass=data['mass'], positions=data['positions']))
        metrics = {str(w): analyze_steps(steps, w, identity_verified=True,
                                        compute_coverage=(w == 32))
                   for w in [16, 32, 64, 128]}
        coverage = [s['prefill_diagnostic']['r10'] for s in metrics['32']['details']
                    if next(r for r in steps if r['step'] == s['step'])['prefix_identity_verified']
                    and 'r10' in s.get('prefill_diagnostic', {})]
        results.append(dict(id=row['id'], role=role, task=row['independent_unit'],
                            calls=len(steps), mean_prompt_tokens=mean([s['prompt_tokens'] for s in steps]),
                            mean_decode_queries=mean([s['decode_samples'] for s in steps]),
                            windows={w:dict(within=m['matched_within_js'], across=m['matched_cross_js'],
                                            matched_boundaries=len(m['matched_boundaries']))
                                     for w, m in metrics.items()},
                            r10_eligible_calls=len(coverage),
                            r10_fixed_coverage=mean([c['mean'] for c in coverage])))
    # Independently recomputed retained-attempt values must agree with the primary analysis.
    analysis = json.loads((root / 'analysis/analysis.json').read_text())
    retained = next(r for r in analysis['trajectories'] if r['id'] == counterpart)
    for w, values in results[0]['windows'].items():
        for key in ['within', 'across']:
            expected = retained['settings'][w]['matched_within_js' if key == 'within'
                                               else 'matched_cross_js']
            actual = values[key]
            assert actual == expected or (actual is not None and expected is not None
                                           and abs(actual - expected) < 1e-12)
    output = dict(scope='Descriptive comparison of two recorded attempts at one canonical task. '
                  'Only the first manifest occurrence enters primary estimates; no independent-task '
                  'confidence interval is inferred from these two attempts.',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=results)
    out = root / 'analysis'
    (out / 'auxiliary_comparison.json').write_text(json.dumps(output, indent=2) + '\n')
    lines = ['# Same-task auxiliary comparison', '', output['scope'], '',
             '| Attempt | Role | Calls | W32 boundaries | Within JS | Across JS | Fixed 10% coverage |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        w = r['windows']['32']
        lines.append(f"| {r['id']} | {r['role']} | {r['calls']} | {w['matched_boundaries']} | "
                     f"{w['within']:.4f} | {w['across']:.4f} | {100*r['r10_fixed_coverage']:.1f}% |")
    lines += ['', 'The attempts have different recorded histories. This is not an identical-input '
              'repeatability experiment or a task-success comparison. The nominal 10% block budget '
              'is rounded up; coverage is conditional on verified candidate historical attention.']
    (out / 'auxiliary_comparison.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'canonical_tasks': 1, 'attempts': len(results), 'primary_included': 1}))


if __name__ == '__main__':
    main()
