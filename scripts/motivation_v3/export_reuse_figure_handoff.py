"""Export portable plotting data plus the original saved attention measurements."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def csv_read(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('benchmark/results/motivation_v3_20260910/corrected80'))
    args = ap.parse_args()
    root = args.root
    out = root/'figure4_handoff_20260923'
    out.mkdir(parents=True, exist_ok=True)
    for folder in ['reference', 'reference_code', 'source_snapshots', 'supplementary']:
        (out/folder).mkdir(exist_ok=True)
    manifest_path = root/'manifest.json'
    analysis_path = root/'analysis_kl/analysis.json'
    locality = root/'query_locality_20260923'
    plotted = locality/'paper_revision_all_domains'
    manifest = json.loads(manifest_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    reuse = json.loads((plotted/'summary.json').read_text())
    labels = {(r['trajectory'], int(r['to_step'])): r for r in csv_read(locality/'boundary_labels.csv')}
    budgets = csv_read(locality/'boundary_budget_coverage.csv')
    fifty = {(r['trajectory'], int(r['to_step'])): r for r in budgets if r['budget'] == '50%'}
    bdomains = ['code', 'tool_state', 'dialogue', 'gui']
    trajectories = {r['id']: r for r in analysis['trajectories']}
    source_trajectories = {r['id']: r for r in manifest['trajectories']}
    domain_labels = dict(zip(bdomains, ['Software agent', 'Tool/state history', 'Dialogue', 'GUI agent']))
    a_traj, a_boundary, metadata, a_summary = [], [], [], []
    b_boundary, b_audit, b_traj, b_summary, b_all_summary = [], [], [], [], []
    for tid, t in trajectories.items():
        s = t['settings']['32']; dom = t['domain']
        a_traj.append(dict(trajectory=tid, domain=dom, domain_label=domain_labels[dom],
            boundary_count=len(s['matched_boundaries']), within_kl_bits=s['matched_within_kl'],
            across_kl_bits=s['matched_cross_kl'], difference_bits=s['matched_cross_kl']-s['matched_within_kl']))
        source = source_trajectories[tid]
        metadata.append(dict(trajectory=tid, domain=dom, independent_unit=t['independent_unit'],
            source_kind=source['source_kind'], source_path=source['source'], source_row=source.get('source_row'),
            source_line=source.get('source_line'), collected_calls=t['collected_steps'],
            first_source_step=source['frames'][0]['source_step'], last_source_step=source['frames'][-1]['source_step'],
            mean_prompt_tokens=t['mean_prompt_tokens'], mean_decode_samples=t['mean_decode_samples'],
            replay=True, closed_loop=False))
        for boundary in s['matched_boundaries']:
            key = tid, boundary['to_step']; label = labels[key]
            step_to_index = {f['source_step']: i for i, f in enumerate(source['frames'])}
            identity = dict(trajectory=tid, domain=dom, from_step=boundary['from_step'], to_step=boundary['to_step'],
                from_call_index=step_to_index[boundary['from_step']], to_call_index=step_to_index[boundary['to_step']],
                boundary_type=label['boundary_type'], session_transition=label['session_transition'])
            a_boundary.append(identity | dict(window_positions=32, common_blocks=boundary['common_blocks'],
                within_kl_bits=boundary['within_kl'], across_kl_bits=boundary['cross_kl'],
                difference_bits=boundary['cross_kl']-boundary['within_kl']))
            r = fifty[key]
            reasons = []
            if dom == 'dialogue' and r['boundary_type'] != 'across_user_query':
                reasons.append('dialogue_without_new_user_query')
            if r['source_saturated'] == 'True':
                reasons.append('source_saturated_after_rounding')
            selected = int(r['strict_selected_blocks']); surviving = int(r['strict_surviving_blocks'])
            common = int(r['common_blocks']); previous = float(r['strict_coverage'])
            record = identity | dict(budget_fraction=.5, source_candidate_blocks=int(r['source_candidate_blocks']),
                selected_blocks=selected, surviving_blocks=surviving, common_blocks=common,
                selected_identity_survival=float(r['strict_surviving_fraction']), uniform=surviving/common,
                previous=previous, oracle=previous+float(r['strict_matched_oracle_gain']),
                common_history_fully_selected=surviving == common,
                eligible=not reasons, exclusion_reason=';'.join(reasons))
            b_audit.append(record)
            if not reasons:
                b_boundary.append(record)
    for dom in bdomains+['all']:
        s = analysis['summary'][dom]['32'] if dom != 'all' else analysis['pooled_kl']
        rs = a_boundary if dom == 'all' else [r for r in a_boundary if r['domain'] == dom]
        row = dict(domain=dom, trajectories=s['paired_rollouts'], boundaries=len(rs),
                   ratio_of_means=s['ratio_of_means'], across_gt_within=s['cross_gt_within'])
        for key, prefix in [('matched_within_kl', 'within'), ('matched_cross_kl', 'across'), ('paired_difference', 'difference')]:
            v = s[key]
            row.update({f'{prefix}_mean_bits': v['mean'], f'{prefix}_ci95_low_bits': v['ci95'][0],
                        f'{prefix}_ci95_high_bits': v['ci95'][1]})
        a_summary.append(row)
    for budget, groups in reuse.items():
        for dom, group in groups.items():
            s = group['primary']
            for method in ['uniform', 'previous', 'oracle']:
                v = s[method]
                row = dict(budget=budget, domain=dom, method=method, trajectories=s['trajectories'],
                           boundaries=s['boundaries'], mean=v['mean'], ci95_low=v['ci95'][0], ci95_high=v['ci95'][1])
                b_all_summary.append(row)
                if budget == '50%': b_summary.append(row)
            if budget == '50%':
                for values in s['trajectory_values']:
                    boundary_count = sum(r['trajectory'] == values['trajectory'] for r in b_boundary)
                    b_traj.append(dict(domain=dom, boundary_count=boundary_count, **values))
    for name, rows in [('panel_a_trajectory.csv', a_traj), ('panel_a_boundary.csv', a_boundary),
            ('panel_a_summary.csv', a_summary), ('panel_b_trajectory.csv', b_traj),
            ('panel_b_boundary.csv', b_boundary), ('panel_b_summary.csv', b_summary),
            ('panel_b_boundary_audit.csv', b_audit), ('trajectory_metadata.csv', metadata),
            ('supplementary/panel_b_summary_all_budgets.csv', b_all_summary)]:
        csv_write(out/name, rows)
    for src, target in [
        (plotted/'plot_data.csv', 'supplementary/panel_b_trajectory_all_budgets.csv'),
        (locality/'boundary_budget_coverage.csv', 'supplementary/source_boundary_budget_coverage.csv'),
        (locality/'boundary_labels.csv', 'source_snapshots/boundary_labels.csv'),
        (plotted/'summary.json', 'source_snapshots/panel_b_summary_original.json'),
        (manifest_path, 'source_snapshots/manifest_original.json'),
        (plotted/'provenance.json', 'source_snapshots/figure_provenance_original.json'),
        (plotted/'validation.json', 'source_snapshots/figure_validation_original.json'),
        (plotted/'figure4_caption.txt', 'figure_caption.txt'),
        (plotted/'figure4_caption.tex', 'figure_caption.tex')]:
        shutil.copyfile(src, out/target)
    a_snapshot = dict(model=analysis['model'], pooled_kl=analysis['pooled_kl'],
        summary={d: analysis['summary'][d]['32'] for d in bdomains},
        trajectories=[dict(id=t['id'], domain=t['domain'],
            matched_boundaries=t['settings']['32']['matched_boundaries'],
            matched_within_kl=t['settings']['32']['matched_within_kl'],
            matched_cross_kl=t['settings']['32']['matched_cross_kl']) for t in analysis['trajectories']],
        source_analysis_sha256=sha(analysis_path), kl_derivation=analysis['kl_derivation'])
    dump(out/'source_snapshots/panel_a_w32_original.json', a_snapshot)
    for suffix in ['png', 'pdf', 'svg']:
        shutil.copyfile(plotted/f'figure4_all_domains.{suffix}', out/f'reference/figure4_current.{suffix}')
    script_dir = Path(__file__).parent
    for name in ['metrics.py', 'trace_io.py', 'collect.py', 'prepare_dataset.py', 'analyze_query_locality.py',
                 'plot_query_locality_evidence.py', 'plot_paper_reweighting_reuse.py',
                 'plot_paper_reweighting_reuse_all_domains.py', Path(__file__).name]:
        shutil.copyfile(script_dir/name, out/'reference_code'/name)
    # Full saved query-position x historical-block matrices, no further reduction.
    raw_zip = root/'figure4_raw_attention_20260923.zip'
    index = []
    raw_prefix = 'figure4_raw_attention_20260923/'
    with zipfile.ZipFile(raw_zip, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.write(manifest_path, raw_prefix+'manifest_original.json')
        for t in manifest['trajectories']:
            for i, frame in enumerate(t['frames']):
                metadata_path = root/'measurements'/t['id']/f'step_{i:03d}.json'
                record = json.loads(metadata_path.read_text())
                entry = dict(trajectory=t['id'], domain=t['domain'], call_index=i, source_step=frame['source_step'],
                    prompt_tokens=record['prompt_tokens'], completion_tokens=record['completion_tokens'],
                    prefill_samples=record['prefill_samples'], decode_samples=record['decode_samples'],
                    block_tokens=record['block_tokens'], layers_json=json.dumps(record['layers']),
                    common_prefix_tokens=record.get('common_prefix_tokens'),
                    prefix_identity_verified=record['prefix_identity_verified'],
                    image_sha256_json=json.dumps(frame.get('image_sha256', [])))
                for kind, src, filename in [('attention', Path(record['path']), f'attention_{i:03d}.npz'),
                        ('tokens', Path(record['input_tokens_path']), f'input_tokens_{i:03d}.json.gz'),
                        ('metadata', metadata_path, f'step_{i:03d}.json')]:
                    relative = f'measurements/{t["id"]}/{filename}'
                    archive.write(src, raw_prefix+relative)
                    entry.update({f'{kind}_relative_path': relative, f'{kind}_source_path': str(src.resolve()),
                                  f'{kind}_sha256': sha(src)})
                index.append(entry)
            print('raw archived:', t['id'], flush=True)
        csv_write(out/'raw_measurement_index.csv', index)
        archive.write(out/'raw_measurement_index.csv', raw_prefix+'raw_measurement_index.csv')
        archive.write(locality/'boundary_labels.csv', raw_prefix+'boundary_labels.csv')
        archive.writestr(raw_prefix+'README.txt',
            'These are the unchanged saved attention NPZ, input-token GZ, and step metadata for all 957 calls.\n'
            'NPZ mass is query-position x 128-token historical block, normalized per layer then averaged over 16 layers.\n'
            'This is not raw Q/K or per-head token-to-token attention. Use raw_measurement_index.csv relative paths;\n'
            'original step JSON absolute paths are provenance and need not exist on another machine.\n'
            'Full definitions, plotting CSVs, and verification code are in the companion figure4_handoff_20260923.zip.\n')
    # Validate exported aggregation before presenting the package.
    errors = []
    for row in a_traj:
        rs = [r for r in a_boundary if r['trajectory'] == row['trajectory']]
        for key in ['within_kl_bits', 'across_kl_bits']:
            errors.append(abs(float(np.mean([r[key] for r in rs]))-row[key]))
    for row in b_traj:
        rs = [r for r in b_boundary if r['trajectory'] == row['trajectory']]
        for key in ['uniform', 'previous', 'oracle']:
            errors.append(abs(float(np.mean([r[key] for r in rs]))-row[key]))
    assert max(errors) < 1e-12
    assert len(a_traj) == len(b_traj) == 80 and len(a_boundary) == 791 and len(b_boundary) == 781
    assert len(index) == 957
    config = dict(version='2026-09-23-four-domain-50pct',
        domain_order=bdomains, labels=domain_labels, main_window_positions=32, block_tokens=128,
        main_selection_fraction=.5, coverage_storage_unit='fraction_0_to_1', coverage_plot_multiplier=100,
        kl_unit='bits', bootstrap_replicates=5000, bootstrap_seed=20260911,
        panel_a_ci_source='original saved CI; retain original trajectory order if recomputing',
        panel_b_ci_order='lexicographic trajectory ID',
        original_reference_figure='reference/figure4_current.png',
        raw_archive=raw_zip.name, raw_archive_prefix=raw_prefix,
        methods=['uniform', 'previous', 'oracle'],
        expected_counts=dict(trajectories_per_domain=20, raw_calls=957, panel_a_boundaries=791, panel_b_boundaries=781))
    dump(out/'figure_config.json', config)
    dump(out/'export_validation.json', dict(maximum_aggregation_error=max(errors),
        panel_a_trajectory_rows=len(a_traj), panel_a_boundary_rows=len(a_boundary),
        panel_b_trajectory_rows=len(b_traj), panel_b_boundary_rows=len(b_boundary), raw_call_count=len(index),
        exclusion_reasons=dict(Counter(r['exclusion_reason'] for r in b_audit if not r['eligible']))))
    provenance = dict(source_files={str(p.resolve()): sha(p) for p in [analysis_path, manifest_path,
        locality/'boundary_budget_coverage.csv', plotted/'summary.json', Path(__file__)]},
        raw_archive=dict(path=str(raw_zip.resolve()), bytes=raw_zip.stat().st_size, sha256=sha(raw_zip)),
        environment={p: importlib.metadata.version(p) for p in ['numpy', 'matplotlib']},
        new_model_calls=0, export_is_lossless_for_saved_npz_and_gz=True)
    dump(out/'provenance.json', provenance)
    (out/'requirements.txt').write_text('numpy\nmatplotlib\n')
    print('Data directory:', out.resolve(), flush=True)
    print('Raw archive MiB:', raw_zip.stat().st_size/1024**2, flush=True)


if __name__ == '__main__':
    main()
