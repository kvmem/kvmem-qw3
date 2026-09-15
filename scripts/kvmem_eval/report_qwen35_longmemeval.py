#!/usr/bin/env python3
"""Export paired pilot outcomes and costs without mixing unpaired results."""
import csv
import json
import math
import statistics

from run_qwen35_longmemeval import OUT, save, sha


def mean(values):
    return statistics.mean(values) if values else None


def main():
    samples = json.loads((OUT / 'inference_samples.json').read_text())
    rows = []
    pairs = []
    for sample in samples:
        qid = sample['question_id']
        pair = []
        for policy in ('kvmem', 'compact_rag'):
            path = OUT / 'trials' / qid / policy / 'result.json'
            score = OUT / 'judge' / 'scores' / f'{qid}_{policy}.json'
            if not path.exists() or not score.exists():
                break
            result = json.loads(path.read_text())
            verdict = json.loads(score.read_text())
            assert result['status'] == 'completed'
            assert verdict['prediction_sha256'] == sha(path)
            history = (result['history_ingest_sec'] if policy == 'kvmem' else
                       result['history_index_sec'] + result['compaction_request_sec'])
            pair.append({
                'question_id': qid, 'question_type': sample['question_type'],
                'stratum': 'abstention' if qid.endswith('_abs') else sample['question_type'],
                'policy': policy, 'local_judge_correct': int(verdict['correct']),
                'history_preparation_sec': history,
                'history_kv_ingest_sec': result.get('history_ingest_sec'),
                'history_index_sec': result.get('history_index_sec'),
                'history_summary_sec': result.get('compaction_request_sec'),
                'summary_rounds': result.get('compaction_rounds', 0),
                'query_prepare_sec': result.get('final_query_prepare_sec', 0),
                'answer_request_sec': result['request_wall_sec'],
                'final_query_total_sec': result['final_query_total_sec'],
                'recorded_service_total_sec': history + result['final_query_total_sec'],
                'output_tokens_including_thinking': result['usage'].get('completion_tokens'),
                'engine_first_token_sec': result['timing'].get('engine_ttft_sec'),
                'api_first_token_sec': result['timing'].get('server_ttft_sec'),
                'finish_reason': result['finish_reason'], 'answer': result['answer'],
            })
        if len(pair) == 2:
            rows.extend(pair)
            pairs.append(pair)
    if not pairs:
        raise RuntimeError('No completed and scored pairs')
    report = OUT / 'report'
    report.mkdir(exist_ok=True)
    with (report / 'paired_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    policies = []
    for policy in ('kvmem', 'compact_rag'):
        arm = [r for r in rows if r['policy'] == policy]
        by_type = {}
        for typ in sorted({r['stratum'] for r in arm}):
            subset = [r for r in arm if r['stratum'] == typ]
            by_type[typ] = {'n': len(subset), 'correct': sum(r['local_judge_correct'] for r in subset)}
        policies.append({'policy': policy, 'n': len(arm),
            'correct': sum(r['local_judge_correct'] for r in arm),
            'accuracy': mean([r['local_judge_correct'] for r in arm]),
            'by_stratum': by_type,
            **{f'mean_{key}': mean([r[key] for r in arm]) for key in (
                'history_preparation_sec', 'final_query_total_sec', 'recorded_service_total_sec',
                'output_tokens_including_thinking', 'summary_rounds')},
            'median_final_query_total_sec': statistics.median(r['final_query_total_sec'] for r in arm)})
    wins = sum(a['local_judge_correct'] > b['local_judge_correct'] for a, b in pairs)
    losses = sum(a['local_judge_correct'] < b['local_judge_correct'] for a, b in pairs)
    discordant = wins + losses
    p = (min(1., 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2**discordant)
         if discordant else 1.)
    caveats = [
        'Fixed balanced 30-question diagnostic pilot; not the natural 500-question distribution.',
        'Local 27B judge with official prompts, not an official leaderboard score. Calibration is a sanity check, not full judge validation.',
        'Final-query time includes generation/thinking but excludes prebuilt history memory and archive.',
        'Recorded service total sums measured ingestion/index/summary/query phases; excludes server/model loading and uninstrumented client work.',
        'API/engine first token can be a thinking token and is not time to the visible final answer.',
        'This is rolling summary plus RAG, not Claude Code native auto-compaction.',
    ]
    summary = {'planned_pairs': len(samples), 'scored_pairs': len(pairs),
        'complete': len(pairs) == len(samples), 'methods': policies,
        'paired_outcomes': {'kvmem_only_correct': wins, 'compact_rag_only_correct': losses,
            'both_correct': sum(a['local_judge_correct'] and b['local_judge_correct'] for a, b in pairs),
            'neither_correct': sum(not a['local_judge_correct'] and not b['local_judge_correct'] for a, b in pairs),
            'exact_two_sided_mcnemar_p': p}, 'caveats': caveats}
    save(report / 'summary.json', summary)
    lines = ['# Qwen3.5-9B LongMemEval-S paired pilot', '',
             f'Completed and locally scored pairs: {len(pairs)}/{len(samples)}.', '',
             '| Method | Correct | History preparation (s) | Final query (s) | Recorded service total (s) |',
             '|---|---:|---:|---:|---:|']
    for row in policies:
        lines.append(f"| {row['policy']} | {row['correct']}/{row['n']} ({row['accuracy']:.1%}) | "
                     f"{row['mean_history_preparation_sec']:.2f} | {row['mean_final_query_total_sec']:.2f} | "
                     f"{row['mean_recorded_service_total_sec']:.2f} |")
    lines += ['', 'Times are arithmetic per-question means. Raw rows: `paired_results.csv`.', '',
              *[f'- {text}' for text in caveats], '']
    (report / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
