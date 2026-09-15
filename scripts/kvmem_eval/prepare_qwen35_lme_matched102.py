#!/usr/bin/env python3
"""Recover the actual historical 27B subset, not a reconstructed sampling recipe."""
from collections import Counter
import dataclasses
import json
from pathlib import Path
import shutil
import time

from dataset import load_all
from run_qwen35_longmemeval import ROOT, save, sha

SOURCE = ROOT / 'benchmark/longmemeval_qwen35_9b_20260915'
OUT = ROOT / 'benchmark/longmemeval_qwen35_9b_matched102_20260915'
ARCHIVE = ROOT / 'selected_12_samples.jsonl'
HISTORICAL = {
    'kvmem_32k': Path('/data/chaidi/kvmem_eval/results/s500_32k_bt32_meank_unified_eval_20260712_143132.jsonl'),
    'full_context': Path('/home/chaidi/kvmem_eval/KVMem_Motivation/outputs/exp1/full_context_dsv4pro_eval.jsonl'),
    'compact_rag_k22': Path('/home/chaidi/kvmem_eval/KVMem_Motivation/outputs/exp2/addTTFT/retrieval_k22/retrieval_dsv4pro_eval.jsonl'),
}


def public_sample(raw):
    data = dataclasses.asdict(raw)
    public = {k: data[k] for k in ('question_id', 'question_type', 'question', 'question_date',
                                  'haystack_dates', 'haystack_session_ids')}
    public['haystack_sessions'] = [[{k: t[k] for k in ('role', 'content')} for t in s]
                                   for s in data['haystack_sessions']]
    return public


def main():
    assert not OUT.exists() or not any(OUT.iterdir()), 'Do not overwrite a prepared experiment'
    OUT.mkdir(exist_ok=True)
    raw = load_all(ARCHIVE)
    assert len(raw) == 102 and len({r.question_id for r in raw}) == 102
    assert set(Counter(r.question_type for r in raw).values()) == {17}
    assert all(not r.question_id.endswith('_abs') for r in raw)
    official = {r.question_id: r for r in load_all(SOURCE / 'data/longmemeval_s_cleaned.json')}
    assert all(r == official[r.question_id] for r in raw), 'Data revision differs semantically'
    samples = [public_sample(r) for r in raw]
    refs = [{k: getattr(r, k) for k in ('question_id', 'question_type', 'question', 'answer')} for r in raw]
    pilot = {r['question_id']: r for r in json.loads((SOURCE / 'inference_samples.json').read_text())}
    reused = []
    for row in samples:
        qid = row['question_id']
        if qid not in pilot:
            continue
        assert row == pilot[qid]
        for policy in ('kvmem', 'compact_rag'):
            src = SOURCE / 'trials' / qid / policy
            assert json.loads((src / 'result.json').read_text())['status'] == 'completed'
            dst = OUT / 'trials' / qid / policy
            shutil.copytree(src, dst)
            assert sha(src / 'result.json') == sha(dst / 'result.json')
            reused.append({'question_id': qid, 'policy': policy, 'source': str(src),
                           'result_sha256': sha(src / 'result.json'), 'sample_payload_identical': True})
        shutil.copytree(SOURCE / 'indices' / qid, OUT / 'indices' / qid)
    plan = json.loads((SOURCE / 'plan.json').read_text())
    plan.update(created_at=time.time(), status='prepared',
        purpose='Qwen3.5-9B on the exact historical 27B 102-question subset, with 32768-token final active budgets',
        sample_description='Exact archived historical 27B subset: 102 questions, 17 per type, no abstention; not a new sample or the natural 500-question distribution.')
    plan['selection'] = {'algorithm': 'Exact archived IDs and order, verified against historical baseline outputs; no new sampling.',
        'source': str(ARCHIVE), 'source_sha256': sha(ARCHIVE), 'question_ids': [r.question_id for r in raw],
        'types': dict(Counter(r.question_type for r in raw)), 'abstention': False}
    plan['reuse'] = {'source': str(SOURCE), 'pairs': len(reused) // 2, 'new_pairs': 102 - len(reused) // 2,
        'policy': 'Only complete arms with identical entire sanitized inputs; retain original source and timings. Grade all answers together.'}
    plan['interpretation'].append('Historical 27B KVMem selection budget was 32768. Historical RAG top-22 did not cap the whole prompt: 42/102 inputs exceed 32768. Old cross-model scores also differ in reader/prompt/precision/generation/judge configuration; treat them as historical references, not a controlled model-size comparison.')
    plan['data_manifest'] = {'source': str(ARCHIVE), 'sha256': sha(ARCHIVE), 'rows': 102,
        'all_102_semantically_identical_to_cleaned_dataset': True,
        'cleaned_dataset_manifest': json.loads((SOURCE / 'data_manifest.json').read_text())}
    for name, value in [('plan.json', plan), ('inference_samples.json', samples), ('references.json', refs),
                        ('data_manifest.json', plan['data_manifest']), ('reuse_manifest.json', reused)]:
        save(OUT / name, value)
    shutil.copyfile(SOURCE / 'retriever_manifest.json', OUT / 'retriever_manifest.json')
    (OUT / 'sample_ids.txt').write_text('\n'.join(r.question_id for r in raw) + '\n')
    historical = {}
    for name, path in HISTORICAL.items():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        indexed = {r['question_id']: r for r in rows}
        assert len(indexed) == len(rows)
        ordered = [indexed[r.question_id] for r in raw]
        if name != 'kvmem_32k':
            assert [r['question_id'] for r in rows] == [r.question_id for r in raw]
        save(OUT / (name + '_historical27b.json'), ordered)
        historical[name] = {'source': str(path), 'source_sha256': sha(path), 'rows': len(ordered),
                            'same_ids_and_order_as_new_experiment': True}
    save(OUT / 'historical27b_manifest.json', historical)
    print(json.dumps({'output': str(OUT), 'questions': 102, 'reused_pairs': len(reused) // 2,
                      'new_pairs': 102 - len(reused) // 2, 'active_budget': 32768}), flush=True)


if __name__ == '__main__':
    main()
