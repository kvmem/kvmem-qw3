"""Export raw measurements and explicitly scoped native deployment references."""
from common import HERE, CAMPAIGN, BASE, WEIGHTS
from collections import defaultdict
import csv
import hashlib
import json
import statistics
import subprocess


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    out=HERE/'export';out.mkdir(exist_ok=True)
    rows=[];groups=defaultdict(list);mtp_mappings={}
    native=list(csv.DictReader((CAMPAIGN/'q8_mtp_20260913/export/runs.csv').open()))
    input_hashes={int(r['history_tokens']):r['input_ids_sha256'] for r in native}
    for method in ('infllm','infinigen'):
        path=HERE/'formal'/f'{method}.json';report=json.loads(path.read_text())
        assert report['status']=='complete'
        assert len(report['results'])==24
        assert report['mtp_mapping']
        mtp_mappings[method]=report['mtp_mapping']
        pairs=defaultdict(dict)
        for result in report['results']:
            pairs[result['prompt_tokens'],result['rep']][result['mtp']]=result
            assert result['input_ids_sha256']==input_hashes[result['prompt_tokens']]
            batches=result['batches']
            drafted=sum(b.get('drafted',0) for b in batches)
            accepted=sum(b.get('accepted',0) for b in batches)
            row=dict(method=method,prompt_tokens=result['prompt_tokens'],mtp=result['mtp'],rep=result['rep'],
                outputs=result['outputs'],timed_tokens=result['outputs']-1,
                decode_s=result['decode_s'],decode_tokens_per_s=result['decode_tokens_per_s'],
                drafted=drafted,accepted=accepted,acceptance=accepted/drafted if drafted else None,
                draft_s=sum(b.get('draft_s',0) for b in batches),
                verify_s=sum(b.get('verify_s',0) for b in batches),
                commit_prime_s=sum(b.get('commit_prime_s',0) for b in batches),
                prefill_and_prime_s=result['prefill_and_prime_s'],graph_setup_s=result['graph_setup_s'],
                peak_allocated_gib=result['peak_allocated_gib'],input_ids_sha256=result['input_ids_sha256'],
                output_ids_sha256=hashlib.sha256(json.dumps(result['output_ids']).encode()).hexdigest(),
                source=str(path))
            rows.append(row);groups[method,result['prompt_tokens'],result['mtp']].append(row)
        for pair in pairs.values():
            assert pair[0]['output_ids']==pair[4]['output_ids']
    summary=[]
    for (method,length,mtp),items in sorted(groups.items()):
        speeds=[r['decode_tokens_per_s'] for r in items]
        drafted=sum(r['drafted'] for r in items)
        summary.append(dict(method=method,prompt_tokens=length,mtp=mtp,n=len(items),
            median_tokens_per_s=statistics.median(speeds),min_tokens_per_s=min(speeds),max_tokens_per_s=max(speeds),
            pooled_tokens_per_s=sum(r['timed_tokens'] for r in items)/sum(r['decode_s'] for r in items),
            acceptance=sum(r['accepted'] for r in items)/drafted if drafted else None))
    write_csv(out/'runs.csv',rows);write_csv(out/'summary.csv',summary)
    write_csv(out/'native_reference.csv',native)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    table=['| Input | InfLLM MTP off | InfLLM MTP=4 | ratio | InfiniGen MTP off | InfiniGen MTP=4 | ratio |',
           '|---|---:|---:|---:|---:|---:|---:|']
    lookup={(r['method'],r['prompt_tokens'],r['mtp']):r['median_tokens_per_s'] for r in summary}
    for length in (8192,32768,122880,245760):
        values=[lookup[m,length,k] for m in ('infllm','infinigen') for k in (0,4)]
        a,b,c,d=values
        table.append(f'| {length/1024:g}K | {a:.2f} | {b:.2f} | {b/a:.2f}x | {c:.2f} | {d:.2f} | {d/c:.2f}x |')
    native_summary=list(csv.DictReader((CAMPAIGN/'q8_mtp_20260913/export/summary.csv').open()))
    native_lookup={(r['method'],int(r['history_tokens'])):float(r['median_tokens_per_s']) for r in native_summary}
    native_table=['| Input | Native qw3 Q8 MTP=4 | Native KVMem Q8 MTP=4 |',
                  '|---|---:|---:|']
    for length in (8192,32768,122880,245760):
        native_table.append(f"| {length/1024:g}K | {native_lookup['qw3_q8_mtp4',length]:.2f} | {native_lookup['kvmem_q8_mtp4',length]:.2f} |")
    report='''# External Qwen Q8 MTP throughput

Medians of three repetitions, token/s. One RTX PRO 6000 Blackwell, exclusive
GPU, CPU affinity 0-15,32-47, two Torch CPU threads. Exact Qwen3.8-27B Q8_0
GGUF; FP8 historical KV; greedy generation, 128 outputs (127 timed outputs).
The complete decode interval includes draft, every speculative retrieval,
transfer, rejected verification, sampling, state restoration and MTP priming.
Model loading, prompt prefill, calibration and graph setup are excluded.

'''+ '\n'.join(table)+ '\n\nPreviously measured native deployment references (token/s):\n\n'+ '\n'.join(native_table)+'''

## Correctness and method provenance

InfLLM uses the pinned author's ContextManager/representatives/retrieval,
local+global attention and LRU. InfiniGen uses its author-derived threshold and
per-head selector, CPU historical KV and cross-layer query forecasts. Both
have independent Qwen hybrid-model adapters. They do not natively provide this
Qwen MTP integration; these results are adapted implementations, not untouched
upstream end-to-end systems.

The actual GGUF block-64 predictor shares the main embedding/output head.
Its prefix pairs token t with target pre-final-norm hidden t-1. It drafts four
tokens; the target verifies and commits only the accepted prefix plus its own
correction/bonus. Each token performs its own causal retrieval. InfiniGen
pipelines the individual forecasts after staging candidate K/V, with selector
i restricted to index[:base+i]. Rejected candidates never extend committed
history. InfLLM also restores local scores, block representatives, LRU state
and physical attention block ordering.

The compute boundary uses standalone Q8 CUDA arithmetic from a frozen qw3
source copy. Only the isolated copy changes: four-warps across verify widths,
explicit rounded accumulation/products, and per-token RMSNorm reductions avoid
batch-dependent rounding changes. The production qw3 source/binary and authors'
repositories remain unchanged. Recurrent and MLP CUDA graphs retain every
prefix state; retrieval and transfers execute outside graphs.

Validation: independent serial prefills, five-token verification crossing a
128-token block boundary, commits of 1/3/5 tokens, serial continuation,
logits/recurrent state and retrieval-state fingerprints all match exactly.
Compute graphs separately check outputs and every stored recurrent prefix.
All 24 formal MTP-on/off pairs generate exactly identical 128-token sequences.
This is implementation consistency evidence, not an agent accuracy evaluation.

## Comparison boundaries

InfLLM retains its small author-style working set: 128 initial + 4096 local +
16 x 128 retrieved tokens; recurrent layers preserve their recurrent states.
InfiniGen has a 2048-token per-query-head selection cap, alpha=4, partial ratio
0.2; it does not necessarily select exactly 2048 tokens. Both MTP drafters use
full GPU FP8 history at true positions, bounded by the validated 262144-position
range. Longer histories require an additional draft-context policy; they are
not claimed supported by this harness.

The previous native Q8/MTP=4 KVMem medians were 111.66/93.18/89.77/72.59 token/s
at 8K/32K/120K/240K. Its selection budget is 128K, its MTP follows the working
set, and native residual arithmetic is FP32 versus the adapters' HF/FLA BF16.
Input IDs, Q8 weight file, FP8 storage, hardware and timed token count align;
working set, kernels and residual arithmetic differ. Treat native numbers as
deployment references, not an isolated retrieval-algorithm comparison or an
accuracy equivalence claim. Raw native results remain in
`../decode_comparison_20260912/q8_mtp_20260913/export/`.

Early failed numerical checks and development probes are retained separately.
The first formal campaign stopped at 32K because upstream load(None) referenced
an uninitialized target_event when restoring an evicted block. Its complete
logs/results are retained in formal_failure_cache_reload_20260913. The adapter
now passes an explicit temporary destination, then reconstructs the saved buffer
order. A 32K eviction/rollback regression check passed. The entire formal matrix
was restarted after this fix; results were not filtered by speed or acceptance.
OSWorld pause/resume evidence is recorded in `resume_status.json` and the suite's
`throughput_pause_external_mtp_20260913/` directory.
'''
    (HERE/'REPORT.md').write_text(report)
    identity=json.loads((CAMPAIGN/'q8_mtp_20260913/weight_identity.json').read_text())
    stat=WEIGHTS.stat()
    assert identity['bytes']==stat.st_size and identity['mtime_ns']==stat.st_mtime_ns
    calibration=HERE/'q8_rotations.pt'
    calibration_source=BASE/'infinigen/InfiniGen/accuracy/setup/pg19_firstbook.txt'
    frozen=json.loads((HERE/'formal/status.json').read_text())['source_sha256']
    inference_files=['common.py','run.py','draft.py','verify.py','transaction.py','graphs.py',
                     'checks.py','infinigen_pipeline.py']
    for name in inference_files:
        path=HERE/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==frozen[str(path)],name
    upstream={}
    for name in ('InfLLM','InfiniGen'):
        path=BASE/'upstream'/name
        upstream[name]=dict(commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),
            tracked_changes=subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True))
        assert not upstream[name]['tracked_changes']
    (out/'audit.json').write_text(json.dumps(dict(formal_measurements=len(rows),paired_sequences_equal=24,
        native_input_hashes_match=True,early_failures_retained=True,
        weight_identity=identity,weight_stat_still_matches_verified_file=True,
        calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        calibration_source_sha256=hashlib.sha256(calibration_source.read_bytes()).hexdigest(),
        mtp_weight_mappings=mtp_mappings,
        inference_files_match_frozen_snapshot=True,upstream=upstream,
        actual_drafted_tokens={m:sum(r['drafted'] for r in rows if r['method']==m) for m in mtp_mappings},
        loader_manifest_scope='raw model.mtp_loaded=false describes the main-only GGUF loader; the separately attached Predictor and its mapping are reported as mtp_mapping'),indent=2)+'\n')


if __name__=='__main__':main()
