"""Publication-exportable data and figures; no cross-model speedup ratios."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
LABELS={'infllm_small':'InfLLM · small window', 'infinigen_small':'InfiniGen · cap 2K',
        'hf_dense_gpu':'HF full GPU KV', 'hf_dense_cpu':'HF full CPU KV',
        'qw3_bf16':'qw3 · full GPU KV', 'kvmem_bf16':'KVMem · 128K + 128K',
        'infllm_128k_fastmap':'InfLLM · 128K budget, exact CPU map optimization',
        'infllm_128k':'InfLLM · 128K budget','infinigen_128k':'InfiniGen · cap 128K',
        'infllm_128k_block2048':'InfLLM · 128K budget, block 2048',
        'kvmem_128k_32out':'KVMem · 128K budget'}


def table(rows):
    lengths=sorted({r['history_tokens'] for r in rows})
    text=['| Method | '+ ' | '.join(f'{n//1024}K' for n in lengths)+' |',
          '|---|'+'---:|'*len(lengths)]
    for method in dict.fromkeys(r['method'] for r in rows):
        cells=[]
        for length in lengths:
            found=[r for r in rows if r['method']==method and r['history_tokens']==length]
            if not found:cells.append('—');continue
            r=found[0]
            cells.append(f"{r['median_tokens_per_s']:.2f} (n=1)" if r['n']==1 else
                         f"{r['median_tokens_per_s']:.2f} [{r['min_tokens_per_s']:.2f}, {r['max_tokens_per_s']:.2f}]")
        text.append('| '+LABELS.get(method,method)+' | '+' | '.join(cells)+' |')
    return text


def main():
    main_rows=json.loads((HERE/'export/summary.json').read_text())
    matched_path=HERE/'matched_budget_export/summary.json'
    matched=json.loads(matched_path.read_text()) if matched_path.exists() else []
    lines=['# Qwen3.8-27B decode throughput comparison','',
           'All primary rows use the exact same original local BF16 safetensors, FP8 E4M3FN KV storage, batch=1, no MTP, no NVMe, exclusive RTX PRO 6000 Blackwell GPU, and CPU affinity on the GPU-local NUMA node. Model computation for external methods is HF/FLA/PyTorch; no qw3 projection library is loaded.','',
           '## Fixed 128-output measurement','',
           'Rates are median [minimum, maximum] tokens/s over three repetitions. The first output belongs to prefill; the timed interval covers the remaining 127 output tokens. Inputs are identical real DeepSWE history prefixes at each length, with the same current-query suffix. These are decode microbenchmarks, not task-success results.','',
           *table(main_rows),'',
           '## Attention budgets','',
           'InfLLM small uses 4096 recent + 128 initial + 16×128 retrieved tokens (~6.3K); its resident historical cache holds 32 blocks. InfiniGen small uses alpha=4, a 20% partial subspace and a 2048-token cap per query head; actual selected counts and H2D bytes are exported. KVMem keeps its 128K selection and 128K generation budgets, block=32, CPU history/index, and full explicit-query replay. Thus the small-window table matches model/precision/workload but deliberately preserves the authors’ small working sets. It does not establish equal task quality.','']
    if matched:
        lines+=['## Bounded equal-budget check','',
                'All methods have a 128K retrieval/selection budget on a 240K input. The exact per-method policy remains unchanged. InfiniGen’s cap is an upper bound, not a fixed selected count. Fixed 32 outputs (31 timed decode forwards); these results are kept separate from the 128-output table.','',*table(matched),'']
        lines+=['InfLLM retains the author configuration’s 2x cache-to-retrieval-block ratio (991 retrieved / 1982 cached at block=128). A separate block=2048 configuration uses 62 retrieved / 124 cached blocks and 3968 recent + 128 initial tokens, for the same total 131072-token attention budget. Changing block granularity can affect selection/accuracy; both measurements are reported, not treated as numerically identical.','']
    validation_path=HERE/'block_map_model_validation.json'
    if validation_path.exists():
        v=json.loads(validation_path.read_text())
        lines += [f"The exact CPU block-map optimization replaces nested list membership with sets and an ascending free-slot iterator. It preserves selected IDs, placement and copy order in the CPU tests, and all {v['generated_tokens']} generated IDs match the original in the real-model check. At unchanged block=128, throughput changes from {v['original_tokens_per_s']:.3f} to {v['optimized_tokens_per_s']:.3f} tokens/s ({v['decode_speedup']:.2f}x in this single check); prefill changes from {v['original_prefill_s']:.1f} to {v['optimized_prefill_s']:.1f} seconds. This is an isolated adapter optimization; the upstream checkout is unchanged. See block_map_model_validation.json.", '']
    lines+=['## Interpretation and limits','',
            '- InfLLM and InfiniGen are Qwen architecture ports of the author methods. Their original end-to-end code natively targets Mistral/OPT, so those unchanged implementations were measured first in `../native_upstream_export/REPORT.md`. Do not call the Qwen port an unmodified upstream reproduction.',
            '- Same-model system throughput can be compared, but the native engine and HF have different operator/launch paths and residual precision (native FP32 versus HF BF16). The HF full-GPU control exposes common execution overhead; the full-CPU control exposes the benefit of selective offloading within the HF pipeline.',
            '- Historical KV storage is FP8 in all primary arms. InfLLM quantizes unrotated historical keys to preserve its position-remapping method; native/InfiniGen may quantize at a different positional boundary. InfLLM recent working views retain exact FP8-rounded values in FP16. These numerical/storage details must accompany accuracy claims.',
            '- On this GQA model the per-query-head InfiniGen FP16 partial-key index is 1.195 times the full shared-head FP8 K+V size (8.965 vs 7.5 GiB at 240K). Thus this port does not establish a GPU-memory advantage over full FP8 KV; quantizing or restructuring its index would be a separate variant. See METHOD_AUDIT.md and infinigen_index_footprint.json.',
            '- HF timing includes all per-token retrieval, CPU gathering, transfers, sampling and state updates. Native timing is first model token to server completion, which also includes final cache bookkeeping/serialization. Raw native decode counters are retained but are not used blindly: plain no-MTP KVMem counters can include initial selection/replay.',
            '- Loading, SVD calibration, prefill, initial offload and initial KVMem query selection/replay are outside decode throughput and reported separately. This table does not measure Claude Code summary stalls or private guided-query generation.',
            '- Important rollback observation: repeating an identical 240K raw completion prompt retracts the previous generated suffix. KVMem hit no_reusable_checkpoint twice, with 109.575 and 108.352 seconds of cold prefill. Those original prefill and request-wall measurements are retained in runs.csv and rollback_checkpoint_observation.json. The decode table cannot be read as an end-to-end request-latency claim, nor does this establish the frequency of that case in append-only agent traffic.',
            '- The 240K history fits the tested full-GPU controls. CPU-offload controls intentionally force CPU history; this is not evidence that full GPU KV is infeasible on this hardware.',
            '- All trials, tuning and validity checks are preserved. NUMA affinity was selected after unbound upstream InfLLM runs showed variability; the original slow measurements remain in the auxiliary export.','',
            '## Provenance and validation','',
            'The native BF16 compatibility directory adds only dense-format loader metadata and symlinks the original safetensors. No weight values or production engine code are changed. See `weight_identity_audit.json`, `native_bf16_container_manifest.json`, `formal_jobs.json`, `bf16_calibration_validation.json` and the raw per-arm reports. InfiniGen full-selection logits and CPU KV match the dense control exactly across eight real-model decode steps; graph checks verify both outputs and recurrent state.','',
            'Sources: [InfLLM](https://github.com/thunlp/InfLLM/tree/12b70798f56e56ebb23c53c7018091a3f540a028), [InfiniGen](https://github.com/snu-comparch/InfiniGen/tree/3754f52d131c30ab7a310e3c4cf356e8e3326ac3).','',
            'Data: `export/runs.csv`, `export/summary.csv`, `export/decode_steps.csv`; bounded budget check: `matched_budget_export/`.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5.4))
    order=['qw3_bf16','kvmem_bf16','hf_dense_gpu','infllm_small','infinigen_small','hf_dense_cpu']
    for method in order:
        rows=sorted((r for r in main_rows if r['method']==method),key=lambda r:r['history_tokens'])
        if not rows:continue
        x=[r['history_tokens']/1024 for r in rows];y=[r['median_tokens_per_s'] for r in rows]
        ax.plot(x,y,'o-',label=LABELS[method])
        ax.fill_between(x,[r['min_tokens_per_s'] for r in rows],[r['max_tokens_per_s'] for r in rows],alpha=.1)
    ax.set(xlabel='Input history (Ki tokens)',ylabel='Decode throughput (tokens/s)',ylim=(0,None),
           title='Qwen3.8-27B · BF16 weights / FP8 KV · batch 1 · MTP off')
    ax.set_xticks([8,32,120,240]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.text(.5,.01,'Median and min–max over 3 repeats. Small retrieval windows differ; system backends differ.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(HERE/'decode_throughput.png',dpi=200);fig.savefig(HERE/'decode_throughput.pdf');plt.close(fig)


if __name__=='__main__':main()
