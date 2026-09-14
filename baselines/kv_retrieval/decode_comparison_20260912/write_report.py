"""Readable report and independent panels; never form cross-model speedups."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
EXPORT=HERE/'export'


def main():
    rows=json.loads((EXPORT/'summary.json').read_text())
    by={(r['family'],r['method'],r['mtp'],r['batch_size'],r['history_tokens']):r for r in rows}
    def rate(family,method,mtp,batch,length):
        return by[family,method,mtp,batch,length]['median_tokens_per_s']
    def cell(family,method,mtp,batch,length):
        r=by[family,method,mtp,batch,length]
        return f"{r['median_tokens_per_s']:.2f} [{r['min_tokens_per_s']:.2f}, {r['max_tokens_per_s']:.2f}]"
    lines=[
        '# Native upstream decode throughput comparison', '',
        'InfLLM and InfiniGen run through the authors’ original implementations. Neither uses qw3 as its backend. '
        'All measured upstream Python files remain byte-for-byte unchanged. Qwen/KVMem is a separate deployment reference.', '',
        '## Measurements', '',
        'Values are median tokens/s across three repetitions; brackets give the observed min–max range, not a confidence interval. '
        'Batch-four throughput aggregates four sequences. All runs use the same RTX PRO 6000 Blackwell GPU exclusively.', '',
        '### InfiniGen: original OPT-6.7B / FlexGen, FP16 weights and KV', '',
        'The authors’ PG19 input, 1,920 prompt tokens, 128 generated tokens, alpha=4, partial-weight-ratio=0.2, '
        'max-num-kv=400. One GPU batch. Author weight calibration and an additional 32-output decode warmup are excluded. '
        'Measured decode is 127 forwards per sequence; prefill and model loading are separate.', '',
        '| Batch | InfiniGen | FlexGen CPU KV | FlexGen GPU KV | InfiniGen / CPU-KV speedup |',
        '|---:|---:|---:|---:|---:|',
    ]
    for batch in [1,4]:
        values=[cell('OPT',m,0,batch,1920) for m in ['InfiniGen','FlexGen CPU KV','FlexGen GPU KV']]
        speed=rate('OPT','InfiniGen',0,batch,1920)/rate('OPT','FlexGen CPU KV',0,batch,1920)
        lines.append(f'| {batch} | '+ ' | '.join(values)+f' | {speed:.2f}x |')
    lines += ['', 'CPU thread counts were tuned independently: InfiniGen=8, full CPU KV=4, GPU KV=1. '
              'The author-supported full-KV overlap option was also tested and did not improve calibration throughput. '
              'InfiniGen uses the authors’ overlap=false path, which contains its actual selective prefetch calls.', '',
              'The GPU-resident control fits on this machine. CPU offloading is an explicitly imposed cache-placement condition, '
              'not evidence that this model requires offloading on a 96-GB GPU. These results characterize the benefit within an '
              'offloading pipeline; they do not imply that selective CPU KV access is faster than keeping all KV on the GPU.', '',
              '### InfLLM: original Mistral-7B-Instruct-v0.2 / HF, BF16 weights and KV', '',
              'Official budgets: 4,096 local tokens, 128 initial tokens, 16 retrieved blocks × 128 tokens, 32 cached history blocks, '
              'repr_topk=4, execution block=512. The unchanged original Triton attention is enabled, with pinned CPU KV and '
              'the synchronous stream option selected from calibration. PyTorch uses two CPU threads. '
              'The native Triton implementation automatically falls back to its 32×32 tile on this GPU’s shared-memory limit.', '',
              '| History tokens | InfLLM | Same-model HF SDPA, GPU KV | InfLLM / GPU-KV speedup |',
              '|---:|---:|---:|---:|']
    for length in [8192,32768,122880]:
        speed=rate('Mistral','InfLLM',0,1,length)/rate('Mistral','HF SDPA GPU KV',0,1,length)
        lines.append(f'| {length:,} | {cell("Mistral","InfLLM",0,1,length)} | {cell("Mistral","HF SDPA GPU KV",0,1,length)} | {speed:.2f}x |')
    lines += ['', 'Input is a real recorded DeepSWE development history, serialized in chronological order and retokenized with '
              'Mistral’s own tokenizer. Calibration and measurement use different histories. No repeated padding or Qwen token IDs '
              'are used. All methods generate a fixed 128 outputs greedily, ignoring EOS only for this throughput microbenchmark. '
              'The first output comes from prefill; the remaining 127 forwards, including retrieval, copies, greedy sampling and '
              'history append, are timed together. Each history length receives a separate full-history decode warmup.', '',
              'The 120K dense run is a compute/capacity control beyond this checkpoint’s configured 32K context window. '
              'It is not an accuracy result or a claim that the unmodified model is reliable at 120K. '
              'The control uses the compatible HF 4.40.2 SDPA implementation; it is not claimed to be the fastest possible Mistral serving engine.', '',
              '### Separate Qwen3.8-27B / native reference, Q8_0 weights and FP8 KV', '',
              'The frozen OSWorld program is reused without source edits. Text-only inputs omit the unused vision tower. '
              'Batch=1, greedy sampling, 128 fixed output tokens, 128K KVMem selection and 128K generation budgets, block=32, '
              'CPU history/index, no NVMe, no GC. Every raw request supplies a current query span; full semantic query replay remains enabled. '
              'The MTP=0 control disables privately guided query generation, which the server requires MTP to support. '
              'No private query is needed for the explicit-query workload in either MTP setting.', '',
              '| History tokens | qw3, MTP=0 | KVMem, MTP=0 | qw3, MTP=4 | KVMem, MTP=4 |',
              '|---:|---:|---:|---:|---:|']
    for length in [8192,32768,122880,245760]:
        values=[cell('Qwen',method,mtp,1,length) for mtp,method in [(0,'qw3 GPU KV'),(0,'KVMem'),(4,'qw3 GPU KV'),(4,'KVMem')]]
        lines.append(f'| {length:,} | '+' | '.join(values)+' |')
    lines += ['', 'Native timing detail: the plain no-MTP KVMem `decode` log also includes initial post-prefill maintenance. '
              'Using it as pure decode would understate throughput. The table therefore consistently uses 127 remaining outputs '
              'divided by the interval from the first model-token callback to server completion, for all native variants. '
              'This interval excludes prefill and the initial selection/replay but retains final cache bookkeeping and response '
              'serialization. The original engine timer, server TTFT and request total are all preserved in the CSV. '
              'The native interval is not a kernel-only timer and differs slightly from the upstream internal timers.', '',
              '## What this establishes', '',
              '- Within each supported model/backend, the controls expose retrieval/offloading cost and its benefit at the measured lengths.',
              '- Across panels, model architecture, parameter count, weight/KV precision, active windows and MTP differ. '
              'Do not quote a KVMem/InfLLM/InfiniGen cross-model throughput ratio as an algorithmic speedup.',
              '- These are decode microbenchmarks. They do not establish task accuracy, full-agent latency or maintenance amortization '
              'over complete trajectories. The prior Qwen adapters and their runs remain separate diagnostics.',
              '- The sweep is bounded tuning on this hardware, not proof of a global performance optimum.', '',
              '## Reproduction and data', '',
              '- `UPSTREAM_PLAN.md`, `frozen_configuration.json`, `formal/frozen.json`: scope, settings and immutable measurement inputs.',
              '- `upstream_model_manifest.json`, `upstream_packages.txt`: pinned weights, file hashes and isolated package versions. '
              'Torch 2.11.0 / CUDA 13 is required for this current Blackwell environment; the authors’ older software/hardware timing is not reproduced literally.',
              '- `formal/`: exact commands, complete logs, token outputs, raw timers and all repetitions.',
              '- `export/runs.csv`, `export/summary.csv`, `export/decode_steps.csv`: plotting and analysis data; `export/audit.json` records integrity checks.',
              '- `upstream_tuning/`: all calibration alternatives, including slower configurations. No tuning run is silently mixed into formal results.',
              '- OSWorld pause/resume provenance: `benchmark/osworld_v2/claudecode_remaining_20260912/throughput_pause_20260912` in the repository.', '',
              'Original implementations: [InfiniGen speedup code](https://github.com/snu-comparch/InfiniGen/tree/3754f52d131c30ab7a310e3c4cf356e8e3326ac3/speedup), '
              '[InfLLM code and configurations](https://github.com/thunlp/InfLLM/tree/12b70798f56e56ebb23c53c7018091a3f540a028).', '']
    (HERE/'REPORT.md').write_text('\n'.join(lines))
    # Optional standalone scientific figure, with separate model/precision panels.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(17,4.9))
    colors=['#b7633c','#557da7','#707070']
    for i,method in enumerate(['InfiniGen','FlexGen CPU KV','FlexGen GPU KV']):
        vals=[by['OPT',method,0,b,1920] for b in [1,4]]
        means=np.array([r['median_tokens_per_s'] for r in vals])
        err=np.array([[r['median_tokens_per_s']-r['min_tokens_per_s'] for r in vals],
                      [r['max_tokens_per_s']-r['median_tokens_per_s'] for r in vals]])
        axes[0].bar(np.arange(2)+(i-1)*.23,means,.23,label=method,color=colors[i],yerr=err,capsize=3)
    axes[0].set_xticks([0,1],['Batch 1','Batch 4 (aggregate)'])
    axes[0].set_title('OPT-6.7B · FP16 / FP16 KV\n1,920-token input')
    for method,color in [('InfLLM',colors[0]),('HF SDPA GPU KV',colors[1])]:
        vals=[by['Mistral',method,0,1,l] for l in [8192,32768,122880]]
        axes[1].plot([8,32,120],[r['median_tokens_per_s'] for r in vals],'-o',label=method,color=color)
        axes[1].fill_between([8,32,120],[r['min_tokens_per_s'] for r in vals],[r['max_tokens_per_s'] for r in vals],alpha=.15,color=color)
    axes[1].set_title('Mistral-7B · BF16 / BF16 KV\nBatch 1')
    axes[1].set_xlabel('History (Ki tokens)')
    for mtp,style in [(0,'--'),(4,'-')]:
        for method,color in [('KVMem',colors[0]),('qw3 GPU KV',colors[1])]:
            vals=[by['Qwen',method,mtp,1,l] for l in [8192,32768,122880,245760]]
            axes[2].plot([8,32,120,240],[r['median_tokens_per_s'] for r in vals],style,marker='o',color=color,label=f'{method}, MTP={mtp}')
    axes[2].set_title('Qwen3.8-27B · Q8_0 / FP8 KV\nBatch 1 · separate deployment reference')
    axes[2].set_xlabel('History (Ki tokens)')
    for ax in axes:
        ax.set_ylim(bottom=0);ax.set_ylabel('Decode throughput (tokens/s)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
    fig.text(.5,.015,'Compare within each panel: models, precision and windows differ across panels. Three repeats; ranges are min–max. Native Qwen timing includes final bookkeeping.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(EXPORT/'throughput_comparison.png',dpi=200);fig.savefig(EXPORT/'throughput_comparison.pdf');plt.close(fig)
    print(HERE/'REPORT.md')


if __name__=='__main__':
    main()
