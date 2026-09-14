"""Raw step export plus task-level window comparisons; no mixed-load speedups."""
import argparse
import csv
import json
from pathlib import Path
from summarize_pilot import summarize, jsonl, stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args()
    variants=list(json.loads((a.campaign/'plan.json').read_text())['variants'])
    out=a.campaign/'analysis';out.mkdir(exist_ok=True)
    summaries,steps=[],[]
    for path in sorted(a.campaign.glob('*/*/attempt_*/result.json')):
        run=path.parent
        summary,rows=summarize(run)
        variant=run.parent.name
        if variant not in variants:raise ValueError('attempt is outside selected campaign scope: '+str(run))
        summary.update(variant=variant,active_window_capacity=196608 if variant=='large' else 6272,
                       seed=1000,weight_dtype='Q8_0',kv_dtype='FP8 E4M3FN')
        phases=jsonl(run/'server/requests.jsonl')
        summary['max_prompt_tokens']=max((r.get('prompt_tokens',0) for r in phases),default=0)
        summary['decode_seconds_per_output_token']=stats([r['decode_s']/r['output_tokens']
            for r in phases if r.get('output_tokens',0)>1 and r.get('decode_s') is not None])
        memory=jsonl(run/'host_memory_samples.jsonl')
        summary['peak_model_tree_pss_gib']=max((sum(p['pss_kib'] for p in r.get('model_processes',[]))/2**20 for r in memory),default=None)
        summaries.append(summary)
        for row in rows:
            steps.append({'task_id':summary['task_id'],'variant':variant,
                **{k:v for k,v in row.items() if not isinstance(v,(dict,list))}})
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    if steps:
        with (out/'steps.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in steps))))
            w.writeheader();w.writerows(steps)
    (out/'README.md').write_text('Recorded original DeepSWE trials; inspect each completion status. Selected windows: '+', '.join(variants)+'. Large=196608 total active tokens; small=6272. '
        'Task outcomes are independent live trajectories. Shared-GPU latencies are diagnostic and are not formal speedup evidence. '
        'Historical native arms use MTP and FP32 residuals; InfLLM uses the same Q8_0 weight blocks with HF/FLA BF16 intermediates, without MTP.\n')


if __name__=='__main__':main()
