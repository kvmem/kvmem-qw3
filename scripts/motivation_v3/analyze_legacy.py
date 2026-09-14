"""Recompute all audited legacy rollouts; preserve limitations of old traces."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metrics import analyze_steps, bootstrap, mean


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    audit=json.loads(args.audit.read_text())
    result=[]
    for r in audit['rollouts']:
        if r['status']!='verified':continue
        steps=[]
        for s in r['steps']:
            z=np.load(s['path'])
            steps.append(s|dict(mass=z['mass'],positions=z['positions']))
        settings={}
        for w,stride in [(16,16),(32,32),(64,64),(128,128),(128,32),(256,256)]:
            settings[f'{w}/{stride}']=analyze_steps(steps,w,stride)
        result.append(dict(instance_id=r['instance_id'],repository=r['repository'],settings=settings))
    summary={}
    for key in result[0]['settings']:
        valid=[r['settings'][key] for r in result if r['settings'][key]['within_js'] is not None
               and r['settings'][key]['cross_js'] is not None]
        summary[key]={m:bootstrap([r[m] for r in valid]) for m in ['within_js','cross_js','within_kl','cross_kl']}
        summary[key]['paired_rollouts']=len(valid)
        summary[key]['cross_gt_within']=sum(r['cross_js']>r['within_js'] for r in valid)
        summary[key]['paired_js_diff']=bootstrap([r['cross_js']-r['within_js'] for r in valid])
        matched=[r['settings'][key] for r in result if r['settings'][key]['matched_within_js'] is not None]
        summary[key]['strict_matched']={m:bootstrap([r[m] for r in matched])
            for m in ['matched_within_js','matched_cross_js']}
        summary[key]['strict_matched']['difference']=bootstrap([
            r['matched_cross_js']-r['matched_within_js'] for r in matched])
    report=dict(note='Legacy positional common support only: original token prefixes unavailable. '
        'Not a verified token-identity comparison. All 49 outcomes were iteration-capped.',
        summary=summary,rollouts=result)
    (args.out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(1,3,figsize=(12.8,3.9),layout='constrained')
    mainkey='32/32'
    valid=[r for r in result if r['settings'][mainkey]['matched_within_js'] is not None and r['settings'][mainkey]['matched_cross_js'] is not None]
    for i,r in enumerate(valid):
        s=r['settings'][mainkey];off=(i-(len(valid)-1)/2)*.003
        axs[0].plot([off,1+off],[s['matched_within_js'],s['matched_cross_js']],color='#b7c0c8',alpha=.6,lw=.6)
        axs[0].scatter([off,1+off],[s['matched_within_js'],s['matched_cross_js']],c=['#3069ad','#cf603f'],s=13,zorder=3)
    axs[0].set(xticks=[0,1],xticklabels=['Within step','Across step'],ylabel='JS divergence (bits)',
        title=f'(a) Matched steps, positional support\n{len(valid)} paired rollouts; window = 32')
    keys=[k for k in summary if k!='16/16']
    for j,k in enumerate(keys):
        for shift,m,c in [(-.14,'within_js','#3069ad'),(.14,'cross_js','#cf603f')]:
            s=summary[k][m]
            if s['mean'] is None:continue
            axs[1].errorbar(j+shift,s['mean'],yerr=[[s['mean']-s['ci95'][0]],[s['ci95'][1]-s['mean']]],
                fmt='o',color=c,capsize=3,label=m.split('_')[0] if j==0 else None)
    axs[1].set(xticks=range(len(keys)),xticklabels=[f"{k}\n(n={summary[k]['paired_rollouts']})" for k in keys],ylabel='Rollout-mean JS divergence',
        xlabel='Window / stride (tokens)',title='(b) Window and overlap sensitivity')
    axs[1].legend(frameon=False)
    # First 32 decode tokens are a diagnostic, never named prefill selection.
    for k,c in [('8','#7d53a2'),('16','#178d7a'),('32','#c28930')]:
        rows=[]
        for r in result:
            ds=[s['early_decode_proxy'][k]['gap'] for s in r['settings'][mainkey]['details']
                if k in s.get('early_decode_proxy',{})]
            if ds:rows.append([mean([d[b] for d in ds]) for b in range(10)])
        if rows:
            bins=[bootstrap([r[b] for r in rows]) for b in range(10)]
            x=np.arange(10)*10+5
            axs[2].plot(x,[v['mean']*100 for v in bins],label=f'k={k}',color=c)
            axs[2].fill_between(x,[v['ci95'][0]*100 for v in bins],[v['ci95'][1]*100 for v in bins],alpha=.12,color=c)
    axs[2].set(xlabel='Held-out decode progress (%)',ylabel='Oracle minus frozen coverage (pp)',
        title='(c) Early-decode diagnostic only\nSelect from first 32 decode tokens')
    axs[2].legend(frameon=False)
    for ax in axs:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    fig.savefig(args.out/'legacy49_controls.pdf',bbox_inches='tight')
    fig.savefig(args.out/'legacy49_controls.png',dpi=180,bbox_inches='tight')
    lines=['# Legacy-49 reanalysis','','All 49 request fingerprints verified. These results use positional historical support; original prompt prefixes were not archived.','',
        '| window/stride | paired rollouts | within JS | cross JS | cross > within |',
        '| --- | ---: | ---: | ---: | ---: |']
    for k,s in summary.items():
        lines.append(f"| {k} | {s['paired_rollouts']} | {s['within_js']['mean']:.5f} | {s['cross_js']['mean']:.5f} | {s['cross_gt_within']}/{s['paired_rollouts']} |")
    lines+=['','## Strict matched boundary comparison','',
        'Both adjacent steps must have at least two windows. Within and across distances use the same historical support and the same two steps. Token identity remains unverified for legacy data.','',
        '| window | paired rollouts | within JS | across JS | difference 95% CI |',
        '| --- | ---: | ---: | ---: | --- |']
    for key in ['16/16','32/32','64/64','128/128']:
        s=summary[key]['strict_matched'];a=s['matched_within_js'];b=s['matched_cross_js']
        lines.append(f"| {key} | {a['n']} | {a['mean']} | {b['mean']} | {s['difference']['ci95']} |")
    lines+=['','The frozen-set curve uses the first 32 decode tokens and evaluates subsequent tokens. The oracle is per token. This is not prefill selection or the production compact-index selector.']
    (args.out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
