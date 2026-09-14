"""Expose budget dependence and trivial full-history coverage explicitly."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metrics import bootstrap

DOMAINS=['code','tool_state','dialogue','gui']
LABELS=['Software agent','Tool/state history','Dialogue','GUI agent']
COLORS=['#376fa2','#ae6136','#33917a','#8358a1']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.analysis.read_text());summary={};lines=['# Budget sensitivity','',
        'Trajectory means and trajectory-bootstrap 95% CI. Full-history fraction is the '
        'trajectory-mean fraction of eligible steps whose candidate history fits entirely in the nominal budget.',
        '', '| Domain | Budget | Trajectories | Frozen coverage (%) | Oracle gap (pp) | Steps fitting entirely, macro mean (%) |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    plt.rcParams.update({'font.size':7,'pdf.fonttype':42})
    fig,axs=plt.subplots(2,4,figsize=(7.2,4.4),layout='constrained')
    for col,domain in enumerate(DOMAINS):
        rows=[r for r in data['trajectories'] if r['domain']==domain];summary[domain]={}
        for ri,keys in enumerate([['r05','r10','r20'],['8','16','32','64']]):
            ax=axs[ri,col];x=np.arange(len(keys));stats=[]
            for k in keys:
                cs=[r['coverage'][k] for r in rows if k in r['coverage']]
                s={m:bootstrap([c[m] for c in cs]) for m in ['mean','oracle_mean','mean_gap','saturated_fraction']}
                summary[domain][k]=s;stats.append(s)
                if cs:
                    lines.append(f"| {domain} | {k} | {len(cs)} | {100*s['mean']['mean']:.1f} | {100*s['mean_gap']['mean']:.1f} | {100*s['saturated_fraction']['mean']:.1f} |")
            for metric,color,label,style in [('mean',COLORS[col],'Prefill-fixed','-'),('oracle_mean','#777777','Token oracle','--')]:
                if not any(s[metric]['mean'] is not None for s in stats):continue
                y=[np.nan if s[metric]['mean'] is None else 100*s[metric]['mean'] for s in stats]
                ax.plot(x,y,style+'o',color=color,label=label,ms=3,lw=1)
                for j,s in enumerate(stats):
                    b=s[metric]
                    if b['ci95'][0] is not None:
                        ax.errorbar(j,100*b['mean'],yerr=[[100*(b['mean']-b['ci95'][0])],
                            [100*(b['ci95'][1]-b['mean'])]],color=color,capsize=2,lw=.7)
            ax.set(xticks=x,xticklabels=['5%','10%','20%'] if ri==0 else keys,
                   ylim=(0,100),xlabel='Candidate fraction (ceil)' if ri==0 else 'Nominal block budget')
            if ri==0:ax.set_title(LABELS[col])
            if col==0:ax.set_ylabel('Historical attention covered (%)')
            if ax.lines:ax.legend(fontsize=6,frameon=False)
            else:ax.text(.5,.5,'Awaiting eligible data',ha='center',transform=ax.transAxes,color='#777777')
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    count=sum(v['complete_trajectories'] for v in data['summary'].values())
    fig.suptitle(f'Budget sensitivity — {count} complete recorded-input trajectories\n95% trajectory-bootstrap CI; budget saturation is reported separately',fontsize=8)
    fig.savefig(args.out/'budget_sensitivity.pdf',bbox_inches='tight')
    fig.savefig(args.out/'budget_sensitivity.png',dpi=240,bbox_inches='tight')
    (args.out/'budget_sensitivity.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.out/'budget_sensitivity.md').write_text('\n'.join(lines)+'\n')
    print('Budget sensitivity plots and saturation table generated')


if __name__=='__main__':main()
