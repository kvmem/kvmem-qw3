"""Render the main paper figure from saved statistics, without rereading raw traces."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DOMAINS=['code','tool_state','dialogue','gui']
LABELS={'code':'Software agent','tool_state':'Tool/state history','dialogue':'Dialogue','gui':'GUI agent'}
COLORS=['#376fa2','#ae6136','#33917a','#8358a1']


def render(report, manifest, out, metric='js'):
    if metric not in ('js', 'kl'):
        raise ValueError(f'Unsupported divergence: {metric}')
    within_key, cross_key = f'matched_within_{metric}', f'matched_cross_{metric}'
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    results=report['trajectories'];summary=report['summary'];status=Counter(report['status_counts'])
    plt.rcParams.update({'font.size':7,'pdf.fonttype':42})
    fig,axs=plt.subplots(2,4,figsize=(7.2,4.4),layout='constrained')
    paired_values=[r['settings']['32'][k] for r in results
        for k in [within_key,cross_key] if r['settings']['32'][k] is not None]
    paired_ymax=max(.01,max(paired_values)*1.12) if paired_values else 1
    if metric=='js':paired_ymax=min(1,paired_ymax)
    for col,domain in enumerate(DOMAINS):
        ax=axs[0,col];rows=[r for r in results if r['domain']==domain]
        valid=[r['settings']['32'] for r in rows if r['settings']['32'][within_key] is not None
               and r['settings']['32'][cross_key] is not None]
        for i,r in enumerate(valid):
            off=(i-(len(valid)-1)/2)*.006
            ax.plot([off,1+off],[r[within_key],r[cross_key]],lw=.7,color='#b9c1c8')
            ax.scatter([off,1+off],[r[within_key],r[cross_key]],c=['#3976ad','#c36842'],s=16,zorder=3)
        for xpos,key in [(-.16,within_key),(1.16,cross_key)]:
            estimate=summary[domain]['32'][key]
            if estimate['mean'] is not None:
                lo,hi=estimate['ci95'];center=estimate['mean']
                error=None if lo is None else [[center-lo],[hi-center]]
                ax.errorbar([xpos],[center],yerr=error,fmt='D',color='#202020',
                            ms=3,capsize=2,lw=1,zorder=5,
                            label=('Mean, 95% CI' if lo is not None else 'Mean (CI unavailable)')
                            if xpos<0 else None)
        if col==0 and valid:ax.legend(loc='upper left',fontsize=5.5,frameon=False)
        ax.set(xticks=[0,1],xticklabels=['Within','Across'],ylim=(0,paired_ymax),
            xlim=(-.28,1.28),
            title=f'{LABELS[domain]}\n{len(valid)} paired trajectories')
        if col==0:ax.set_ylabel(f'{metric.upper()} divergence (bits)')
        if not valid:ax.text(.5,.5,'Awaiting eligible data',ha='center',transform=ax.transAxes,color='#777777')
        ax=axs[1,col];c=summary[domain]['coverage']['r10'];x=np.arange(10)*10+5
        for coverage_metric,color,label in [('frozen',COLORS[col],'Prefill-fixed (10%, ceil)'),('oracle','#7b8389','Token oracle (10%, ceil)')]:
            ys=c[coverage_metric]
            if any(v['mean'] is not None for v in ys):
                ax.plot(x,[np.nan if v['mean'] is None else 100*v['mean'] for v in ys],color=color,label=label,
                        linestyle='--' if coverage_metric=='oracle' else '-')
                ax.fill_between(x,[np.nan if v['ci95'][0] is None else 100*v['ci95'][0] for v in ys],
                    [np.nan if v['ci95'][1] is None else 100*v['ci95'][1] for v in ys],color=color,alpha=.12)
        ax.set(xlabel='Decode progress (%)',ylim=(0,100),xlim=(0,100),xticks=[0,50,100])
        if col==0:ax.set_ylabel('Historical attention covered (%)')
        if ax.lines:ax.legend(fontsize=5.5,frameon=False)
        else:ax.text(.5,.5,'Awaiting eligible data',ha='center',transform=ax.transAxes,color='#777777')
        for a in axs[:,col]:a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.16)
    complete=sum(s['complete_trajectories'] for s in summary.values())
    fig.suptitle(f'Full-visible attention on recorded agent inputs — {complete}/{len(manifest["trajectories"])} complete\n'
        'Matched history and non-overlapping 32-query windows; trajectory bootstrap 95% CI'
        + ('\nKL(earlier || later); probability smoothing: 1e-9; linear scale' if metric=='kl' else ''),fontsize=8)
    fig.savefig(out/'multi_domain_scheduling.pdf',bbox_inches='tight')
    fig.savefig(out/'multi_domain_scheduling.png',dpi=240,bbox_inches='tight')
    boundary_counts=[sum(len(r['settings']['32']['matched_boundaries']) for r in results
                         if r['domain']==domain) for domain in DOMAINS]
    caption=(f'Historical-attention scheduling diagnostics on {complete} complete recorded-input trajectories '
        f'({status["collected"]} measured model calls; {manifest["model"]}). '
        'Top: each connected pair is a trajectory mean; across-step and adjacent within-step '
        f'{metric.upper()} divergences use the same two steps, verified common historical prefix, and disjoint '
        '32-query windows. Black diamonds show workload means; trajectory-bootstrap 95% '
        'confidence intervals are shown when at least two trajectories are eligible. '
        'Eligibility counts are shown separately by domain. '
        f'Eligible adjacent boundaries (software/tool/dialogue/GUI): {"/".join(map(str,boundary_counts))}. '
        'Bottom: history blocks selected from the last 64 prefill queries remain fixed during '
        'decode, at a budget of 10% of candidate blocks (rounded up). The dashed curve is a '
        'per-token oracle at the same budget; bands are trajectory-bootstrap 95% confidence '
        'intervals. Coverage is conditional on candidate historical attention, excluding the '
        'first block and eight recent prompt blocks. Inputs are replayed from recorded histories; '
        'new outputs are not executed. This measures exact-attention scheduling diagnostics, '
        'not production compact-index retrieval or task success.'
        + (' KL is computed as D_KL(earlier || later), with base-2 logarithms. '
           'Each normalized block probability receives epsilon=1e-9 and is renormalized '
           'before computing KL. Across compares the last window of the previous step '
           'with the first window of the next; within averages the two adjacent-window '
           'KL values on either side of that boundary. The vertical axis is linear and '
           'shared across workloads; no additional logarithm is applied to divergence.'
           if metric=='kl' else '') + '\n')
    (out/'figure_caption.txt').write_text(caption)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--metric',choices=['js','kl'],default='js')
    args=parser.parse_args()
    render(json.loads(args.analysis.read_text()),json.loads(args.manifest.read_text()),args.out,args.metric)


if __name__=='__main__':main()
