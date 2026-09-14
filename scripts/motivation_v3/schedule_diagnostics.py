"""Offline causal scheduling controls; never a production latency measurement."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metrics import normalize, mean, bootstrap


def schedules(mass, prefill_scores, fraction=.1, intervals=(32,128,512)):
    p=normalize(mass); valid=mass.sum(1)>0
    if not valid.any() or prefill_scores.sum()<=0:return {}
    k=max(1,int(np.ceil(p.shape[1]*fraction)))
    initial=np.argsort(-prefill_scores,kind='stable')[:k]
    curves={'prefill_fixed':p[:,initial].sum(1),
            'token_oracle':np.sort(p,axis=1)[:,-k:].sum(1)}
    updates={'prefill_fixed':1,'token_oracle':len(p)}
    for interval in intervals:
        v=np.zeros(len(p));count=0
        for start in range(0,len(p),interval):
            # The update sees only queries preceding the evaluated token.
            selected=initial if start==0 else np.argsort(-p[max(0,start-32):start].mean(0),kind='stable')[:k]
            v[start:start+interval]=p[start:start+interval,selected].sum(1);count+=1
        curves[f'periodic_{interval}']=v;updates[f'periodic_{interval}']=count
    return {name:dict(coverage=mean(v[valid]),updates=updates[name],
        absolute_attention_mass=mean((v*mass.sum(1))[valid]),
        first_quarter=mean(v[:max(1,len(v)//4)][valid[:max(1,len(v)//4)]]),
        last_quarter=mean(v[3*len(v)//4:][valid[3*len(v)//4:]])) for name,v in curves.items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--measurements',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text());rows=[];exclusions=Counter()
    for trajectory in manifest['trajectories']:
        for path in sorted((args.measurements/trajectory['id']).glob('step_*.json')):
            s=json.loads(path.read_text())
            if s['status']!='collected':continue
            if not s['prefix_identity_verified']:
                exclusions['no_verified_previous_prefix']+=1;continue
            z=np.load(s['path']);mass,pos=z['mass'],z['positions'];prompt=s['prompt_tokens']
            end=min(mass.shape[1],prompt//128-8,s['history_end_tokens']//128)
            if end<=1:
                exclusions['no_candidate_history_after_exclusions']+=1;continue
            prefill=mass[pos<prompt,1:end];decode=mass[pos>=prompt,1:end]
            if not len(prefill) or not len(decode):
                exclusions['no_prefill_or_decode_samples']+=1;continue
            controls=schedules(decode,normalize(prefill).mean(0))
            if not controls:
                exclusions['zero_historical_mass_or_prefill_score']+=1;continue
            n=len(decode)
            rows.append(dict(trajectory=trajectory['id'],domain=trajectory['domain'],step=s['step'],
                prompt_tokens=prompt,decode_tokens=n,completion_tokens=s['completion_tokens'],
                truncated=s['finish_reason']=='length' or s['completion_tokens']>=manifest['max_decode_tokens'],
                context_group='<8K' if prompt<8192 else '8–32K' if prompt<32768 else '32K+',
                decode_group='<128' if n<128 else '128–511' if n<512 else '512+',
                candidate_group='<16 blocks' if end-1<16 else '16–63 blocks' if end-1<64 else '64+ blocks',
                historical_mass=mean(decode.sum(1)),candidate_blocks=end-1,
                selected_blocks=max(1,int(np.ceil((end-1)*.1))),
                actual_selected_fraction=max(1,int(np.ceil((end-1)*.1)))/(end-1),
                schedules=controls))
    summary={}
    # Cluster by trajectory separately inside each stratum. No window pseudoreplication.
    for group in ['domain','context_group','decode_group','candidate_group','truncated']:
        summary[group]={}
        for value in sorted(set(str(r[group]) for r in rows)):
            subset=[r for r in rows if str(r[group])==value];bytraj=defaultdict(list)
            for r in subset:bytraj[r['trajectory']].append(r)
            detail={'steps':len(subset),'trajectories':len(bytraj),
                'actual_selected_fraction':bootstrap([mean([r['actual_selected_fraction'] for r in rs]) for rs in bytraj.values()]),
                'historical_mass':bootstrap([mean([r['historical_mass'] for r in rs]) for rs in bytraj.values()])}
            for schedule in ['prefill_fixed','periodic_32','periodic_128','periodic_512','token_oracle']:
                detail[schedule]={metric:bootstrap([mean([r['schedules'].get(schedule,{}).get(metric) for r in rs])
                    for rs in bytraj.values()]) for metric in ['coverage','absolute_attention_mass','updates','first_quarter','last_quarter']}
            summary[group][value]=detail
    output={'scope':'Offline full-visible attention diagnostic at 10% of candidate blocks. Periodic updates use the previous 32 query rows; token oracle is noncausal. Update counts are not latency measurements.',
            'summary':summary,'exclusions':dict(exclusions),'steps':rows}
    (args.out/'schedule_diagnostics.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':7.5,'pdf.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.3),layout='constrained')
    colors={'code':'#376fa2','tool_state':'#ae6136','dialogue':'#33917a','gui':'#8358a1'}
    for domain,color in colors.items():
        ds=[r for r in rows if r['domain']==domain]
        if not ds:continue
        for ax,key,scale in [(axs[0,0],'prompt_tokens',1),(axs[0,1],'candidate_blocks',1),
                             (axs[1,0],'historical_mass',100)]:
            vals=np.sort([r[key]*scale for r in ds]);ax.step(vals,np.arange(1,len(vals)+1)/len(vals),
                where='post',color=color,label=f'{domain} (steps={len(ds)})')
        s=summary['domain'][domain];names=['prefill_fixed','periodic_512','periodic_128','periodic_32']
        axs[1,1].plot([s[n]['updates']['mean'] for n in names],
            [100*s[n]['coverage']['mean'] for n in names],'-o',color=color,label=domain,ms=4)
    axs[0,0].set(xlabel='Input tokens',ylabel='Fraction of eligible requests',xscale='log')
    axs[0,0].set(xlim=(1024,65536),xticks=[1024,4096,16384,65536],xticklabels=['1K','4K','16K','64K'])
    axs[0,0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[0,1].set(xlabel='Verified historical candidate blocks',ylabel='Fraction of eligible requests',xscale='log')
    axs[0,1].set(xlim=(1,512),xticks=[1,8,64,512],xticklabels=['1','8','64','512'])
    axs[0,1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[1,0].set(xlabel='Attention mass on candidate history (%)',ylabel='Fraction of eligible requests',xlim=(0,100))
    axs[1,1].set(xlabel='Mean selection updates per request',ylabel='Historical attention covered (%)',xscale='log',ylim=(0,100))
    for ax in axs.flat:
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
        if ax.lines:ax.legend(fontsize=6.5,frameon=False)
    for ax in [axs[0,0],axs[0,1],axs[1,0]]:ax.set(ylim=(0,1.03),yticks=[0,.5,1])
    fig.suptitle('Support checks and offline causal scheduling controls\n10% block budget; update counts are not measured latency',fontsize=9)
    fig.savefig(args.out/'scheduling_controls.pdf',bbox_inches='tight')
    fig.savefig(args.out/'scheduling_controls.png',dpi=180,bbox_inches='tight')
    (args.out/'scheduling_controls_caption.txt').write_text(
        'Input length, verified candidate-history size, and total attention assigned to candidate history '
        'are shown as empirical distributions over eligible requests. Bottom right compares the once-per-step '
        'prefill selection with additional causal refreshes every 512, 128, and 32 decode tokens '
        '(left to right within each domain, with coincident points possible). Refreshes use the previous '
        '32 query rows. Both coverage and update counts are averaged within trajectories before '
        'averaging across trajectories. Counts do not measure runtime.\n')
    print(f'{len(rows)} eligible request diagnostics')


if __name__=='__main__':main()
