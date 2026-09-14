"""Layer-wise matched-boundary control from preserved raw attention rows."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metrics import js,mean,bootstrap

LAYERS=list(range(3,64,4))
DOMAINS=['code','tool_state','dialogue','gui']
POSITION=re.compile(br'"position"\s*:\s*(\d+)')


def read_edge_queries(record,window):
    p,n=record['prompt_tokens'],record['decode_samples']
    wanted=set(range(p,p+2*window))|set(range(p+n-2*window,p+n))
    rows=defaultdict(dict)
    with Path(record['trace_path']).open('rb') as f:
        f.seek(record['trace_start'])
        while f.tell()<record['trace_end']:
            line=f.readline()
            if not line:break
            match=POSITION.search(line[:180])
            if not match or int(match[1]) not in wanted:continue
            r=json.loads(line)
            if r.get('kind')!='attention_mass' or r.get('mode')!='global':continue
            pos,layer=int(r['position']),int(r['layer'])
            if pos in rows[layer]:raise ValueError('duplicate layer/query row')
            ids=np.asarray(r['block_ids']);v=np.asarray(r['mass'],dtype=float);keep=ids>=0
            if not np.isfinite(v).all() or (v < -1e-6).any():raise ValueError('invalid mass')
            dense=np.zeros(int(ids[keep].max())+1);dense[ids[keep]]=np.maximum(v[keep],0)
            if dense.sum()<=0:raise ValueError('zero total mass')
            rows[layer][pos]=dense/dense.sum()
    if set(rows)!=set(LAYERS) or any(set(v)!=wanted for v in rows.values()):
        raise ValueError('missing layer/edge query')
    return rows


def interval_average(rows,layer,start,stop,upper):
    if layer is None:
        return np.mean([rows[l][p][1:upper] for p in range(start,stop) for l in LAYERS],axis=0)
    return np.mean([rows[layer][p][1:upper] for p in range(start,stop)],axis=0)


def analyze_trajectory(records,window):
    results=defaultdict(list);previous=None
    for d in records:
        if d['decode_samples']<2*window:previous=None;continue
        current=read_edge_queries(d,window)
        if previous is not None and d['prefix_identity_verified'] and d['source_step']==previous[0]['source_step']+1:
            old,oldrows=previous
            upper=min(d['prompt_tokens']//128-8,old['prompt_tokens']//128-8,d['common_prefix_tokens']//128)
            if upper>1:
                oldend=old['prompt_tokens']+old['decode_samples'];newstart=d['prompt_tokens']
                for layer in LAYERS+[None]:
                    a=interval_average(oldrows,layer,oldend-window,oldend,upper)
                    b=interval_average(current,layer,newstart,newstart+window,upper)
                    before=interval_average(oldrows,layer,oldend-2*window,oldend-window,upper)
                    after=interval_average(current,layer,newstart+window,newstart+2*window,upper)
                    within=mean([js(before,a),js(b,after)]);across=js(a,b)
                    if within is not None and across is not None:
                        results['layer_mean_first' if layer is None else str(layer)].append(
                            dict(within=within,across=across,difference=across-within))
        previous=(d,current)
    return {layer:{metric:mean([r[metric] for r in rows]) for metric in ['within','across','difference']}
        |{'boundaries':len(rows)} for layer,rows in results.items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--measurements',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--window',type=int,default=32);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    cache=args.out/'layer_cache';cache.mkdir(exist_ok=True)
    manifest=json.loads(args.manifest.read_text());results=[]
    codehash=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('metrics.py').read_bytes()).hexdigest()
    for r in manifest['trajectories']:
        records=[];raw=[]
        for i in range(len(r['frames'])):
            p=args.measurements/r['id']/f'step_{i:03d}.json'
            if not p.exists():break
            text=p.read_text();d=json.loads(text)
            if d['status']!='collected':break
            records.append(d);raw.append(text)
        if len(records)!=len(r['frames']):continue
        fingerprint=hashlib.sha256((codehash+str(args.window)+''.join(raw)).encode()).hexdigest()
        p=cache/(r['id']+'.json');cached=json.loads(p.read_text()) if p.exists() else None
        if cached is None or cached['fingerprint']!=fingerprint:
            cached=dict(id=r['id'],domain=r['domain'],fingerprint=fingerprint,
                values=analyze_trajectory(records,args.window))
            p.write_text(json.dumps(cached,indent=2)+'\n')
            print('layer audit:',r['id'],flush=True)
        results.append(cached)
    summary={};grid=np.full((len(LAYERS),len(DOMAINS)),np.nan)
    lines=['# Layer-wise matched boundaries','',
        'Completed trajectories only. Each layer uses the same matched requests, verified historical support, '
        'and disjoint windows as the primary analysis. Heads are summed by the raw tracer; this is not a per-head control.',
        '', '| Domain | Layer (zero-based) | Trajectories | Mean across-minus-within JS | 95% CI |',
        '| --- | --- | ---: | ---: | --- |']
    for j,domain in enumerate(DOMAINS):
        summary[domain]={}
        for layer in [str(l) for l in LAYERS]+['layer_mean_first']:
            rows=[r['values'][layer] for r in results if r['domain']==domain and layer in r['values']]
            s={m:bootstrap([r[m] for r in rows]) for m in ['within','across','difference']};summary[domain][layer]=s
            b=s['difference'];lines.append(f"| {domain} | {layer} | {b['n']} | {b['mean']} | {b['ci95']} |")
            if layer!='layer_mean_first' and b['mean'] is not None:grid[LAYERS.index(int(layer)),j]=b['mean']
    report=dict(window=args.window,complete_trajectories=len(results),analysis_source_sha256=codehash,
                summary=summary,trajectories=results)
    (args.out/'layer_robustness.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (args.out/'layer_robustness.md').write_text('\n'.join(lines)+'\n')
    primary_path=args.out/'analysis.json'
    if primary_path.exists():
        primary={r['id']:r for r in json.loads(primary_path.read_text())['trajectories']}
        differences=[];ids=[]
        for r in results:
            old=primary.get(r['id']);new=r['values'].get('layer_mean_first')
            if old is None or new is None or old['collected_steps']!=old['planned_steps']:continue
            old=old['settings'][str(args.window)];ids.append(r['id'])
            for metric,key in [('within','matched_within_js'),('across','matched_cross_js')]:
                if old[key] is not None:differences.append(abs(new[metric]-old[key]))
        check={'trajectories':ids,'comparisons':len(differences),'max_absolute_error':max(differences,default=0),
            'tolerance':1e-6,'scope':'Reconstruct the primary layer-mean distribution independently from raw rows.'}
        (args.out/'layer_reconstruction_check.json').write_text(json.dumps(check,indent=2)+'\n')
        if check['max_absolute_error']>=check['tolerance']:raise ValueError('raw-layer reconstruction differs from primary metrics')
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(5.4,5.2),layout='constrained')
    bound=max(.01,float(np.nanmax(np.abs(grid)))) if np.isfinite(grid).any() else 1
    im=ax.imshow(np.ma.masked_invalid(grid),cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
    ax.set(xticks=range(4),xticklabels=['Software','Tool/state','Dialogue','GUI'],yticks=range(len(LAYERS)),
        yticklabels=LAYERS,ylabel='Standard-attention layer (zero-based)',
        title=f'Layer-wise across-minus-within JS (W={args.window})\n{len(results)} complete trajectories; counts and CIs in accompanying table')
    fig.colorbar(im,ax=ax,label='Difference in JS divergence (bits)')
    fig.savefig(args.out/'layer_robustness.pdf',bbox_inches='tight')
    fig.savefig(args.out/'layer_robustness.png',dpi=200,bbox_inches='tight')
    print('complete trajectories:',len(results))


if __name__=='__main__':main()
