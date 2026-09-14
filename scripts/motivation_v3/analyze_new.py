"""Create auditable multi-domain figures from completed request artifacts only."""
import argparse
from collections import Counter,defaultdict
import json
import hashlib
import subprocess
import sys
from pathlib import Path
import numpy as np
from metrics import analyze_steps,bootstrap,mean

from plot_main_figure import render, DOMAINS, LABELS, COLORS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--measurements',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text());results=[];status=Counter();exclusions=Counter()
    for row in manifest['trajectories']:
        steps=[]
        for i,frame in enumerate(row['frames']):
            p=args.measurements/row['id']/f'step_{i:03d}.json'
            if not p.exists():status['pending']+=1;continue
            d=json.loads(p.read_text());status[d['status']]+=1
            if d['status']!='collected':continue
            z=np.load(d['path'])
            steps.append(d|dict(step=d['source_step'],mass=z['mass'],positions=z['positions']))
            if not d['prefix_identity_verified']:exclusions['no_verified_previous_prefix']+=1
        if not steps:continue
        bywindow={str(w):analyze_steps(steps,w,identity_verified=True) for w in [16,32,64,128]}
        natural=[s for s in steps if s.get('finish_reason')!='length' and
                 s['completion_tokens']<manifest['max_decode_tokens']]
        natural_settings={str(w):analyze_steps(natural,w,identity_verified=True,compute_coverage=False)
                          for w in [16,32,64,128]}
        coverage=defaultdict(list)
        for d in bywindow['32']['details']:
            original=next(s for s in steps if s['step']==d['step'])
            if not original.get('prefix_identity_verified'):continue
            for k,c in d.get('prefill_diagnostic',{}).items():coverage[k].append(c)
        csummary={}
        for k,ds in coverage.items():
            csummary[k]={metric:[mean([d[metric][b] for d in ds]) for b in range(10)]
                         for metric in ['frozen','oracle','gap']}
            csummary[k]['mean']=mean([d['mean'] for d in ds])
            csummary[k]['oracle_mean']=mean([d['oracle_mean'] for d in ds])
            csummary[k]['mean_gap']=mean([d['oracle_mean']-d['mean'] for d in ds])
            csummary[k]['saturated_fraction']=mean([d['selected_blocks']==d['candidate_blocks'] for d in ds])
            csummary[k]['mean_candidate_blocks']=mean([d['candidate_blocks'] for d in ds])
            csummary[k]['mean_selected_blocks']=mean([d['selected_blocks'] for d in ds])
            csummary[k]['steps']=len(ds)
        results.append(dict(id=row['id'],domain=row['domain'],independent_unit=row['independent_unit'],
            collected_steps=len(steps),planned_steps=len(row['frames']),settings=bywindow,coverage=csummary,
            natural_end_settings=natural_settings,
            mean_prompt_tokens=mean([s['prompt_tokens'] for s in steps]),
            mean_decode_samples=mean([s['decode_samples'] for s in steps]),
            truncated_steps=sum(s.get('finish_reason')=='length' or
                s['completion_tokens']>=manifest['max_decode_tokens'] for s in steps)))
    summary={}
    for domain in DOMAINS:
        rows=[r for r in results if r['domain']==domain];s={}
        for w in ['16','32','64','128']:
            paired=[r['settings'][w] for r in rows if r['settings'][w]['matched_within_js'] is not None
                    and r['settings'][w]['matched_cross_js'] is not None]
            s[w]={m:bootstrap([r[m] for r in paired]) for m in ['matched_within_js','matched_cross_js']}
            s[w]['paired_difference']=bootstrap([r['matched_cross_js']-r['matched_within_js'] for r in paired])
            s[w]['cross_gt_within']=sum(r['matched_cross_js']>r['matched_within_js'] for r in paired)
            s[w]['paired_rollouts']=len(paired)
        s['coverage']={}
        s['natural_end_only']={}
        for w in ['16','32','64','128']:
            paired=[r['natural_end_settings'][w] for r in rows
                    if r['natural_end_settings'][w]['matched_within_js'] is not None and
                    r['natural_end_settings'][w]['matched_cross_js'] is not None]
            s['natural_end_only'][w]={m:bootstrap([r[m] for r in paired])
                for m in ['matched_within_js','matched_cross_js']}
            s['natural_end_only'][w]['paired_difference']=bootstrap([
                r['matched_cross_js']-r['matched_within_js'] for r in paired])
        for k in ['8','16','32','64','r05','r10','r20']:
            rows_k=[r['coverage'][k] for r in rows if k in r['coverage']]
            s['coverage'][k]={metric:[bootstrap([r[metric][b] for r in rows_k]) for b in range(10)]
                                 for metric in ['frozen','oracle','gap']}
        s['trajectories']=len(rows);s['complete_trajectories']=sum(r['collected_steps']==r['planned_steps'] for r in rows)
        summary[domain]=s
    report=dict(model=manifest['model'],status_counts=dict(status),exclusions=dict(exclusions),
        analysis_source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ['analyze_new.py','metrics.py','plot_main_figure.py']},
        evidence_scope=manifest['evidence_scope'],selector='Exact prefill-tail attention diagnostic top-k; '
        'not production compact-index retrieval',summary=summary,trajectories=results)
    (args.out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    render(report,manifest,args.out)
    complete=sum(v['complete_trajectories'] for v in summary.values())
    lines=['# Multi-domain attention collection','','## Scope','',manifest['evidence_scope'],'',
        'Prefill selection uses exact attention over the last recorded prefill queries. It does not measure the production compact index.',
        '',f'Request status: {dict(status)}','',
        '| domain | collected trajectories | complete trajectories | matched trajectories (W=32) | within JS | across JS |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for domain,s in summary.items():
        d=s['32'];lines.append(f"| {domain} | {s['trajectories']} | {s['complete_trajectories']} | {d['paired_rollouts']} | {d['matched_within_js']['mean']} | {d['matched_cross_js']['mean']} |")
    (args.out/'report.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable,str(Path(__file__).with_name('plot_budget_sensitivity.py')),
        '--analysis',str(args.out/'analysis.json'),'--out',str(args.out)],check=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_layers.py')),
        '--manifest',str(args.manifest),'--measurements',str(args.measurements),'--out',str(args.out)],check=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('refresh_distance_controls.py')),
        '--manifest',str(args.manifest),'--measurements',str(args.measurements),'--out',str(args.out)],check=True)
    print(json.dumps({'status':dict(status),'complete_trajectories':complete}))


if __name__=='__main__':main()
