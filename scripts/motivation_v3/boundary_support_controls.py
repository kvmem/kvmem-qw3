"""Exploratory boundary drift stratified by verified common-history size."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from metrics import bootstrap, mean


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();data=json.loads(args.analysis.read_text());groups=defaultdict(list)
    boundary_count=0
    for trajectory in data['trajectories']:
        if trajectory['collected_steps']!=trajectory['planned_steps']:continue
        bins=defaultdict(list)
        for boundary in trajectory['settings']['32']['matched_boundaries']:
            if boundary['within_js'] is None or boundary['cross_js'] is None:continue
            size=boundary['common_blocks']
            label='<16' if size<16 else '16–63' if size<64 else '64+'
            bins[label].append(boundary);boundary_count+=1
        for label,rows in bins.items():
            groups[(trajectory['domain'],label)].append(dict(
                trajectory=trajectory['id'],boundaries=len(rows),
                within=mean([r['within_js'] for r in rows]),
                across=mean([r['cross_js'] for r in rows]),
                difference=mean([r['cross_js']-r['within_js'] for r in rows])))
    summary={}
    for domain in ['code','tool_state','dialogue','gui']:
        summary[domain]={}
        for label in ['<16','16–63','64+']:
            rows=groups[(domain,label)]
            summary[domain][label]=dict(trajectories=len(rows),
                boundaries=sum(r['boundaries'] for r in rows),
                **{key:bootstrap([r[key] for r in rows]) for key in ['within','across','difference']})
    assert sum(s['boundaries'] for d in summary.values() for s in d.values())==boundary_count
    result=dict(window=32,complete_trajectories=sum(
        r['collected_steps']==r['planned_steps'] for r in data['trajectories']),
        eligible_boundaries=boundary_count,summary=summary,
        trajectory_strata=[dict(domain=domain,support_blocks=label,**r)
                           for (domain,label),rows in groups.items() for r in rows],
        scope='Exploratory addition after observing compacted GUI inputs and short verified prefixes. '
        'Reuses the existing <16/16–63/64+ block thresholds for a new boundary-support analysis. '
        'Primary eligibility and estimates remain unchanged. Trajectories may enter several strata; '
        'bootstrap resamples trajectory means within each workload/stratum. Strata differ in sample '
        'composition, so differences between strata are descriptive, not causal. A large common '
        'support does not explicitly remove system/tool-schema tokens.',
        source_analysis_sha256=hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'boundary_support_controls.json').write_text(json.dumps(result,indent=2)+'\n')
    def show(x):
        if x['mean'] is None:return '—'
        lo,hi=x['ci95']
        return f"{x['mean']:.3f} [CI unavailable]" if lo is None else f"{x['mean']:.3f} [{lo:.3f}, {hi:.3f}]"
    lines=['# Exploratory common-history support check','',result['scope'],'',
           'W=32, disjoint query windows; one historical block contains 128 tokens. '
           'The 64+ stratum therefore requires at least 8192 verified common historical tokens.','',
           '| Workload | Common blocks | Trajectories / boundaries | Across − within JS (bits), 95% CI |',
           '| --- | --- | ---: | --- |']
    for domain,values in summary.items():
        for label,s in values.items():
            lines.append(f"| {domain} | {label} | {s['trajectories']} / {s['boundaries']} | {show(s['difference'])} |")
    (args.out/'boundary_support_controls.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'eligible_boundaries':boundary_count,'complete_trajectories':result['complete_trajectories']}))


if __name__=='__main__':main()
