"""Absolute decode distance and an identity-preserving stale-set control."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metrics import normalize, mean, bootstrap

# The native 2048-output cap yields at most 2047 decode forward queries.
EDGES = [0, 32, 64, 128, 256, 512, 1024, 1536, 2047]
DOMAINS = ['code', 'tool_state', 'dialogue', 'gui']


def matched_stale(decode, scores, previous_selected, end):
    """Keep only still-valid old block identities; match refresh to that count.

    decode/scores use current candidates [1,end). Old block IDs are global
    positions in an input prefix whose identity the caller has verified.
    Never backfill a stale set using new scores or claim its nominal budget.
    """
    kept = [int(b) for b in previous_selected if 1 <= b < end]
    if not kept or scores.sum() <= 0:
        return None
    p = normalize(decode)
    valid = decode.sum(1) > 0
    if not valid.any():
        return None
    k = len(kept)
    fresh = np.argsort(-scores, kind='stable')[:k]
    stale = p[:, np.asarray(kept)-1].sum(1)
    refreshed = p[:, fresh].sum(1)
    oracle = np.sort(p, axis=1)[:, -k:].sum(1)
    return dict(stale=mean(stale[valid]), refreshed=mean(refreshed[valid]),
        oracle=mean(oracle[valid]), gain=mean((refreshed-stale)[valid]),
        old_selected=len(previous_selected), surviving_selected=k,
        lost_selected=len(previous_selected)-k, candidate_blocks=end-1,
        effective_fraction=k/(end-1))


def distance_bins(values, valid):
    # Full bins only: do not compare a short bin prefix against a full bin.
    return [mean(values[a:b][valid[a:b]]) if len(values) >= b else None
            for a, b in zip(EDGES[:-1], EDGES[1:])]


def main():
    ap = argparse.ArgumentParser()
    for name in ['manifest', 'measurements', 'out']:
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    distance, stale_rows, excluded = [], [], Counter()
    for trajectory in manifest['trajectories']:
        previous = None
        for path in sorted((args.measurements/trajectory['id']).glob('step_*.json')):
            s = json.loads(path.read_text())
            if s['status'] != 'collected':
                previous = None
                continue
            with np.load(s['path']) as z:
                mass, pos = z['mass'], z['positions']
            prompt, bt = s['prompt_tokens'], s['block_tokens']
            end = min(mass.shape[1], prompt//bt-8, s.get('history_end_tokens',0)//bt)
            if not s['prefix_identity_verified'] or end <= 1:
                excluded['no_verified_candidate_history'] += 1
                previous = None
                continue
            prefill = mass[pos < prompt, 1:end]
            decode = mass[pos >= prompt, 1:end]
            scores = normalize(prefill).mean(0) if len(prefill) else np.zeros(end-1)
            if scores.sum() <= 0:
                excluded['no_prefill_scores'] += 1
                previous = None
                continue
            k = max(1, int(np.ceil((end-1)*.1)))
            selected = np.argsort(-scores, kind='stable')[:k]+1
            identity = dict(trajectory=trajectory['id'], domain=trajectory['domain'],
                            source_step=s['source_step'])
            if len(decode) and (decode.sum(1)>0).any():
                p = normalize(decode); valid = decode.sum(1)>0
                distance.append(identity | dict(decode_queries=len(decode),
                    fixed=distance_bins(p[:, selected-1].sum(1), valid),
                    oracle=distance_bins(np.sort(p, axis=1)[:, -k:].sum(1), valid)))
                if previous is not None and previous['step']+1 == s['source_step']:
                    control = matched_stale(decode, scores, previous['selected'], end)
                    if control is not None:
                        stale_rows.append(identity | control)
                    else:
                        excluded['stale_no_surviving_blocks'] += 1
                else:
                    excluded['stale_no_adjacent_previous_selection'] += 1
            else:
                excluded['no_positive_decode_history'] += 1
            previous = dict(step=s['source_step'], selected=selected)

    summary = {}
    for domain in DOMAINS:
        ds = [r for r in distance if r['domain'] == domain]
        bins = []
        for i in range(len(EDGES)-1):
            eligible = [r for r in ds if r['fixed'][i] is not None]
            groups = defaultdict(list)
            for r in eligible: groups[r['trajectory']].append(r)
            bins.append(dict(start=EDGES[i], stop=EDGES[i+1], requests=len(eligible),
                trajectories=len(groups), **{key:bootstrap([
                    mean([r[key][i] for r in rs]) for rs in groups.values()])
                    for key in ['fixed', 'oracle']}))
        rs = [r for r in stale_rows if r['domain'] == domain]
        groups = defaultdict(list)
        for r in rs: groups[r['trajectory']].append(r)
        summary[domain] = dict(distance=bins, stale=dict(requests=len(rs),
            trajectories=len(groups), **{key:bootstrap([
                mean([r[key] for r in group]) for group in groups.values()])
                for key in ['stale', 'refreshed', 'oracle', 'gain',
                            'effective_fraction', 'lost_selected']}))
    result = dict(summary=summary, distance_requests=distance, stale_requests=stale_rows,
        exclusions=dict(excluded), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Offline exact-attention diagnostic. Stale retains the previous prefill-selected '
        '10%-budget set, intersected with the current verified historical candidates. Refresh '
        'and oracle use the surviving block count, with no stale backfill. This paired effective '
        'budget differs from the main current-step 10% budget. Distance bins include only '
        'requests reaching the bin end; changing survivor composition prevents a causal length interpretation.')
    (args.out/'refresh_distance_controls.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':7, 'pdf.fonttype':42})
    fig, axes = plt.subplots(2,4,figsize=(7.2,4.8),layout='constrained')
    for col, domain in enumerate(DOMAINS):
        ax = axes[0,col]; bins = summary[domain]['distance']
        x = [(r['start']+r['stop'])/2 for r in bins]
        for key, style, label in [('fixed','-','Prefill fixed'),('oracle','--','Token oracle')]:
            y = [100*r[key]['mean'] if r[key]['mean'] is not None else np.nan for r in bins]
            color = '#376fa2' if key=='fixed' else '#888888'
            # A lone trajectory remains visible, but is not drawn as a supported trend.
            supported = [v if r['trajectories'] >= 2 else np.nan for v,r in zip(y,bins)]
            ax.plot(x,supported,style, label=label, color=color)
            single = [(position,value) for position,value,r in zip(x,y,bins)
                      if r['trajectories'] == 1]
            if single:
                ax.plot(*zip(*single),linestyle='none',marker='o',markersize=3,
                        markerfacecolor='none',markeredgecolor=color)
            lo=[100*r[key]['ci95'][0] if r[key]['ci95'][0] is not None else np.nan for r in bins]
            hi=[100*r[key]['ci95'][1] if r[key]['ci95'][1] is not None else np.nan for r in bins]
            ax.fill_between(x,lo,hi,alpha=.12,color='#376fa2' if key=='fixed' else '#888888')
        ax.set(title=domain,xscale='log',xticks=[32,128,512,2048],
            xticklabels=['32','128','512','2048'],xlim=(12,2200),ylim=(0,100),
            xlabel='Decode-query distance')
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.text(.02,.98,'Trajectories/bin (left to right):\n' +
                ' / '.join(str(r['trajectories']) for r in bins),
                transform=ax.transAxes,fontsize=5.5,va='top')
        if col==0:
            ax.plot([],[],linestyle='none',marker='o',markersize=3,
                    markerfacecolor='none',markeredgecolor='#555555',label='n=1 (no CI)')
            ax.set_ylabel('Historical coverage (%)')
            ax.legend(frameon=False,fontsize=6,loc='upper left',bbox_to_anchor=(.02,.84))
        s = summary[domain]['stale']; ax = axes[1,col]
        for i,key in enumerate(['stale','refreshed','oracle']):
            b=s[key]
            if b['mean'] is not None:
                ax.plot(i,100*b['mean'],'o',color='#376fa2')
                if b['ci95'][0] is not None:
                    ax.vlines(i,100*b['ci95'][0],100*b['ci95'][1],color='#376fa2')
        ax.set(xticks=[0,1,2],xticklabels=['Stale','Refresh','Oracle'],xlim=(-.4,2.4),
            ylim=(0,100),title=f"n={s['trajectories']}; {s['requests']} paired requests")
        if col==0: ax.set_ylabel('Matched-budget coverage (%)')
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.15)
    fig.suptitle('Absolute decode distance and refresh of a stale historical set',fontsize=9)
    for ext in ['pdf','png']:
        fig.savefig(args.out/f'refresh_distance_controls.{ext}',dpi=200,bbox_inches='tight')
    lines=['# Absolute-distance and stale-set controls','',result['scope'],'',
        'All means first average requests within a trajectory. Bands/intervals bootstrap trajectories.',
        'Absolute-distance bins below report changing eligibility explicitly.','',
        '| Domain | Query interval [start, stop) | Trajectories | Requests |',
        '| --- | --- | ---: | ---: |']
    for domain,s in summary.items():
        for b in s['distance']:
            lines.append(f"| {domain} | {b['start']}–{b['stop']} | {b['trajectories']} | {b['requests']} |")
    (args.out/'refresh_distance_controls.md').write_text('\n'.join(lines)+'\n')
    (args.out/'refresh_distance_controls_caption.txt').write_text(
        'Top: fixed-prefill and same-budget token-oracle historical coverage versus '
        'absolute decode-query distance, using a 10% candidate-block budget rounded up. '
        'Only requests reaching a bin end contribute; eligibility counts for every bin '
        'are in refresh_distance_controls.md and trajectory counts appear inside each panel. '
        'Hollow, unconnected points denote bins with just one trajectory and no estimable CI. '
        'The last bin ends at 2047 queries to match '
        'the native 2048-output cap. Bottom: the previous step\'s selected blocks are '
        'retained only where their identity survives in the current verified candidate '
        'prefix. Current-prefill refresh and the token oracle use that surviving block '
        'count. No new blocks fill stale slots. Means first aggregate within trajectories; '
        'intervals bootstrap trajectories and are omitted for n=1. Exact attention is '
        'an offline scheduling diagnostic, not a production retriever or task-quality measure.\n')
    print(json.dumps({'distance_requests':len(distance),'stale_pairs':len(stale_rows),'exclusions':dict(excluded)}))


if __name__ == '__main__': main()
