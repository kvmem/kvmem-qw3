"""Audit all 49 collected SWE rollouts against immutable raw attention data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from trace_io import iter_segments, save_segment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=Path('benchmark/results/motivation_v2'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index_file = args.out / 'segments.json'
    raw = args.source / 'raw/attn_shared.jsonl'
    if index_file.exists():
        cache = json.loads(index_file.read_text())
        if cache['raw_size'] != raw.stat().st_size:
            raise ValueError('raw size changed; use a new output directory')
        segments = cache['segments']
    else:
        segments = []
        for i, seg in enumerate(iter_segments(raw)):
            record = save_segment(args.out / 'segments' / f'{i:04d}.npz', seg)
            record['segment'] = i
            segments.append(record)
            if i % 25 == 0:
                print(f'indexed {i+1} segments ({record["byte_end"]/1e9:.2f} GB)', flush=True)
        h = hashlib.sha256()
        with raw.open('rb') as f:
            for chunk in iter(lambda: f.read(8 << 20), b''):
                h.update(chunk)
        index_file.write_text(json.dumps(dict(raw_path=str(raw.resolve()),
            raw_size=raw.stat().st_size, raw_sha256=h.hexdigest(), segments=segments), indent=2)+'\n')
    manifest = [json.loads(s) for s in (args.source/'manifest.jsonl').read_text().splitlines() if s.strip()]
    audit, used = [], set()
    for row in manifest:
        if row['status'] != 'collected':
            continue
        iid = row['instance_id']
        p = args.source/'rollouts'/iid/'output.jsonl'
        outputs = [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
        valid = [d for d in outputs if d.get('metrics', {}).get('token_usages')]
        rec = dict(instance_id=iid, repository=row['repository'], manifest_status='collected')
        if not valid:
            audit.append(rec | dict(status='excluded', reason='no_llm_metrics'))
            continue
        d = valid[-1]
        usages = d['metrics']['token_usages']
        fingerprint = [(int(u['prompt_tokens']), int(u['completion_tokens'])) for u in usages]
        matches = []
        for i in range(len(segments)-len(usages)+1):
            got = [(s['first_position'], s['samples']) for s in segments[i:i+len(usages)]]
            if got == fingerprint:
                matches.append(i)
        rec.update(requests=len(usages), prompt_tokens=[x[0] for x in fingerprint],
                   completion_tokens=[x[1] for x in fingerprint],
                   outcome='max_iterations' if 'MaxIterations' in str(d.get('error')) else 'other',
                   evaluation=d.get('test_result'), candidates=matches)
        if len(matches) != 1:
            audit.append(rec | dict(status='excluded', reason='ambiguous_or_missing_fingerprint'))
            continue
        indices = list(range(matches[0], matches[0]+len(usages)))
        if used.intersection(indices):
            raise ValueError('different instances assigned the same trace segment')
        used.update(indices)
        rec.update(status='verified', segment_indices=indices,
                   steps=[segments[i] | dict(step=j, prompt_tokens=fingerprint[j][0],
                           completion_tokens=fingerprint[j][1]) for j, i in enumerate(indices)],
                   prefix_identity_verified=False,
                   caution='Request counts align exactly; legacy prompt-token prefixes were not archived.')
        audit.append(rec)
    report = dict(source='OpenHands / SWE-bench Lite', model='Qwen3.6-27B-Q8_0',
                  max_agent_iterations=6, kvmem_enabled=False,
                  status_counts=dict(Counter(x['status'] for x in audit)),
                  repository_counts=dict(Counter(x['repository'] for x in audit)),
                  outcome_counts=dict(Counter(x.get('outcome') for x in audit)),
                  total_segments=len(segments), assigned_segments=len(used), rollouts=audit)
    (args.out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rollouts'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
