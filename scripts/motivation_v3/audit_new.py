"""Independently verify completed request artifacts against the frozen cohort."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from trace_io import expected_decode_queries


def prefix_length(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return i
    return min(len(a),len(b))


def collection_digest(row, manifest, default_digest, origins):
    """Check lineage without rewriting the original measurement's manifest hash."""
    if 'collection_cohort' not in row:
        assert not manifest.get('collection_manifests'), 'missing collection lineage'
        return default_digest
    name = row['collection_cohort']
    reference = manifest['collection_manifests'][name]
    if name not in origins:
        raw = Path(reference['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == reference['sha256'], 'origin manifest checksum mismatch'
        origin = json.loads(raw)
        for key in ['seed','model','kv_dtype','ctx','max_decode_tokens',
                    'prefill_query_tokens','thinking_budget','prefill_instrumentation']:
            assert origin[key] == manifest[key], f'origin protocol mismatch: {key}'
        origins[name] = {r['id']:r for r in origin['trajectories']}
    source = origins[name][row['id']]
    if row['domain'] != 'code':
        assert row['independent_unit'] == source['independent_unit'], 'non-software independence unit changed'
    # Only the documented independent-unit label and lineage annotation may change.
    normalized = lambda r: {k:v for k,v in r.items()
                             if k not in {'independent_unit','collection_cohort'}}
    assert normalized(row) == normalized(source), 'origin source/frame identity mismatch'
    return reference['sha256']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--measurements',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--require-complete',action='store_true');args=ap.parse_args()
    manifest=json.loads(args.manifest.read_text());digest=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    counts=Counter();domains=defaultdict(Counter);errors=[];intervals=defaultdict(list);verified=[];units=set();origins={}
    for row in manifest['trajectories']:
        # Resolve the underlying task independently of the manifest's displayed label.
        canonical=row['independent_unit']
        if row['domain']=='code':
            parts=Path(row['source']).parts
            canonical=parts[parts.index('pier_jobs')+1]
        unit=(row['domain'],canonical)
        if unit in units:errors.append({'trajectory':row['id'],'error':'duplicate independence unit'})
        units.add(unit);previous=None
        try:
            assert row['independent_unit']==canonical,'source task label differs from canonical task'
            expected_digest=collection_digest(row,manifest,digest,origins)
        except Exception as e:
            errors.append({'trajectory':row['id'],'error':str(e)})
            continue
        for i,frame in enumerate(row['frames']):
            path=args.measurements/row['id']/f'step_{i:03d}.json'
            if not path.exists():counts['pending']+=1;domains[row['domain']]['pending']+=1;previous=None;continue
            try:
                d=json.loads(path.read_text());counts[d['status']]+=1;domains[row['domain']][d['status']]+=1
                if d['status']!='collected':previous=None;continue
                assert d['manifest_sha256']==expected_digest,'manifest checksum mismatch'
                assert d['trajectory_id']==row['id'] and d['step']==i,'request identity mismatch'
                assert d['source_step']==frame['source_step'],'source frame mismatch'
                assert d['domain']==row['domain'],'domain mismatch'
                assert d['replay'] and not d['closed_loop'],'evidence-scope mismatch'
                with gzip.open(d['input_tokens_path'],'rt') as f:inp=json.load(f)
                assert inp['append_base']==0,'not a full-input reference'
                assert len(inp['tokens'])==d['prompt_tokens']==d['api_prompt_tokens'],'prompt length mismatch'
                if inp['mrope']:assert len(inp['mrope'])==len(inp['tokens']),'mRoPE length mismatch'
                with gzip.open(path.with_name(f'response_{i:03d}.json.gz'),'rt') as f:response=json.load(f)
                with gzip.open(path.with_name(f'request_{i:03d}.json.gz'),'rt') as f:request=json.load(f)
                assert not request.get('stop') and not request.get('stream'),'unsupported stop/stream accounting'
                assert request['max_tokens']==manifest['max_decode_tokens'],'output budget mismatch'
                assert response['usage']['completion_tokens']==d['completion_tokens'],'completion count mismatch'
                assert expected_decode_queries(d['completion_tokens'],request['max_tokens'])==d['decode_samples'],'decode query count mismatch'
                assert response['choices'][0]['finish_reason']==d['finish_reason'],'finish reason mismatch'
                assert d['prefill_samples']==min(manifest['prefill_query_tokens'],d['prompt_tokens']),'prefill sample count mismatch'
                z=np.load(d['path']);pos,mass=z['positions'],z['mass']
                np.testing.assert_array_equal(pos,np.arange(d['prompt_tokens']-d['prefill_samples'],
                    d['prompt_tokens']+d['decode_samples']))
                assert mass.shape[0]==len(pos) and np.isfinite(mass).all() and (mass>=0).all(),'invalid matrix'
                np.testing.assert_allclose(mass.sum(1),1,atol=1e-6)
                assert d['layers']==list(range(3,64,4)),'unexpected standard-attention layers'
                assert d['rows']==len(pos)*len(d['layers']),'raw row count mismatch'
                for j,position in enumerate(pos):
                    assert not mass[j,position//128+1:].any(),'future-block mass'
                assert d['trace_end']<=Path(d['trace_path']).stat().st_size,'missing raw bytes'
                assert d['trace_end']>d['trace_start'],'empty raw interval'
                intervals[d['trace_path']].append((d['trace_start'],d['trace_end'],row['id'],i))
                expected_prefix=False
                if previous is not None and frame['source_step']==previous[1]['source_step']+1:
                    oldimages=previous[1].get('image_sha256',[]);newimages=frame.get('image_sha256',[])
                    expected_prefix=not oldimages or oldimages==newimages[:len(oldimages)]
                assert d['prefix_identity_verified']==expected_prefix,'prefix eligibility mismatch'
                if d['prefix_identity_verified']:
                    assert previous is not None,'verified prefix without completed previous frame'
                    old,oldframe=previous
                    assert frame['source_step']==oldframe['source_step']+1,'nonconsecutive boundary'
                    common=prefix_length(old['tokens'],inp['tokens'])
                    if old['mrope'] and inp['mrope']:common=min(common,prefix_length(old['mrope'],inp['mrope']))
                    oldimages=oldframe.get('image_sha256',[]);newimages=frame.get('image_sha256',[])
                    assert not oldimages or oldimages==newimages[:len(oldimages)],'changed image prefix'
                    assert common==d['common_prefix_tokens']==d['history_end_tokens'],'prefix-length mismatch'
                previous=(inp,frame)
                verified.append(dict(trajectory=row['id'],step=i,prompt_tokens=d['prompt_tokens'],
                    decode_tokens=d['decode_samples'],completion_tokens=d['completion_tokens'],
                    budget_reached=d['completion_tokens']>=request['max_tokens'],
                    common_prefix_tokens=d.get('common_prefix_tokens'),
                    finish_reason=d['finish_reason']))
            except Exception as e:
                errors.append({'trajectory':row['id'],'step':i,'error':str(e)});previous=None
    for trace,spans in intervals.items():
        spans.sort()
        for a,b in zip(spans,spans[1:]):
            if a[1]>b[0]:errors.append({'trace':trace,'error':'overlapping request byte intervals'})
    server_configs=[]
    for trace in intervals:
        try:
            directory=Path(trace).parent;s=json.loads((directory/'server_manifest.json').read_text());cmd=s['command']
            option=lambda key:cmd[cmd.index(key)+1]
            assert option('--ctx')==str(manifest['ctx']),'context cap mismatch'
            assert option('--kv-dtype')==manifest['kv_dtype'],'KV dtype mismatch'
            assert option('--mtp-chain')=='0' and '--kvmem' not in cmd,'not dense non-speculative reference'
            assert Path(option('--model')).name==manifest['model']+'.gguf','serving model mismatch'
            env=s['trace_environment']
            assert env['QW3_ATTN_TRACE_PREFILL_TOKENS']==str(manifest['prefill_query_tokens']),'probe size mismatch'
            worker=directory/'qw3_vision_cpu_worker.py'
            assert hashlib.sha256(worker.read_bytes()).hexdigest()==s['vision_worker_sha256'],'worker snapshot checksum mismatch'
            server_configs.append({'directory':str(directory),'binary_sha256':s['binary_sha256'],
                'vision_worker_sha256':s['vision_worker_sha256']})
        except Exception as e:errors.append({'trace':trace,'error':str(e)})
    for key in ['binary_sha256','vision_worker_sha256']:
        if len({s[key] for s in server_configs})>1:errors.append({'error':f'{key} differs across server batches'})
    planned=sum(len(r['frames']) for r in manifest['trajectories'])
    complete=len(verified)==planned and not errors
    report={'planned_requests':planned,'status_counts':dict(counts),'domain_counts':dict(domains),
        'independence_units':len(units),'verified_requests':len(verified),'complete':complete,
        'server_configs':server_configs,'analysis_manifest_sha256':digest,
        'collection_manifests':manifest.get('collection_manifests',{}),
        'errors':errors,'checks':'Frozen manifest identity, exact input/token/mRoPE lengths, response counts, '
        'causal positions/support, all standard-attention layers, normalized finite mass, raw byte intervals, '
        'and independently recomputed common input/image prefixes.', 'requests':verified}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['requests','checks','domain_counts']}))
    if errors or (args.require_complete and not complete):raise SystemExit(1)


if __name__=='__main__':main()
