"""Read-only host receipt/input binding and conservative 035 message exposure."""
import argparse
from collections import defaultdict,Counter
import hashlib,json,re
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(text):
    return ' '.join(str(text).split())


def observation_inputs(body):
    """Only actual tool observation text, never model claims or user prompts."""
    result=[]
    for m in body.get('messages',[]):
        if m.get('role')!='tool' or not isinstance(m.get('content'),list):continue
        for p in m['content']:
            if p.get('type')!='text':continue
            text=p.get('text','')
            start=text.find('{')
            if start<0:continue
            try:value=json.loads(text[start:])
            except ValueError:continue
            if not isinstance(value,dict):continue
            observation=value.get('observation',value)
            if not isinstance(observation,dict) or not observation.get('observation_id'):continue
            controls=observation.get('accessibility',{}).get('controls',[])
            result.append({'observation_id':observation['observation_id'],'image_sha256':observation.get('sha256'),
                           'accessibility_status':observation.get('accessibility',{}).get('status'),
                           'control_names':[norm(c.get('name','')) for c in controls if isinstance(c,dict)]})
    return result


def analyze(directory):
    directory=Path(directory)
    receipt_path=directory/'relay/requests.jsonl'
    rows=[json.loads(line) for line in receipt_path.read_text().splitlines()]
    forwards=[r for r in rows if r['event']=='request_forwarding']
    terminals=defaultdict(list)
    for r in rows:
        if r['event']=='request_terminal':terminals[r['request_index']].append(r)
    inputs=defaultdict(list)
    for path in (directory/'guest/app/inference').glob('*/request.json'):
        body=json.loads(path.read_text())
        key=hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        inputs[key].append({'trace_id':path.parent.name,'request_sha256':digest(path),
                            'observations':observation_inputs(body)})
    bindings=[]
    for row in forwards:
        candidates=inputs.get(row.get('canonical_request_sha256'),[])
        if row.get('backend_trace_tag'):
            candidates=[c for c in candidates if c['trace_id']==row['backend_trace_tag']]
        bound=candidates[0] if len(candidates)==1 else None
        terminal=terminals[row['request_index']]
        bindings.append({'request_index':row['request_index'],
                         'host_received_monotonic_ns':row['host_received_monotonic_ns'],
                         'receipt_inspection_ms':row.get('receipt_inspection_ns',0)/1e6,
                         'canonical_request_sha256':row.get('canonical_request_sha256'),
                         'binding_status':'unique' if bound else ('ambiguous' if candidates else 'missing'),
                         'trace_id':bound['trace_id'] if bound else None,
                         'request_file_sha256':bound['request_sha256'] if bound else None,
                         'terminal_status':terminal[0]['status'] if len(terminal)==1 else 'unknown',
                         'latest_image_sha256':row.get('latest_image_sha256'),
                         'latest_image_matches_observation':bool(bound and row.get('latest_image_sha256') and any(o['image_sha256']==row['latest_image_sha256'] for o in bound['observations'])),
                         'observations':bound['observations'] if bound else []})
    bindings.sort(key=lambda b:b['host_received_monotonic_ns'])
    batches=[]
    for path in sorted((directory/'reviews').glob('*/delivery/receipt.json')):
        receipt=json.loads(path.read_text()); batch=receipt['batch']; _,schedule=__import__('osworld_windows.events035',fromlist=['schedule']).schedule()
        delivered=receipt.get('host_confirmed_monotonic_ns')
        members=[e for e in schedule['batches'][batch-1] if e['messageId'] in receipt['message_ids']]
        after=[b for b in bindings if delivered and b['host_received_monotonic_ns']>=delivered]
        exposures=[]
        for message in members:
            found=None
            for b in after:
                for obs in b['observations']:
                    if any(norm(message['content']) in text for text in obs['control_names']):
                        found={'host_request_index':b['request_index'],'trace_id':b['trace_id'],
                               'observation_id':obs['observation_id'],'image_sha256':obs['image_sha256'],
                               'host_first_input_monotonic_ns':b['host_received_monotonic_ns'],
                               'delivery_to_first_input_seconds':(b['host_received_monotonic_ns']-delivered)/1e9}
                        break
                if found:break
            exposures.append({'message_id':message['messageId'],'exact_full_message_in_accessibility_input':found is not None,
                              'first_exposure':found,'pixel_visibility_reviewed':False,'model_reading_or_understanding_proven':False})
        batches.append({'batch':batch,'delivery_receipt_sha256':digest(path),'delivery_confirmed':receipt.get('delivery_confirmed') is True,
                        'host_confirmed_monotonic_ns':delivered,'first_host_request_after_delivery':after[0]['request_index'] if after else None,
                        'message_exposures':exposures})
    # Control names may contain task content; retain only IDs and exposure proof.
    for b in bindings:
        b['observation_ids']=[o['observation_id'] for o in b.pop('observations')]
    return {'kind':'host_relay_input_binding_analysis','formal_run':False,'receipt_sha256':digest(receipt_path),
            'request_count':len(forwards),'terminal_count':sum(map(len,terminals.values())),
            'receipt_integrity_passed':bool(bindings) and len({b['request_index'] for b in bindings})==len(bindings) and set(terminals)=={b['request_index'] for b in bindings} and all(len(v)==1 for v in terminals.values()),
            'binding_counts':dict(Counter(b['binding_status'] for b in bindings)),
            'all_forwarded_inputs_uniquely_bound':bool(bindings) and all(b['binding_status']=='unique' for b in bindings),
            'limitations':['Host request receipt is an upper bound on model-input availability, not proof of reading or understanding.',
                          'Exact full message in tool accessibility text proves textual input exposure; pixel visibility requires separate review.',
                          'Absent text matches remain unknown because image-only exposure is not detected.',
                          'Guest and host monotonic clocks are never subtracted.'],
            'batches':batches,'requests':bindings}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=analyze(args.directory)
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ['request_count','binding_counts','all_forwarded_inputs_uniquely_bound']}))
