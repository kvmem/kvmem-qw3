"""Freeze distinct source trajectories before measuring attention.

These are replay inputs from existing data, not new closed-loop benchmark runs.
No generated action is executed and no success rate is inferred from replay.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
ALB=Path('/home/chaidi/kvmem_eval/KVMem_Motivation/data/raw/AgentLongBench/benchmark')
LME=Path('/data/chaidi/kvmem_eval/data/longmemeval_s.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_message(m):
    keys=('role','content','tool_calls','tool_call_id','name','reasoning_content')
    return {k:m[k] for k in keys if k in m and m[k] is not None}


def replay_cuts(messages, nsteps=12, min_chars=4000, max_chars=190000):
    """Choose the first consecutive eligible assistant-prediction window."""
    cuts=[];chars=0;turn=0
    for i,m in enumerate(messages):
        if m.get('role')=='assistant':
            if min_chars<=chars<=max_chars:
                cuts.append({'cut':i,'source_step':turn,'input_chars':chars})
                if len(cuts)==nsteps:break
            elif chars>max_chars:
                break
            turn+=1
        chars+=len(json.dumps(clean_message(m),ensure_ascii=False))
    return cuts


def pick(candidates,n,rng):
    candidates=sorted(candidates,key=lambda x:x['id'])
    rng.shuffle(candidates)
    return candidates[:n]


def prepare_code(n,nsteps,rng):
    groups={}
    for p in sorted((ROOT/'benchmark/mini_swe_deepswe/recordings').rglob('mini-swe-agent.trajectory.json')):
        parts=p.parts
        if 'pier_jobs' not in parts:continue
        # An attempt_N directory can precede pier_jobs; the task is its child.
        task=parts[parts.index('pier_jobs')+1]
        groups.setdefault(task,[]).append(p)
    candidates=[]
    for task,paths in sorted(groups.items()):
        # Source selection is file-order/length eligibility, never task outcome.
        for p in paths:
            try:d=json.loads(p.read_text())
            except (ValueError,OSError):continue
            cuts=replay_cuts(d.get('messages',[]),nsteps)
            if len(cuts)<8:continue
            candidates.append(dict(id='code_'+task,domain='code',source_kind='mini_swe_replay',
                independent_unit=task,source=str(p),source_sha256=sha(p),frames=cuts,
                replay=True,closed_loop=False))
            break
    return pick(candidates,n,rng),len(candidates)


def prepare_tool(n,nsteps,rng):
    pools=[]
    for variant in ['kf-c','ki-c','kf-v','ki-v']:
        seen=set();candidates=[]
        for p in sorted((ALB/variant/'32k').rglob('*.jsonl')):
            with p.open() as f:
                for line_index,line in enumerate(f):
                    d=json.loads(line);sid=str(d['sample_id'])
                    if sid in seen:continue
                    cuts=replay_cuts(d['messages'],nsteps)
                    if len(cuts)<8:continue
                    seen.add(sid)
                    candidates.append(dict(id=f'tool_{variant}_{sid}',domain='tool_state',
                        source_kind='agentlongbench_replay',independent_unit=sid,
                        source=str(p),source_line=line_index,frames=cuts,
                        replay=True,closed_loop=False,variant=variant))
        pools.append(pick(candidates,len(candidates),rng))
    # Masked/unmasked and concise/verbose versions can share the base episode.
    # Sample distinct episode IDs, and redistribute quotas before measurement
    # when a variant lacks eight successive usable steps.
    selected=[];used=set();available=sum(map(len,pools))
    while len(selected)<n and any(pools):
        for pool in pools:
            while pool and pool[0]['independent_unit'] in used:pool.pop(0)
            if pool and len(selected)<n:
                row=pool.pop(0);selected.append(row);used.add(row['independent_unit'])
    hashes={p:sha(p) for p in {x['source'] for x in selected}}
    for x in selected:x['source_sha256']=hashes[x['source']]
    return selected,available


def dialogue_messages(row):
    messages=[{'role':'system','content':'You are a helpful assistant.'}]
    for session in row['haystack_sessions']:
        messages.extend(clean_message(m) for m in session)
    return messages


def prepare_dialogue(n,nsteps,rng):
    rows=json.loads(LME.read_text());seen=set();candidates=[]
    for i,row in enumerate(rows):
        messages=dialogue_messages(row)
        identity=hashlib.sha256(json.dumps(row['haystack_sessions'],sort_keys=True).encode()).hexdigest()
        if identity in seen:continue
        cuts=replay_cuts(messages,nsteps)
        if len(cuts)<8:continue
        seen.add(identity)
        candidates.append(dict(id='dialogue_'+str(row['question_id']),domain='dialogue',
            source_kind='longmemeval_conversation_replay',independent_unit=identity,
            source=str(LME),source_row=i,frames=cuts,replay=True,closed_loop=False,
            source_question_type=row['question_type']))
    selected=pick(candidates,n,rng);h=sha(LME)
    for x in selected:x['source_sha256']=h
    return selected,len(candidates)


def prepare_gui(n,nsteps,rng):
    candidates={}
    for p in sorted((ROOT/'benchmark/osworld_v2/results').rglob('agent_step_trace.jsonl')):
        task=p.parent.name
        try:rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        except (ValueError,OSError):continue
        image_map={};frames=[]
        for row in rows:
            q=row.get('request',{});image=q.get('input_file');h=q.get('input_sha256')
            if image and h and (p.parent/image).is_file():image_map[h]=str(p.parent/image)
        for row in rows:
            q=row.get('request',{});name=q.get('message_file')
            if not name or not (p.parent/name).is_file():continue
            if row.get('status')!='success':continue
            m=json.loads((p.parent/name).read_text())
            needed=[]
            for msg in m:
                if not isinstance(msg.get('content'),list):continue
                needed.extend(b.get('image_url',{}).get('sha256') for b in msg['content'] if b.get('type')=='image_url')
            if any(h not in image_map for h in needed):continue
            frames.append(dict(source_step=int(row['step_num']),message_file=str(p.parent/name),
                               message_sha256=sha(p.parent/name),image_sha256=needed))
        # Require an uninterrupted sequence of recorded successful requests.
        streak=[];best=[]
        for f in frames:
            if streak and f['source_step']!=streak[-1]['source_step']+1:streak=[]
            streak.append(f)
            if len(streak)>len(best):best=streak.copy()
        if len(best)<8:continue
        prev=candidates.get(task)
        if prev and prev['available_steps']>=len(best):continue
        candidates[task]=dict(id='gui_'+task,domain='gui',source_kind='osworld_gui_replay',
            independent_unit=task,source=str(p),source_sha256=sha(p),available_steps=len(best),
            frames=best[:nsteps],image_map=image_map,replay=True,closed_loop=False)
    return pick(list(candidates.values()),n,rng),len(candidates)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--per-domain',type=int,default=20);ap.add_argument('--steps',type=int,default=12)
    ap.add_argument('--seed',type=int,default=20260911);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);target=args.out/'manifest.json'
    if target.exists():raise FileExistsError('Frozen manifest already exists')
    rng=random.Random(args.seed);rows=[];counts={}
    for name,func in [('code',prepare_code),('tool_state',prepare_tool),('dialogue',prepare_dialogue),('gui',prepare_gui)]:
        selected,available=func(args.per_domain,args.steps,rng)
        counts[name]=dict(eligible_candidates=available,selected=len(selected))
        rows.extend(selected);print(name,counts[name],flush=True)
    data=dict(seed=args.seed,requested_per_domain=args.per_domain,requested_steps=args.steps,
        model='Qwen3.8-27B-Q8_0',kv_dtype='fp16',ctx=65536,max_decode_tokens=2048,
        prefill_query_tokens=64,thinking_budget=512,counts=counts,
        prefill_instrumentation='Read-only side probe over existing batched RoPE-applied Q/K, causal mask per sampled row; production batching unchanged.',
        evidence_scope='New full-visible attention measurements on recorded input prefixes. '
        'Counterfactual outputs are not executed and not appended to the next recorded prefix. '
        'No new closed-loop task success is measured. Legacy live runs remain a separate cohort.',
        planned_primary_windows=[16,32,64],additional_windows=[128],
        trajectories=rows)
    target.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    print('Wrote',target,'trajectories',len(rows),'requests',sum(len(r['frames']) for r in rows))


if __name__=='__main__':main()
