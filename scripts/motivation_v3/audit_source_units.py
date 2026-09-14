"""Verify cohort units against source content and recorded task lists."""
import argparse
from collections import Counter,defaultdict
import hashlib
import itertools
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text());hashes={};dialogues={};units=set();rows=[];errors=[]
    gui_instructions=defaultdict(list)
    def digest(path):
        path=Path(path)
        if str(path) not in hashes:
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
            hashes[str(path)]=h.hexdigest()
        return hashes[str(path)]
    for row in manifest['trajectories']:
        try:
            source=Path(row['source']);assert digest(source)==row['source_sha256'],'source checksum mismatch'
            detail={}
            if row['domain']=='code':
                parts=source.parts;canonical=parts[parts.index('pier_jobs')+1]
            elif row['domain']=='tool_state':
                with source.open() as stream:
                    record=json.loads(next(itertools.islice(stream,row['source_line'],row['source_line']+1)))
                canonical=str(record['sample_id'])
            elif row['domain']=='dialogue':
                if str(source) not in dialogues:
                    with source.open() as stream:dialogues[str(source)]=json.load(stream)
                record=dialogues[str(source)][row['source_row']]
                canonical=hashlib.sha256(json.dumps(record['haystack_sessions'],sort_keys=True).encode()).hexdigest()
                assert row['id']=='dialogue_'+str(record['question_id']),'dialogue question identity mismatch'
            elif row['domain']=='gui':
                directory=source.parent;canonical=directory.name
                args_path=next(a/'args.json' for a in directory.parents if (a/'args.json').is_file())
                recorded=json.loads(args_path.read_text());meta=Path(recorded['test_all_meta_path'])
                task_list=json.loads(meta.read_text())
                assert canonical in task_list['tasks'],'GUI folder is not a recorded task ID'
                frame=row['frames'][0];message_path=Path(frame['message_file'])
                assert digest(message_path)==frame['message_sha256'],'GUI request checksum mismatch'
                messages=json.loads(message_path.read_text());user=next(m for m in messages if m['role']=='user')
                content=user['content'];text=content if isinstance(content,str) else '\n'.join(b.get('text','') for b in content if b.get('type')=='text')
                instruction=text.split('Instruction:',1)[1].split('\n\n',1)[0].strip()
                assert instruction,'empty GUI instruction'
                instruction_sha=hashlib.sha256(instruction.encode()).hexdigest();gui_instructions[instruction_sha].append(row['id'])
                detail={'task_meta':str(meta),'task_meta_sha256':digest(meta),'instruction_sha256':instruction_sha}
            else:raise ValueError('Unknown domain')
            assert canonical==row['independent_unit'],'declared unit differs from source unit'
            unit=(row['domain'],canonical);assert unit not in units,'duplicate source unit';units.add(unit)
            rows.append({'trajectory':row['id'],'domain':row['domain'],'canonical_unit':canonical,
                         'source_sha256':hashes[str(source)],**detail})
        except Exception as error:errors.append({'trajectory':row['id'],'error':str(error)})
    repeated={k:v for k,v in gui_instructions.items() if len(v)>1}
    if repeated:errors.append({'error':'GUI instruction fingerprints require duplicate-task review','groups':repeated})
    result={'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            'verified_units':len(rows),'domain_counts':dict(Counter(r['domain'] for r in rows)),
            'errors':errors,'units':rows,'source_hashes':hashes,
            'scope':'Distinct recorded source units, not proof of statistical independence. GUI instruction fingerprints cover the instruction first paragraph; task IDs are separately checked against recorded task lists.'}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['units','source_hashes','scope']}))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
