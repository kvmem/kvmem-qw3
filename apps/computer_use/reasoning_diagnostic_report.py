"""Audit frozen P2 requests and report paired outcomes without rewriting scores."""
import argparse
import json
from pathlib import Path
import statistics

from reasoning_diagnostic_run import grade,request_for,validate
from visual_diagnostic_run import digest,save


def report(root,corpus_path):
    data=json.loads((root/'result.json').read_text());corpus=json.loads(corpus_path.read_text());samples={s['id']:s for s in validate(corpus,corpus_path.parent)}
    if data['status']!='complete' or len(data['trials'])!=96 or digest(corpus_path)!=data['corpus_sha256']:raise ValueError('P2 not complete or corpus identity changed')
    for i,trial in enumerate(data['trials']):
        directory=root/f'{i:02d}-{trial["sample"]}';request=json.loads((directory/'request.json').read_text())
        expected=request_for(samples[trial['sample']],corpus_path.parent,trial['representation'],trial['thinking'],4096)
        if request!=expected or digest(directory/'request.json')!=trial['request_sha256']:raise ValueError('Request identity differs from frozen condition')
        actual_reasoning='';actual_answer='';usage=None
        for line in (directory/'stream.jsonl').read_text().splitlines():
            event=json.loads(line)
            if event.get('usage'):usage=event['usage']
            for choice in event.get('choices',[]):
                delta=choice.get('delta',{});actual_reasoning+=delta.get('reasoning_content') or '';actual_answer+=delta.get('content') or ''
        if actual_answer!=trial['answer'] or bool(actual_reasoning)!=trial['thinking'] or usage!=trial['usage']:raise ValueError('Stored answer or actual reasoning mode mismatch')
        scored=grade(samples[trial['sample']],actual_answer)
        if trial.get('budget_exhausted'):scored['passed']=False
        if scored!=trial['score']:raise ValueError('Stored grading differs from frozen rules')
    groups={}
    for representation in ('screenshot','text'):
        for thinking in (False,True):
            ts=[t for t in data['trials'] if t['representation']==representation and t['thinking']==thinking]
            groups[representation+'-'+('on' if thinking else 'off')]={
                'requests':len(ts),'passed':sum(t['score']['passed'] for t in ts),
                'median_seconds':statistics.median(t['elapsed_seconds'] for t in ts),
                'median_completion_tokens':statistics.median(t['usage']['completion_tokens'] for t in ts),
                'budget_exhausted':sum(t.get('budget_exhausted',False) for t in ts),
                'format_failures':sum(not t['score']['format_valid'] for t in ts),
                'splits':{split:{'passed':sum(t['score']['passed'] for t in ts if t['split']==split),
                    'requests':sum(t['split']==split for t in ts)} for split in ('development','heldout')}}
    pairs=[]
    for identity in samples:
        cells={}
        for t in data['trials']:
            if t['sample']==identity:cells.setdefault(t['representation']+'-'+('on' if t['thinking'] else 'off'),[]).append(t['score']['passed'])
        pairs.append({'sample':identity,'split':samples[identity]['split'],'cells':cells})
    result={'audited':True,'request_count':96,'result_sha256':digest(root/'result.json'),'corpus_sha256':digest(corpus_path),
            'groups':groups,'pairs':pairs,'limits':['12 neutral short-context rules, two repeated requests per cell; not 96 independent tasks.',
            'No self-correction rate: final answers only; reasoning content retained for mode validation.',
            'Task facts are equivalent; screenshot contains extra non-task desktop chrome and advertisement.']}
    save(root/'audit.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--corpus',type=Path,required=True)
    args=parser.parse_args();result=report(args.run,args.corpus);print(json.dumps({'audited':result['audited'],'groups':result['groups']},ensure_ascii=False,indent=2))
