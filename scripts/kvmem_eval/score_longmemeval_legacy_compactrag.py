#!/usr/bin/env python3
"""Use the existing judge, with explicit unsolved accounting for pipeline failures."""
import fcntl
import argparse
import getpass
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import score_qwen35_longmemeval_deepseek as judge
import run_longmemeval_9b_legacy_compactrag as reader

OUT = reader.DATA
ROOT = OUT/'judge_deepseek_v41_flash_v2'


def terminal_jobs(out, samples):
    jobs = []
    for sample in samples:
        pair = []
        for policy in ('kvmem','compact_rag'):
            path = out/'trials'/sample['question_id']/policy/'result.json'
            if not path.exists():break
            result = json.loads(path.read_text())
            failure = result.get('status') == 'failed'
            if failure:
                assert result.get('terminal_failure') and result.get('failure_kind') == 'model_generation'
                response_path = Path(result['failure_response'])
                assert response_path.resolve().parent == path.resolve().parent
                assert judge.sha(response_path) == result['failure_response_sha256']
                try:
                    reader.checked_content(json.loads(response_path.read_text()),
                                           result['failure_stage'],response_path)
                except reader.ModelGenerationFailure as exc:
                    assert exc.reason == result['failure_reason']
                else:raise AssertionError('Failure record is not supported by response evidence')
            elif result.get('status') != 'completed':break
            pair.append({'question_id':sample['question_id'],'policy':policy,
                'prediction_sha256':judge.sha(path),'answer':result['answer'],
                'pipeline_failure':failure,'failure_reason':result.get('failure_reason')})
        if len(pair)==2:jobs.extend(pair)
    return sorted(jobs,key=lambda j:judge.digest('blind-judge-20260915:'+j['question_id']+':'+j['policy']))


def export(batch, rows, planned):
    summary = judge.export(OUT,ROOT,batch,rows,planned)
    summary['response_models'] = sorted({r['response_model'] for r in rows if not r.get('pipeline_failure')})
    summary['notes'] += ['Terminal model-generation failures are unsolved, remain in the denominator, and are not sent to the judge.',
        'Transport interruptions are not scored as model failures. Existing 9B KVMem answers/grades are reused references.']
    for method in summary['methods']:
        failures = sum(bool(r.get('pipeline_failure')) for r in rows if r['policy']==method['policy'])
        method['pipeline_failures'] = failures
        method['answers_generated'] = method['n']-failures
        method['accuracy_among_generated_answers'] = method['correct']/(method['n']-failures) if method['n']>failures else None
    for p in (batch,ROOT/'report'):
        judge.save(p/'summary.json',summary)
        judge.save(p/'pipeline_failures.json',[r for r in rows if r.get('pipeline_failure')])
    return summary


def main():
    global OUT, ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=OUT)
    args = parser.parse_args()
    OUT = args.out.resolve(); ROOT = OUT/'judge_deepseek_v41_flash_v2'
    ROOT.mkdir(parents=True,exist_ok=True)
    lock = (ROOT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    key = getpass.getpass('DeepSeek API key (hidden): ')
    client = judge.Client(key.strip())
    status = {'status':'connecting','pid':os.getpid(),'started_at':time.time(),'planned_pairs':102}
    judge.save(ROOT/'status.json',status)
    try:
        identities = {'script':judge.sha(Path(__file__)), 'shared_judge':judge.sha(Path(judge.__file__)),
            'reader_validation':judge.sha(Path(reader.__file__)), 'official_prompt':judge.sha(judge.OFFICIAL),
            'parameters':judge.PARAMETERS, 'references':judge.sha(OUT/'references.json'),
            'failure_policy':'verified terminal model failures count unsolved; never judged as valid answers'}
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==identities
        else:judge.save(ROOT/'identity.json',identities)
        models,_ = client.request('/models')
        assert judge.PARAMETERS['model'] in [m['id'] for m in models['data']]
        calibration = json.loads((OUT/'judge_deepseek_v41_flash/calibration.json').read_text())
        old_identity = json.loads((OUT/'judge_deepseek_v41_flash/identity.json').read_text())
        assert calibration['passed'] and old_identity['parameters']==judge.PARAMETERS
        assert old_identity['official_prompt']==judge.sha(judge.OFFICIAL)
        judge.save(ROOT/'calibration.json',calibration)
        check = judge.prompt_function()
        samples = json.loads((OUT/'inference_samples.json').read_text())
        refs = {r['question_id']:r for r in json.loads((OUT/'references.json').read_text())}
        previous = None
        while True:
            jobs = terminal_jobs(OUT,samples);signature=judge.digest(jobs)
            if jobs and signature != previous:
                batch = ROOT/'runs'/f'{time.time_ns()}_{len(jobs)//2}_pairs'
                judge.save(batch/'selection.json',{'created_at':time.time(),'predictions':jobs})
                status.update(status='grading',selected_pairs=len(jobs)//2,snapshot=str(batch))
                judge.save(ROOT/'status.json',status)
                def grade(job):
                    qid,policy=job['question_id'],job['policy'];p=ROOT/'scores'/f'{qid}_{policy}.json'
                    if p.exists():
                        row=json.loads(p.read_text());assert row['prediction_sha256']==job['prediction_sha256']
                        assert bool(row.get('pipeline_failure'))==job['pipeline_failure']
                        return row
                    common={'question_id':qid,'question_type':refs[qid]['question_type'],
                        'policy':policy,'prediction_sha256':job['prediction_sha256'],'hypothesis':job['answer']}
                    if job['pipeline_failure']:
                        row={**common,'correct':False,'pipeline_failure':True,'failure_reason':job['failure_reason'],
                             'response_model':'not_called_pipeline_failure','raw_judge_response':'not_called',
                             'score_basis':'pipeline_failed_to_produce_answer'}
                    else:
                        old=OUT/'judge_deepseek_v41_flash/scores'/p.name
                        if old.exists():
                            row=json.loads(old.read_text());assert row['prediction_sha256']==job['prediction_sha256']
                        else:
                            label,raw,request_sha=judge.assess(client,check,refs[qid],job['answer'],
                                                   batch/'requests'/judge.digest(qid+':'+policy))
                            row={**common,'correct':label,'request_sha256':request_sha,
                                'response_model':raw['model'],'response_id':raw.get('id'),
                                'raw_judge_response':raw['choices'][0]['message']['content'],
                                'usage':raw.get('usage'),'system_fingerprint':raw.get('system_fingerprint')}
                    assert judge.sha(OUT/'trials'/qid/policy/'result.json')==job['prediction_sha256']
                    judge.save(p,row);return row
                with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(grade,jobs))
                summary=export(batch,rows,len(samples));previous=signature
                status.update(scored_pairs=summary['scored_pairs'],updated_at=time.time())
            if len(jobs)==2*len(samples):
                status.update(status='complete',finished_at=time.time());break
            inference=json.loads((OUT/'status.json').read_text())
            if inference['status']=='needs_attention':raise RuntimeError('Reader needs attention; completed scores preserved')
            status.update(status='waiting_for_terminal_pairs');judge.save(ROOT/'status.json',status)
            time.sleep(30)
    except Exception as exc:
        status.update(status='needs_attention',error=str(exc).replace(client.key,'[REDACTED]'),finished_at=time.time())
        print(status['error'],file=sys.stderr,flush=True)
        return 1
    finally:judge.save(ROOT/'status.json',status)
    return 0


if __name__=='__main__':
    sys.exit(main())
