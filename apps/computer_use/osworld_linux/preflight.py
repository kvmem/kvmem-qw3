"""Freeze selected original task material without exposing graders to the agent."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

from osworld_windows.source import ROOT, SELECTION, scenario, asset


TASKS = ('013','004','006','010','038','022','035','043','080','051')


def preflight():
    selection = json.loads(SELECTION.read_text())
    results = []
    for task in TASKS:
        item = {'task_id':task,'ready':False,'source_verified':False,'materials_verified':False}
        try:
            source = scenario(task)
            item.update(source_file=source['source_file'],source_sha256=source['source_sha256'],source_verified=True)
            tree = ast.parse((ROOT/source['source_file']).read_text())
            cls = next(node for node in tree.body if isinstance(node,ast.ClassDef))
            item['has_independent_evaluator'] = any(isinstance(node,ast.FunctionDef) and node.name=='evaluate' for node in cls.body)
            files = []
            for key, identity in source['assets'].items():
                path = asset(task,key.removeprefix('task_'+task+'/'))
                files.append({'path':str(path.relative_to(ROOT)),'bytes':identity['bytes'],'sha256':identity['sha256']})
            item.update(materials_verified=True,materials=files,ready=item['has_independent_evaluator'])
        except Exception as exc:
            item['error'] = type(exc).__name__ + ': ' + str(exc)
        results.append(item)
    return {'kind':'linux-osworld-material-preflight-not-agent-results','created_at':time.time(),
            'source':selection['source'],'tasks':results,'ready_count':sum(t['ready'] for t in results),
            'not_verified':['guest application versions','website rendering and fresh per-attempt state',
                            'task setup/evaluator positive controls','agent task success'],
            'policy':'Never replace failed preflight tasks with easier samples; retain them as infrastructure entries.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():raise SystemExit('Refusing to overwrite an existing preflight')
    result = preflight()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'ready_materials':result['ready_count'],'tasks':len(TASKS),'formal_attempts':0}))


if __name__=='__main__':main()
