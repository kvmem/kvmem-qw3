"""Prepare a public native task launcher without staging host-only evaluators."""
import json
from pathlib import Path
import re
import shutil

from .source import ROOT, sha256
from .capabilities import TASKS as CAPABILITY_TASKS


def prepare(task, run, prompt, collection, bundle, staged, output, *, timeout, tool_limit, backend):
    if (task not in {'013','004','006','010','038','022','035','043','080','051', *CAPABILITY_TASKS}
            or not re.fullmatch(r'[a-zA-Z0-9_-]{8,60}',run) or not isinstance(prompt,str) or not prompt.strip()
            or type(timeout) is not int or not 1 <= timeout <= 14400
            or type(tool_limit) is not int or not 1 <= tool_limit <= 10000
            or not re.fullmatch(r'http://[0-9.]+:18084/v1',backend)):
        raise ValueError('Invalid native task launch configuration')
    archive = json.loads((Path(bundle) / 'archive.json').read_text())
    if not re.fullmatch(r'[0-9a-f]{64}',archive['sha256']):
        raise ValueError('Malformed native bundle identity')
    output = Path(output);output.mkdir(parents=True,exist_ok=False)
    staged = Path(staged)
    if staged.name != run+'-inputs' or not staged.is_dir():
        raise ValueError('Launcher must use the current attempt public staging directory')
    with (staged/'prompt.txt').open('x',encoding='utf-8') as handle:handle.write(prompt)
    with (staged/'collection.json').open('x',encoding='utf-8') as handle:json.dump(collection,handle,indent=2)
    for name in ('client_resources.ps1','run_native_task.ps1'):
        destination=staged/name
        if destination.exists():raise ValueError('Launcher helper already staged')
        shutil.copyfile(ROOT/'apps/computer_use/vm'/name,destination)
    configuration = dict(run=run,task=task,bundle_sha256=archive['sha256'],backend=backend,
        timeout_seconds=timeout,tool_call_limit=tool_limit,review_batches=3 if task=='035' else 0,
        public_sha256={name:sha256(staged/name) for name in ('prompt.txt','collection.json','client_resources.ps1')})
    with (staged/'native-task.json').open('x',encoding='utf-8') as handle:json.dump(configuration,handle,indent=2)
    script = ("$ErrorActionPreference='Stop'\n"
              f"$script='\\\\host.lan\\Data\\{run}-inputs\\run_native_task.ps1'\n"
              f"if ((Get-FileHash $script -Algorithm SHA256).Hash.ToLowerInvariant() -cne '{sha256(staged/'run_native_task.ps1')}') {{throw 'Native launcher hash mismatch'}}\n"
              f"& $script -RunId '{run}'\n")
    (output/'launch.ps1').write_text(script)
    (output/'manifest.json').write_text(json.dumps(configuration,indent=2))
    return configuration
