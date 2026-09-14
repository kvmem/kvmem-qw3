#!/usr/bin/env python3
"""Fixed 4B/9B pilot using the existing DeepSWE agent, services and verifier.

No task-selection, prompting, compaction or evaluator implementation is replaced.
Run --prepare-only before GPU admission; --run resumes only completed arms.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'benchmark/qwen35_small_20260914'
sys.path.insert(0, str(ROOT / 'benchmark/claude_deepswe_ab'))
import batch
import ordinary_only_batch as ordinary

REFERENCE = ROOT / ('benchmark/claude_deepswe_ab/recordings/'
    'kvmem3m_128gen64_full113_claude_pass4_pro_reuse_passmajor_s1000_4000_20260829/'
    'trials/superjson-error-stack-serialization/seed-1000/manifest.json')
Q = batch.CONDITIONS[0]
K = batch.KVMEM_128_GEN64_CONDITION
BINARY = HERE / 'frozen/qw3'
PORT = 18560
original_markers = batch.service_readiness_markers


def readiness(*args, **kwargs):
    return tuple(x.replace('Qwen3.8-27B-Q8_0.gguf', batch.MODEL.name)
                 for x in original_markers(*args, **kwargs))


batch.service_readiness_markers = readiness


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    batch.atomic_json(path, value)


def prepare():
    reference = json.loads(REFERENCE.read_text())
    catalog = json.loads((HERE / 'tasks.json').read_text())
    tasks = [{**batch.task_metadata(row['task_id']),
              'verifier_image': row['verifier_image']} for row in catalog['tasks']]
    frozen = BINARY.parent
    frozen.mkdir(parents=True, exist_ok=True)
    if not BINARY.exists():
        shutil.copy2(ROOT / 'build-qwen35-small/qw3', BINARY)
    identities = {}
    for size in ('4B', '9B'):
        model = ROOT / f'models/Qwen3.5-{size}-Q8_0-with-MTP.gguf'
        identities[size] = dict(qw3_binary_sha256=batch.sha256(BINARY),
            model_sha256=batch.sha256(model), model_size_bytes=model.stat().st_size,
            server_execution_backend='local')
        for policy, condition in (('qw3', Q), ('kvmem', K)):
            root = HERE / 'pilot' / size / policy
            manifest_path = root / 'manifest.json'
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
            else:
                manifest = copy.deepcopy(reference)
                manifest.pop('locked_artifacts', None)
                manifest.update(created_at=batch.utc_now(), experiment_type='qwen35_dense_pilot',
                    tasks_file=str(HERE / 'tasks.json'),
                    tasks_file_sha256=batch.sha256(HERE / 'tasks.json'),
                    tasks=tasks, model=str(model), condition=condition,
                    claude_model_id=f'Qwen3.5-{size}[3m]' if policy == 'kvmem' else f'Qwen3.5-{size}[1m]',
                    reference_27b_manifest=str(REFERENCE), reference_27b_manifest_sha256=batch.sha256(REFERENCE))
                manifest['ordinary'] = dict(context=262144, claude_auto_compact_window=262144,
                    expected_effective_window=242144, expected_auto_compact_threshold=229144,
                    claude_auto_compaction=True, compaction_implementation='Claude Code default',
                    prefix_cache=True)
                manifest['sampling_reference']['profile'] = 'copied_from_existing_27b_deepswe'
                manifest['execution'].update(agent_container_memory='16g',
                    order='task-major; alternate model and policy order by task index', concurrency=1)
                manifest['weight_provenance'] = dict(base=str(model).replace('-with-MTP.gguf', '.gguf.provenance.json'),
                    augmentation=str(model) + '.augmentation.json', base_q8_tensors_unchanged=True)
                manifest['precision'] = dict(weights='Q8_0', kv='FP8', mtp_chain=4)
                save(manifest_path, manifest)
            batch.rebuild_and_lock(root, manifest, external_server_artifacts=identities[size])
    jobs = []
    for index, task in enumerate(tasks):
        for size in (('4B', '9B') if index % 2 == 0 else ('9B', '4B')):
            policies = ('qw3', 'kvmem') if (index + (size == '9B')) % 2 == 0 else ('kvmem', 'qw3')
            for policy in policies:
                jobs.append(dict(task_id=task['task_id'], model=size, policy=policy,
                                 condition=Q if policy == 'qw3' else K, task_index=index + 1))
    plan = dict(created_at=batch.utc_now(), seed=1000, jobs=jobs, trials=len(jobs),
        model_artifacts=identities, binary=str(BINARY), source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        runner_sha256=batch.sha256(Path(__file__)),
        reference_27b_manifest=str(REFERENCE),
        environment=dict(QW3_BINARY_PATH=str(BINARY)),
        harness_sources={str(p): batch.sha256(p) for p in (Path(batch.__file__),
            Path(ordinary.__file__), batch.ANALYZE, batch.HERE / 'service_watchdog.py')},
        notes=['No visual input in this text pilot; small-model vision numerical parity remains under investigation.',
               'Short 8K/128K qualification timings are diagnostics, not full-trajectory results.'])
    path = HERE / 'pilot_plan.json'
    if path.exists():
        old = json.loads(path.read_text())
        for key in ('jobs', 'model_artifacts', 'runner_sha256', 'harness_sources'):
            if old[key] != plan[key]:
                raise RuntimeError(f'pilot resume identity mismatch: {key}')
        return old
    save(path, plan)
    return plan


def summarize(plan):
    rows = []
    for job in plan['jobs']:
        path = HERE / 'pilot' / job['model'] / job['policy'] / 'runs' / job['task_id'] / job['condition'] / 'metrics.json'
        if path.exists():
            rows.append({**job, 'metrics_path': str(path), 'metrics': json.loads(path.read_text())})
    save(HERE / 'pilot_results.json', dict(updated_at=batch.utc_now(), completed=len(rows),
         total=plan['trials'], rows=rows))
    return rows


def run(plan):
    # Clear proxy settings for container setup too. No model downloads occur here.
    for key in tuple(os.environ):
        if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy'):
            os.environ.pop(key)
    os.environ['QW3_BINARY_PATH'] = str(BINARY)
    state = dict(status='starting', pid=os.getpid(), total=plan['trials'], started_at=batch.utc_now())
    try:
        for job in plan['jobs']:
            root = HERE / 'pilot' / job['model'] / job['policy']
            manifest = json.loads((root / 'manifest.json').read_text())
            task = next(t for t in manifest['tasks'] if t['task_id'] == job['task_id'])
            if batch.arm_complete(root, job['task_id'], job['condition']):
                metrics = batch.read_json(root / 'runs' / job['task_id'] / job['condition'] / 'metrics.json')
                if ordinary.abnormal_reasons(metrics):
                    raise RuntimeError(f'completed arm needs audit before reuse: {job}')
                continue
            active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            if active:
                raise RuntimeError(f'GPU is already occupied: {active}')
            batch.MODEL = Path(manifest['model'])
            batch.QW3 = BINARY
            batch.publish_supervisor_owner(root)
            state.update(status='starting_service', job=job, completed=len(summarize(plan)), updated_at=batch.utc_now())
            save(HERE / 'pilot_status.json', state)
            network_policy = batch.network_policy_for_task(task)
            with batch.DockerNetwork(root, 'sample-' + str(job['task_index']) + '-' + job['task_id'],
                    allow_wan=network_policy == 'docker_bridge_wan_audited') as network:
                service = batch.Service(root, job['condition'], job['task_index'], PORT,
                    seed=manifest['seed'], sampling=manifest['sampling'], bind_host=network.gateway,
                    agent_network=network.name, context=manifest['kvmem']['context'] if job['policy'] == 'kvmem' else None,
                    kvmem_cpu_gb=manifest['kvmem']['cpu_gb'],
                    kvmem_guided_query_tokens=manifest['kvmem']['guided_query_tokens'])
                try:
                    service.wait_ready()
                    state.update(status='running', updated_at=batch.utc_now())
                    save(HERE / 'pilot_status.json', state)
                    code = batch.run_arm(root, task, job['condition'], service)
                    metrics = batch.read_json(root / 'runs' / job['task_id'] / job['condition'] / 'metrics.json', {})
                    reasons = ordinary.abnormal_reasons(metrics)
                    if (metrics.get('claude') or {}).get('private_query_leak_count', 0):
                        reasons.append('private_query_leak')
                    if code or reasons:
                        raise RuntimeError(f'arm requires investigation: exit={code}, reasons={reasons}')
                finally:
                    service.stop()
            summarize(plan)
        state.update(status='complete', completed=len(summarize(plan)), finished_at=batch.utc_now())
    except BaseException as error:
        state.update(status='needs_attention', error=repr(error), traceback=traceback.format_exc(),
                     completed=len(summarize(plan)), updated_at=batch.utc_now())
        raise
    finally:
        save(HERE / 'pilot_status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    plan = prepare()
    if args.run:
        run(plan)
    else:
        print(json.dumps(dict(status='prepared', trials=plan['trials'], path=str(HERE / 'pilot_plan.json'))))
