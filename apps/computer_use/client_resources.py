"""Validate PID/creation-time ownership before reporting Windows client counters."""
from datetime import datetime, timezone
import math


def birth_ticks(value):
    if not isinstance(value, str) or not value.isdecimal():
        raise ValueError('Missing decimal-string process creation ticks')
    ticks = int(value)
    if not 0 < ticks < 2**63 or ticks % 10:
        raise ValueError('Invalid normalized process creation ticks')
    return ticks


def timestamp_ticks(value):
    instant = datetime.fromisoformat(value)
    if instant.utcoffset() is None:
        raise ValueError('Process timestamp has no timezone')
    delta = instant.astimezone(timezone.utc) - datetime(1, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 10_000_000 + delta.microseconds * 10


def process_id(value):
    if type(value) is not int or value <= 0:
        raise ValueError('Invalid process ID')
    return value


def validate_sample(sample):
    if sample.get('schema_version') != 2:
        raise ValueError('Legacy sample lacks process creation identities')
    if sample.get('error') or sample.get('ownership_verified') is not True or sample.get('snapshot_complete') is not True:
        raise ValueError('Incomplete or unverified process snapshot')
    if sample.get('omitted_processes') != []:
        raise ValueError('Process counters were omitted')
    root = process_id(sample['app_pid'])
    root_birth = birth_ticks(sample['app_birth_ticks'])
    if timestamp_ticks(sample['app_start_time_utc']) != root_birth:
        raise ValueError('Root timestamp differs from creation identity')
    processes = {}
    for process in sample['processes']:
        pid = process_id(process['pid'])
        if pid in processes:
            raise ValueError('Duplicate PID in one process snapshot')
        birth = birth_ticks(process['birth_ticks'])
        if any(timestamp_ticks(process[k]) != birth for k in ('start_time_utc', 'sampled_start_time_utc')):
            raise ValueError('CIM/counter process creation identities differ')
        for key in ('working_set_bytes', 'peak_working_set_bytes', 'private_bytes', 'cpu_seconds'):
            v = process[key]
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                raise ValueError('Invalid process counter: ' + key)
            if key != 'cpu_seconds' and type(v) is not int:
                raise ValueError('Memory counters must be integer bytes')
        processes[pid] = process
    if root not in processes or birth_ticks(processes[root]['birth_ticks']) != root_birth:
        raise ValueError('Root identity missing from measured process set')
    if processes[root]['parent_birth_ticks'] is not None or processes[root]['parent_start_time_utc'] is not None:
        raise ValueError('Root parent is outside the measured client tree')
    for pid, process in processes.items():
        if pid == root:
            continue
        parent = processes.get(process_id(process['parent_pid']))
        if parent is None:
            raise ValueError('Missing measured parent identity')
        expected = birth_ticks(parent['birth_ticks'])
        if birth_ticks(process['parent_birth_ticks']) != expected or timestamp_ticks(process['parent_start_time_utc']) != expected:
            raise ValueError('Parent creation identity mismatch')
        if birth_ticks(process['birth_ticks']) < expected:
            raise ValueError('Child predates reused parent PID')
        cursor, seen = pid, set()
        while cursor != root:
            if cursor in seen:
                raise ValueError('Process ancestry cycle')
            seen.add(cursor)
            if cursor not in processes:
                raise ValueError('Process ancestry does not reach client root')
            cursor = processes[cursor]['parent_pid']
    for edge in sample['rejected_edges']:
        if edge['reason'] != 'child_predates_current_parent':
            raise ValueError('Unknown process ancestry was excluded')
        parent = processes.get(process_id(edge['parent_pid']))
        if edge['pid'] in processes or parent is None:
            raise ValueError('Rejected ancestry contradicts measured set')
        if birth_ticks(edge['parent_birth_ticks']) != birth_ticks(parent['birth_ticks']) or birth_ticks(edge['birth_ticks']) >= birth_ticks(parent['birth_ticks']):
            raise ValueError('Invalid excluded PID reuse edge')
    return root, root_birth


def summarize_client_resources(samples):
    result = {'client_samples': len(samples), 'client_resource_sample_errors': 0,
              'client_resource_comparison_valid': False, 'client_resource_errors': [],
              'client_sum_working_sets_peak_bytes': None, 'client_sum_private_bytes_peak': None,
              'client_sampled_cpu_seconds': None}
    roots = set()
    for number, sample in enumerate(samples):
        try:
            roots.add(validate_sample(sample))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            result['client_resource_errors'].append({'sample': number, 'error': str(exc)})
    result['client_resource_sample_errors'] = len(result['client_resource_errors'])
    if not samples or result['client_resource_errors']:
        return result
    if len(roots) != 1:
        result['client_resource_errors'].append({'error': 'Client root identity changed between samples'})
        return result
    # Shared DLL pages may occur in multiple working sets. These are sums of
    # counters, not unique physical RAM. CPU covers only sampled lifetimes.
    result['client_sum_working_sets_peak_bytes'] = max(sum(p['working_set_bytes'] for p in s['processes']) for s in samples)
    result['client_sum_private_bytes_peak'] = max(sum(p['private_bytes'] for p in s['processes']) for s in samples)
    cpu = {}
    for sample in samples:
        for process in sample['processes']:
            key = (process['pid'], process['birth_ticks'])
            cpu[key] = max(cpu.get(key, 0), process['cpu_seconds'])
    result['client_sampled_cpu_seconds'] = sum(cpu.values())
    result['client_resource_comparison_valid'] = True
    return result
