"""Focused diagnostics after the user paused the seven desktop cohorts.

Build a separate wire candidate from the actual image-history framework, then
replay recorded evidence. Never alter a frozen cohort, score or live desktop.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'apps/computer_use/.runtime'
SOURCE = RUNTIME / 'visual-context-retest-20261004'
ACCEPTANCE_SHA = '7463873601c60dad2bd0782bd13215685206c1d07355927baed8734974645e73'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def build(output):
    import condition_schema_candidate as wire
    acceptance = SOURCE / 'acceptance.json'
    if digest(acceptance) != ACCEPTANCE_SHA:
        raise ValueError('Frozen acceptance changed')
    manifest = json.loads(acceptance.read_text())
    source = SOURCE / 'framework'
    files = {name: sha for name, sha in manifest['framework_files'].items()
             if name.startswith('qw3_computer/')}
    contents = {}
    for name, sha in files.items():
        path = source / name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or digest(path) != sha:
            raise ValueError('Frozen inventory changed: ' + name)
        contents[name] = path.read_bytes()
    if 'qw3_computer' in sys.modules:
        raise ValueError('Select the private source before package imports')
    sys.path.insert(0, str(source))
    from qw3_computer.tool_schema import WORKFLOW_TOOLS
    declaration = WORKFLOW_TOOLS[0]['inputSchema']['properties']['conditions']
    sha = hashlib.sha256(json.dumps(declaration, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    helper = Path(__file__).with_name('condition_schema_wire.py').read_text()
    old = "CONDITION_SHA256 = 'a1dc0d942bb1107f8728e47644fcb0d507bcfc294813de6a72c60293c78d21c6'"
    if helper.count(old) != 1:
        raise ValueError('Trusted helper anchor changed')
    helper = helper.replace(old, 'CONDITION_SHA256 = ' + repr(sha))
    # The observed pinned client also preserves workflow root properties while
    # replacing each oneOf branch with {}. Restore this exact known shape only.
    helper += '''

_restore_act_conditions = restore_inline_conditions

def restore_inline_conditions(name, parameters, declared_conditions):
    if name != 'mcp__computer__workflow':
        return _restore_act_conditions(name, parameters, declared_conditions)
    result = copy.deepcopy(parameters)
    digest = hashlib.sha256(json.dumps(declared_conditions, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    if digest != CONDITION_SHA256:
        raise ValueError('Trusted condition declaration changed')
    properties = result.get('properties')
    if not isinstance(properties, dict):
        raise ValueError('Workflow lost its object fields')
    conditions = properties.get('conditions')
    if not isinstance(conditions, dict) or conditions.get('type') != 'array':
        raise ValueError('Workflow lost its condition array')
    normalized = codex_normalized_items(declared_conditions['items'])
    opaque_branches = copy.deepcopy(normalized)
    opaque_branches['oneOf'] = [{} for _ in declared_conditions['items']['oneOf']]
    opaque_without_pattern = copy.deepcopy(opaque_branches)
    opaque_without_pattern['properties']['field_ref'].pop('pattern', None)
    if conditions.get('items') not in ({}, declared_conditions['items'], normalized, opaque_branches, opaque_without_pattern):
        raise ValueError('Do not replace an unknown workflow condition definition')
    for key in ('minItems', 'maxItems'):
        if key in conditions and conditions[key] != declared_conditions[key]:
            raise ValueError('Workflow condition bounds changed')
    properties['conditions'] = copy.deepcopy(declared_conditions)
    return result
'''
    compile(helper, 'private-diagnostic-wire-helper', 'exec')
    contents['qw3_computer/condition_schema_wire.py'] = helper.encode()
    contents['qw3_computer/protocol.py'] = wire.patch_protocol(contents['qw3_computer/protocol.py'])
    output = Path(output).resolve()
    if not output.is_relative_to(RUNTIME) or output.is_relative_to(SOURCE):
        raise ValueError('Use a fresh diagnostic directory outside frozen trials')
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    hashes = {}
    for name, data in contents.items():
        target = output / 'framework' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[name] = digest(target)
    result = dict(kind='isolated_image_history_field_identity_condition_wire_candidate',
        source_acceptance_sha256=ACCEPTANCE_SHA, source_framework=str(source),
        source_files_sha256=files, candidate_files_sha256=hashes,
        changed_package_files=[n for n in hashes if hashes[n] != files.get(n)],
        declared_conditions=declaration, declared_conditions_sha256=sha,
        recipe_sha256=digest(__file__), production_deployed=False,
        existing_trials_modified=False, native_field_identity_retained=True,
        model_requests=0, device_inputs=0)
    save(output / 'build.json', result)
    return result


def recorded_observations(directory):
    """Read full visible metadata from the terminal task's observation ledger."""
    database = directory.parent / 'desktop/desktop.sqlite'
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        for metadata, in conn.execute('SELECT metadata FROM observations ORDER BY created'):
            yield database, json.loads(metadata)


def replay(candidate, output):
    import bounded_turn_continuation as continuation
    from condition_schema_wire import codex_normalized_items
    candidate = Path(candidate).resolve()
    manifest = json.loads((candidate / 'build.json').read_text())
    for name, sha in manifest['candidate_files_sha256'].items():
        if digest(candidate / 'framework' / name) != sha:
            raise ValueError('Diagnostic candidate changed')
    sys.path.insert(0, str(candidate / 'framework'))
    from qw3_computer.condition_schema_wire import restore_inline_conditions
    from qw3_computer.execution import check_observation, validate_checks
    from qw3_computer.field_identity import identify_fields, identity_signature
    from qw3_computer.tool_schema import WORKFLOW_TOOLS, tools_for_platform
    declaration = WORKFLOW_TOOLS[0]['inputSchema']['properties']['conditions']
    checks = {}
    for incoming in ({}, declaration['items'], codex_normalized_items(declaration['items'])):
        params = copy.deepcopy(tools_for_platform('linux', True)[1]['inputSchema'])
        params['properties']['step']['properties']['conditions']['items'] = incoming
        restored = restore_inline_conditions('mcp__computer__act', params, declaration)
        checks['wire_restores_' + ('opaque' if not incoming else 'declared' if incoming == declaration['items'] else 'normalized')] = (
            restored['properties']['step']['properties']['conditions'] == declaration)
    params = copy.deepcopy(tools_for_platform('linux', True)[1]['inputSchema'])
    params['properties']['step']['properties']['conditions']['items'] = {'type': 'string'}
    try:
        restore_inline_conditions('mcp__computer__act', params, declaration)
    except ValueError:
        checks['unknown_nonempty_schema_refused'] = True
    else:
        checks['unknown_nonempty_schema_refused'] = False
    checks['field_value_in_wire_and_runtime'] = any(
        b.get('properties', {}).get('kind', {}).get('const') == 'field_value'
        for b in declaration['items']['oneOf'])
    workflow = copy.deepcopy(WORKFLOW_TOOLS[0]['inputSchema'])
    normalized = codex_normalized_items(declaration['items'])
    normalized['oneOf'] = [{} for _ in declaration['items']['oneOf']]
    normalized['properties']['field_ref'].pop('pattern', None)
    workflow['properties']['conditions']['items'] = normalized
    restored = restore_inline_conditions('mcp__computer__workflow', workflow, declaration)
    checks['standalone_workflow_opaque_branches_restored'] = restored['properties']['conditions'] == declaration
    workflow['properties']['conditions']['items'] = {'type': 'string'}
    try:
        restore_inline_conditions('mcp__computer__workflow', workflow, declaration)
    except ValueError:
        checks['unknown_workflow_schema_refused'] = True
    else:
        checks['unknown_workflow_schema_refused'] = False
    from qw3_computer.linux_driver import controls_from_xml
    xml = '<frame name="Diagnostic" visible="true" showing="true" enabled="true" screencoord="(0,0)" size="(300,200)"><entry name="Amount" visible="true" showing="true" enabled="true" focused="true" editable="true" screencoord="(20,20)" size="(100,30)">126</entry></frame>'
    parsed = controls_from_xml(xml, {'title': 'Diagnostic'}, 300, 200)
    entry = next(c for c in parsed['controls'] if c['kind'] == 'entry')
    checks['reader_drops_explicit_native_field_focus'] = 'focused' not in entry
    target = None
    traces = SOURCE / 'attempts/013-recent-observations-r1/app/inference'
    for path, obs in recorded_observations(traces):
        for control in obs.get('accessibility', {}).get('controls', []):
            if control.get('name') == 'First Prize Total Amount *' and control.get('value') == '126' and control.get('field_ref'):
                target = (path, obs, control)
                break
        if target:
            break
    evidence = None
    if target:
        path, obs, control = target
        conditions = [dict(kind='control_value', text='First Prize Total Amount', value='126', control_kind=control['kind']),
                      dict(kind='control_value', text=control['name'], value='126', control_kind=control['kind']),
                      dict(kind='field_value', field_ref=control['field_ref'], value='126')]
        for condition in conditions:
            validate_checks([condition])
        values = [check_observation([condition], obs, obs)[0] for condition in conditions]
        checks['recorded_missing_star_fails'] = values[0]['passed'] is False
        checks['recorded_exact_label_passes'] = values[1]['passed'] is True
        checks['recorded_native_reference_passes'] = values[2]['passed'] is True
        wrong = dict(conditions[2], field_ref='fld_' + '0' * 32)
        checks['different_native_field_ref_fails'] = not check_observation([wrong], obs, obs)[0]['passed']
        evidence = dict(source=str(path), source_sha256=digest(path), observation=obs, conditions=values)
    else:
        checks['recorded_native_field_found'] = False
    database = traces.parent / 'desktop/desktop.sqlite'
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        task, metadata = conn.execute('SELECT task,metadata FROM observations WHERE id=?',
            ('286c6cab51bd45f7bd32594dfa02c689',)).fetchone()
    before_input = json.loads(metadata)
    field = next(c for c in before_input['accessibility']['controls'] if
        c.get('kind') == 'entry' and c.get('name') == 'First Prize Total Amount *')
    checks['native_identity_with_unknown_value_lacks_reference'] = (
        identity_signature(field) is not None and field.get('editable') is True
        and field.get('value') is None and 'field_ref' not in field)
    hypothetical = copy.deepcopy(before_input)
    for c in hypothetical['accessibility']['controls']:
        if c.get('control_id') == field['control_id']:
            c['value'] = ''
    identified = identify_fields(hypothetical, task)
    identified_field = next(c for c in identified['accessibility']['controls'] if c.get('control_id') == field['control_id'])
    checks['identity_reference_creation_currently_depends_on_value_type'] = bool(identified_field.get('field_ref'))
    owner = 'diagnostic-owner'
    state = dict(status='paused', turn_status='completed', id='task', thread_id='thread', turn_id='turn',
                 tool_calls_used=0, tool_call_limit=250, bounded_turn_continuation=continuation.checkpoint(owner, 0, now=100))
    context = dict(owner=owner, origin='running', ready=True, paused=True, unresolved=[], workflow=True, software=False, now=101)
    decisions = {}
    decisions['normal_zero_tool_turn'] = continuation.eligibility(state, **context)
    truncated = copy.deepcopy(state)
    truncated.update(status='failed', turn_status='failed', turn_error={'message': 'Incomplete response returned, reason: max_output_tokens'})
    decisions['actual_failure_shape'] = continuation.eligibility(truncated, **context)
    truncated['status'] = 'paused'
    decisions['truncation_even_if_paused'] = continuation.eligibility(truncated, **context)
    checks['existing_continuation_does_not_recover_output_cap'] = (
        decisions['normal_zero_tool_turn'] == 'eligible' and decisions['actual_failure_shape'] != 'eligible'
        and decisions['truncation_even_if_paused'] == 'turn_or_evidence_error')
    checks['bounded_field_input_tool_absent'] = not any(
        'fill_field' in str(t) for t in tools_for_platform('linux', True))
    result = dict(kind='paused_cohort_focused_contract_and_field_replay', checks=checks,
        passed=all(checks.values()), model_requests=0, device_inputs=0,
        continuation_decisions=decisions, recorded_field_evidence=evidence,
        business_success_improvement_proven=False, candidate_build_sha256=digest(candidate / 'build.json'),
        limitation='Recorded evidence and pure validators; no new autonomous task or live input executor is tested.')
    save(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('build', 'replay'))
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.output) if args.mode == 'build' else replay(args.candidate, args.output)
    print(json.dumps({k: result[k] for k in ('kind', 'passed', 'checks', 'changed_package_files') if k in result}))
