"""Trusted inline-condition restoration for a separately built private bridge.

This helper is copied into that candidate package only. The normal package and
all frozen queues keep their original converter. No runtime test dependency is
needed. Only an empty definition or the exact known Codex normalization of the
trusted declaration is restored; unknown nonempty definitions are refused.
"""
import copy
import hashlib
import json

CONDITION_SHA256 = 'a1dc0d942bb1107f8728e47644fcb0d507bcfc294813de6a72c60293c78d21c6'


def codex_normalized_items(value):
    """The observed pinned-client normalization, applied to trusted data only."""
    if isinstance(value, list):return [codex_normalized_items(v) for v in value]
    if not isinstance(value, dict):return value
    result = {k:codex_normalized_items(v) for k,v in value.items() if k not in ('description', 'const')}
    if 'const' in value:
        if not isinstance(value['const'], str):raise ValueError('Unexpected trusted condition selector')
        result.update(type='string', enum=[value['const']])
    return result


def restore_inline_conditions(name, parameters, declared_conditions):
    result = copy.deepcopy(parameters)
    if name != 'mcp__computer__act':
        return result
    if not isinstance(result, dict) or not isinstance(result.get('properties'), dict):
        raise ValueError('Computer parameters lost their object fields')
    step = result['properties'].get('step')
    if step is None:
        return result
    if not isinstance(step, dict) or not isinstance(step.get('properties'), dict):
        raise ValueError('Computer step lost its object fields')
    conditions = step['properties'].get('conditions')
    if conditions is None:
        raise ValueError('Computer step lost its conditions')
    digest = hashlib.sha256(json.dumps(declared_conditions, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    if digest != CONDITION_SHA256:
        raise ValueError('Trusted condition declaration changed; qualify a new candidate')
    if not isinstance(conditions, dict) or conditions.get('type') != 'array':
        raise ValueError('Unexpected inline condition container')
    for key in ('minItems', 'maxItems'):
        if key in conditions and conditions[key] != declared_conditions[key]:
            raise ValueError('Do not replace different inline condition bounds')
    if conditions.get('items') not in ({}, declared_conditions['items'], codex_normalized_items(declared_conditions['items'])):
        raise ValueError('Do not overwrite an unknown/nonempty inline definition')
    conditions['items'] = copy.deepcopy(declared_conditions['items'])
    for key in ('minItems', 'maxItems'):
        conditions[key] = declared_conditions[key]
    return result
