"""Restore an opaque computer-condition schema from its trusted declaration.

This is an isolated prototype, not installed in a live/frozen bridge. It changes
no argument, action, tool permission, task, completion predicate or dispatcher.
"""
import copy

from jsonschema import Draft202012Validator


def repair_condition_parameters(name,parameters,declared_conditions):
    if not isinstance(parameters,dict):raise ValueError('Tool parameters must be an object')
    result=copy.deepcopy(parameters)
    if name not in ('mcp__computer__act','mcp__computer__workflow'):return result,False
    fields=result.get('properties',{})
    if name=='mcp__computer__act':
        step=fields.get('step',{})
        fields=step.get('properties',{}) if isinstance(step,dict) else {}
    conditions=fields.get('conditions')
    if conditions is None:return result,False
    if not isinstance(conditions,dict) or conditions.get('type')!='array':raise ValueError('Unexpected computer condition container')
    if not isinstance(declared_conditions,dict) or declared_conditions.get('type')!='array':raise ValueError('Require the trusted declared condition array')
    declared_items=declared_conditions.get('items')
    if (not isinstance(declared_items,dict) or declared_items.get('type')!='object'
            or not declared_items.get('properties') or not declared_items.get('oneOf')):
        raise ValueError('Trusted declaration lacks explicit condition fields and variants')
    Draft202012Validator.check_schema(declared_conditions)
    for key in ('minItems','maxItems'):
        if key in conditions and conditions[key]!=declared_conditions.get(key):raise ValueError('Do not replace different declared condition bounds')
    if conditions.get('items')==declared_items:return result,False
    if conditions.get('items')!={}:raise ValueError('Do not overwrite an unknown or nonempty condition definition')
    conditions['items']=copy.deepcopy(declared_items)
    for key in ('minItems','maxItems'):
        if key in declared_conditions:conditions[key]=declared_conditions[key]
    return result,True
