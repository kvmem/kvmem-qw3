"""Positive AT-SPI button evidence; native object identity is never guessed.

This reader does not invoke actions. A missing or partial native interface
cannot authorize a semantic-control dispatch.
"""
import json

from field_identity import atspi_identity_script, identity_signature


STATE_NAMES = ('enabled', 'sensitive', 'visible', 'showing', 'defunct', 'busy')
ACTION_PROVIDER = 'atspi-native-button-action-v1'


def read_button_metadata(node, pyatspi):
    """Also embedded verbatim in the fixed guest reader/dispatcher."""
    if node.getRoleName() != 'push button':
        return None
    state = node.getState()
    flags = {name: bool(state.contains(getattr(pyatspi, 'STATE_' + name.upper())))
             for name in ('enabled', 'sensitive', 'visible', 'showing', 'defunct', 'busy')}
    interface = node.queryAction()
    count = interface.nActions
    if type(count) is not int or not 0 < count <= 8:
        return None
    names = [interface.getName(index) for index in range(count)]
    if any(not isinstance(name, str) or not name or len(name) > 64 for name in names):
        return None
    return {'provider': 'atspi-native-button-action-v1', 'states': flags, 'actions': names}


def button_metadata_from_attrs(attrs):
    try:
        metadata = json.loads(attrs['qw3-native-button-action'])
    except (KeyError, ValueError, TypeError):
        return None
    if (not isinstance(metadata, dict) or set(metadata) != {'provider', 'states', 'actions'}
            or metadata['provider'] != ACTION_PROVIDER):
        return None
    states, actions = metadata['states'], metadata['actions']
    if (not isinstance(states, dict) or set(states) != set(STATE_NAMES)
            or any(type(value) is not bool for value in states.values())
            or not isinstance(actions, list) or not 0 < len(actions) <= 8
            or any(not isinstance(name, str) or not name or len(name) > 64 for name in actions)):
        return None
    return metadata


def native_button(control):
    """Require a bound, enabled, visible button with one canonical action."""
    if not isinstance(control, dict) or control.get('kind') != 'push-button':
        return None
    if identity_signature(control) is None:
        return None
    metadata = button_metadata_from_attrs({
        'qw3-native-button-action': json.dumps(control.get('native_button_action'))})
    if metadata is None:
        return None
    states = metadata['states']
    if (not all(states[name] for name in ('enabled', 'sensitive', 'visible', 'showing'))
            or states['defunct'] or states['busy']):
        return None
    choices = [index for index, name in enumerate(metadata['actions']) if name in {'click', 'press'}]
    if len(choices) != 1:
        return None
    return {'identity': control['provider_identity'], 'metadata': metadata,
            'action_index': choices[0], 'action_name': metadata['actions'][choices[0]]}


def atspi_button_script(source):
    """Extend the exact official reader in this process; guest files stay intact."""
    import inspect
    source = atspi_identity_script(source)
    anchor = " xml=a.lxml.etree.tostring(a.create_atspi_node(frames[0]),encoding='unicode')"
    if source.count(anchor) != 1:
        raise ValueError('Qualified AT-SPI button traversal anchor changed')
    reader = inspect.getsource(read_button_metadata)
    injection = '\n'.join(' ' + line for line in reader.splitlines()) + '''
 button_original=a.create_atspi_node
 def with_native_button(node,depth=0,flag=None):
  xml=button_original(node,depth,flag)
  try:
   metadata=read_button_metadata(node,a.pyatspi)
   if metadata is not None:
    xml.set('qw3-native-button-action',json.dumps(metadata,sort_keys=True))
  except Exception:
   pass
  return xml
 a.create_atspi_node=with_native_button
'''
    result = source.replace(anchor, injection + anchor, 1)
    compile(result, 'fixed-readonly-atspi-button-observer', 'exec')
    return result
