"""Revalidate a native button and invoke its fixed AT-SPI action exactly once.

Pixel changes outside the button are admitted only for this semantic route.
Visual mouse actions retain their existing evidence requirements. An invocation
receipt never proves that a document was saved or a business task completed.
"""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import re
import secrets

from PIL import Image

from native_control_evidence import native_button, read_button_metadata
from native_keyboard_execution import dispatch_expectation, native_dispatch_matches
from native_window_evidence import read_native_window, same_native_window


REJECTION_KIND = 'qw3-native-button-pre-invoke-rejection-v1'
INVOCATION_KIND = 'qw3-native-button-invocation-v1'


def semantic_controls(observation):
    a11y = observation.get('accessibility')
    if (not isinstance(a11y, dict) or a11y.get('status') != 'ok'
            or a11y.get('truncated') is not False or a11y.get('scope') != 'foreground_window'
            or a11y.get('provider') != 'official-osworld-create_atspi_node'
            or not isinstance(a11y.get('controls'), list)
            or any(not isinstance(c, dict) for c in a11y['controls'])):
        return None
    return [{key: value for key, value in control.items() if key != 'control_id'}
            for control in a11y['controls']]


def native_control_revalidation_evidence(runtime, old, fresh, target):
    """Positive evidence for a native invocation, not a relaxed visual click."""
    button = native_button(target)
    if button is None or dispatch_expectation(old) is None or dispatch_expectation(fresh) is None:
        return None
    proof = same_native_window(old, fresh)
    if proof is None or button['identity']['owner_pid'] != proof['reported_owner']['pid']:
        return None
    before, after = semantic_controls(old), semantic_controls(fresh)
    if before is None or before != after:
        return None
    identity = button['identity']
    if any(sum(control.get('provider_identity') == identity for control in controls) != 1
           for controls in (before, after)):
        return None
    signature = {key: value for key, value in target.items() if key != 'control_id'}
    matches = [control for control in fresh['accessibility']['controls']
               if {key: value for key, value in control.items() if key != 'control_id'} == signature]
    if len(matches) != 1 or not isinstance(matches[0].get('control_id'), str):
        return None
    fields = ('x', 'y', 'width', 'height', 'center_x', 'center_y')
    if any(type(target.get(field)) is not int for field in fields):
        return None
    x, y, w, h = [target[key] for key in fields[:4]]
    left, top, right, bottom = proof['bounds']
    if not left <= x < x + w <= right or not top <= y < y + h <= bottom:
        return None
    bounds = (x, y, x + w, y + h)
    paths = []
    for observation in (old, fresh):
        sha = observation.get('sha256')
        if not isinstance(sha, str) or not re.fullmatch('[a-f0-9]{64}', sha):
            return None
        path = Path(runtime) / 'screenshots' / (sha + '.png')
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            return None
        paths.append(path)
    with Image.open(paths[0]) as first, Image.open(paths[1]) as second:
        if (first.format != 'PNG' or second.format != 'PNG'
                or first.size != (old['width'], old['height']) or second.size != first.size
                or first.crop(bounds).convert('RGBA').tobytes() != second.crop(bounds).convert('RGBA').tobytes()):
            return None
    return proof | dict(basis='native_button_identity_action_and_unchanged_semantic_controls',
                        fresh_control_id=matches[0]['control_id'], target_bounds=list(bounds),
                        native_identity=identity, native_action_name=button['action_name'],
                        native_action_index=button['action_index'], target_pixels_unchanged=True,
                        entire_client_pixels_required=False, observed_controls_unchanged=True,
                        requires_immediate_native_dispatch_check=True, input_authorized=False)


def control_dispatch_expectation(observation, control):
    native = dispatch_expectation(observation)
    button = native_button(control)
    if native is None or button is None or semantic_controls(observation) is None:
        return None
    if button['identity']['owner_pid'] != native['reported_owner']['pid']:
        return None
    candidates = observation['accessibility']['controls']
    if sum(c.get('provider_identity') == button['identity'] for c in candidates) != 1:
        return None
    if control.get('value') not in (None, '', control.get('name')):
        return None
    return dict(native=copy.deepcopy(native), control=copy.deepcopy(control), button=copy.deepcopy(button))


def resolve_guest_button(expected, pyatspi):
    """Fixed bounded native-object traversal; no coordinate or name fallback."""
    identity = expected['button']['identity']
    frames = []
    for app in pyatspi.Registry.getDesktop(0):
        if app.app.bus_name != identity['bus_name']:
            continue
        for frame in app:
            if frame.path == identity['window_path']:
                frames.append(frame)
    if len(frames) != 1:
        raise ValueError('Native frame identity is not unique')
    frame = frames[0]
    if (frame.get_process_id() != identity['owner_pid']
            or frame.name != expected['native']['title']
            or not frame.getState().contains(pyatspi.STATE_ACTIVE)):
        raise ValueError('Native frame owner or active title changed')
    stack = [(frame, 0)]
    targets = []
    visited = set()
    while stack:
        node, depth = stack.pop()
        if depth > 16 or len(visited) >= 512 or node.path in visited:
            raise ValueError('Native traversal is partial or ambiguous')
        visited.add(node.path)
        if node.app.bus_name != identity['bus_name']:
            raise ValueError('Native subtree owner changed')
        if node.path == identity['object_path']:
            targets.append(node)
        stack.extend((child, depth + 1) for child in node)
    if len(targets) != 1:
        raise ValueError('Native button identity is not unique')
    node = targets[0]
    control = expected['control']
    bounds = node.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
    if (node.name != control['name'] or node.getRoleName() != 'push button'
            or [bounds.x, bounds.y, bounds.width, bounds.height]
               != [control[key] for key in ('x', 'y', 'width', 'height')]
            or read_button_metadata(node, pyatspi) != expected['button']['metadata']):
        raise ValueError('Native button state, geometry or action changed')
    return node


def native_control_script(expected, *, nonce=None):
    if not isinstance(expected, dict) or set(expected) != {'native', 'control', 'button'}:
        raise ValueError('Use an observed native button dispatch expectation')
    nonce = secrets.token_hex(16) if nonce is None else nonce
    if not isinstance(nonce, str) or not re.fullmatch('[a-f0-9]{32}', nonce):
        raise ValueError('Use an exact invocation nonce')
    helpers = '\n'.join(inspect.getsource(fn) for fn in
                        (read_native_window, native_dispatch_matches, read_button_metadata, resolve_guest_button))
    body = '''
import json,signal
import pyatspi
_expected=json.loads(EXPECTED_LITERAL)
_nonce=NONCE_LITERAL
def _main():
 def expired(*_):raise TimeoutError('Native button read deadline')
 signal.signal(signal.SIGALRM,expired)
 signal.setitimer(signal.ITIMER_REAL,5)
 try:
  if not native_dispatch_matches(_expected['native'],read_native_window()):
   raise ValueError('Native window, process or focus changed')
  # Two complete reads detect object replacement or state changes before input.
  resolve_guest_button(_expected,pyatspi)
  node=resolve_guest_button(_expected,pyatspi)
  interface=node.queryAction()
  if [interface.getName(i) for i in range(interface.nActions)]!=_expected['button']['metadata']['actions']:
   raise ValueError('Native actions changed immediately before invocation')
  if not native_dispatch_matches(_expected['native'],read_native_window()):
   raise ValueError('Native window, process or focus changed before invocation')
 except Exception as error:
  signal.setitimer(signal.ITIMER_REAL,0)
  print(json.dumps(dict(kind=REJECTION_LITERAL,nonce=_nonce,input_sent=False,reason='native_button_or_context_changed_or_unavailable',error_kind=type(error).__name__)))
  return
 signal.setitimer(signal.ITIMER_REAL,0)
 # From this line onwards errors/timeouts are uncertain. Never convert them
 # to a no-input rejection or call doAction again, including when it returnsFalse.
 accepted=interface.doAction(_expected['button']['action_index'])
 print(json.dumps(dict(kind=INVOCATION_LITERAL,nonce=_nonce,invocation_attempted=True,invocation_accepted=bool(accepted),action_index=_expected['button']['action_index'],action_name=_expected['button']['action_name'])))
_main()
'''
    replacements = {'EXPECTED_LITERAL': repr(json.dumps(expected, ensure_ascii=True)), 'NONCE_LITERAL': repr(nonce),
                    'REJECTION_LITERAL': repr(REJECTION_KIND), 'INVOCATION_LITERAL': repr(INVOCATION_KIND)}
    for token, literal in replacements.items():
        body = body.replace(token, literal)
    script = helpers + '\n' + body
    compile(script, 'fixed-native-button-invocation', 'exec')
    return script, nonce


def checked_native_control_result(result, nonce):
    if result.get('returncode', 0) != 0 or result.get('status') == 'error':
        raise RuntimeError('Native invocation delivery is uncertain')
    try:
        output = json.loads(result['output'])
    except (ValueError, TypeError, KeyError) as error:
        raise RuntimeError('Native invocation receipt is unavailable; delivery is uncertain') from error
    if not isinstance(output, dict) or output.get('nonce') != nonce:
        raise RuntimeError('Native invocation receipt is unbound; delivery is uncertain')
    if (set(output) == {'kind', 'nonce', 'input_sent', 'reason', 'error_kind'}
            and output.get('kind') == REJECTION_KIND and output.get('input_sent') is False
            and output.get('reason') == 'native_button_or_context_changed_or_unavailable'
            and isinstance(output.get('error_kind'), str)):
        return 'rejected', output
    if (set(output) == {'kind', 'nonce', 'invocation_attempted', 'invocation_accepted', 'action_index', 'action_name'}
            and output.get('kind') == INVOCATION_KIND and output.get('invocation_attempted') is True
            and type(output.get('invocation_accepted')) is bool and type(output.get('action_index')) is int
            and output['action_index'] >= 0 and output.get('action_name') in {'click', 'press'}):
        return 'invoked', output
    raise RuntimeError('Native invocation receipt is invalid; delivery is uncertain')


async def execute_native_control(driver, expected):
    """One transport call; do not inherit a legacy traceback-based rejection."""
    script, nonce = native_control_script(expected)
    payload = {'command': ['python', '-c', script], 'shell': False, 'timeout': 8}
    async with driver.session.post(driver.endpoint + '/execute', json=payload) as response:
        result = await response.json()
        if response.status != 200:
            raise RuntimeError('Native invocation transport failed; delivery is uncertain')
    status, receipt = checked_native_control_result(result, nonce)
    if status == 'invoked' and (receipt['action_index'] != expected['button']['action_index']
                                or receipt['action_name'] != expected['button']['action_name']):
        raise RuntimeError('Native invocation action receipt differs; delivery is uncertain')
    if status == 'invoked' and receipt['invocation_accepted'] is False:
        # The native method was called. Its False result proves neither actual
        # delivery nor non-delivery; preserve the reservation and never retry.
        raise RuntimeError('Native invocation was not accepted; delivery is uncertain')
    return status, receipt
