"""Additional Linux keyboard evidence and a fixed native dispatch gate.

Only a screenshot-bound, unchanged native client can support this extra path.
The native gate runs in the guest process sending input, with no automatic
retry. It also checks again after clipboard preparation, before paste.
"""
import copy
import inspect
import json
import secrets

from native_window_adapter import NATIVE_FIELDS, exact_native_window_pixels
from native_window_evidence import native_window_context, read_native_window, valid_native_window


REJECTION_KIND = 'qw3-native-pre-dispatch-rejection-v1'


def dispatch_expectation(observation):
    native = native_window_context(observation)
    if native is None:
        return None
    capture = observation.get('native_window_capture', {})
    before, after = capture.get('before'), capture.get('after')
    if (capture.get('screenshot_bracket_matched') is not True or after != native
            or valid_native_window(before, observation['foreground'], observation['width'], observation['height']) is None
            or before['captured_at'] > after['captured_at']
            or any(before[key] != after[key] for key in NATIVE_FIELDS)):
        return None
    return copy.deepcopy(native)


def keyboard_revalidation_evidence(runtime, old, fresh, kind):
    if kind not in {'key', 'type'} or dispatch_expectation(fresh) is None:
        return None

    def semantic_observation(value):
        a11y = value.get('accessibility')
        if (not isinstance(a11y, dict) or a11y.get('status') not in {'ok', 'unavailable', 'foreground_unmatched'}
                or not isinstance(a11y.get('controls'), list)
                or any(not isinstance(c, dict) for c in a11y['controls'])):
            return None
        return {key:a11y.get(key) for key in ('status', 'truncated', 'scope', 'provider')} | {
            'controls':[{k:v for k,v in c.items() if k != 'control_id'} for c in a11y['controls']]}

    before, after = semantic_observation(old), semantic_observation(fresh)
    if before is None or before != after:
        return None
    proof = exact_native_window_pixels(runtime, old, fresh)
    if proof is None:
        return None
    return proof | dict(action_kind=kind, observed_controls_unchanged=True,
                        accessibility_available=after['status'] == 'ok',
                        missing_controls_not_inferred=True,
                        requires_immediate_native_dispatch_check=True)


def native_dispatch_matches(expected, actual):
    """Used unchanged in the fixed guest script; no authorization from a PID."""
    fields = ('window_id', 'root_id', 'display_name', 'display_size', 'title', 'wm_class',
              'bounds', 'reported_owner', 'focus_window_id', 'focus_ancestry')
    return (isinstance(actual, dict) and actual.get('provider') == 'x11-xlib-native-window-v1'
            and actual.get('status') == 'ok' and actual.get('consistency') == 'two_equal_native_reads'
            and actual.get('input_sent') is False and actual.get('focus_within_active') is True
            and actual.get('same_screen') is True and actual.get('viewable') is True
            and actual.get('override_redirect') is False
            and type(actual.get('captured_at')) is int
            and actual['captured_at'] >= expected['captured_at']
            and all(actual.get(key) == expected[key] for key in fields))


def guarded_input_script(original, expected, kind, *, nonce=None):
    """Return one fixed script and a receipt nonce, without sending anything.

A gate rejection prints a structured, nonce-bound no-input result. Transport
errors and failed input remain uncertain. Traceback/source text is never used
as evidence of non-delivery. The original xprop guard is retained.
"""
    if expected is None:
        return original, None
    nonce = secrets.token_hex(16) if nonce is None else nonce
    if not isinstance(nonce, str) or len(nonce) != 32 or any(c not in '0123456789abcdef' for c in nonce):
        raise ValueError('Use an exact dispatch receipt nonce')
    if kind not in {'click', 'type', 'key', 'scroll', 'drag'}:
        raise ValueError('Native dispatch is only for desktop input')
    # Imports occur before the gate. Clipboard preparation is allowed only after
    # the first gate, and a second check precedes the actual paste key event.
    anchor = 'import pyautogui;pyautogui.FAILSAFE=False;'
    if original.count(anchor) != 1:
        raise ValueError('Original SDK51 input script changed')
    prefix, body = original.split(anchor)
    if kind == 'type':
        paste = "pyautogui.hotkey('ctrl','v')"
        if body.count(paste) != 1:
            raise ValueError('Original SDK51 paste script changed')
        body = body.replace(paste,
            "\nif not _qw3_gate(clipboard_prepared=True):\n return\n" + paste, 1)
    function = '\n'.join(' ' + line for line in body.splitlines())
    script = (inspect.getsource(read_native_window) + '\n' + inspect.getsource(native_dispatch_matches)
              + '\nimport json\n'
              + '_qw3_expected=json.loads(' + repr(json.dumps(expected, ensure_ascii=True)) + ')\n'
              + 'def _qw3_gate(clipboard_prepared=False):\n'
              + ' actual=read_native_window()\n'
              + ' if native_dispatch_matches(_qw3_expected,actual):return True\n'
              + ' print(json.dumps(dict(kind=' + repr(REJECTION_KIND) + ',nonce=' + repr(nonce)
              + ',input_sent=False,clipboard_prepared=clipboard_prepared,reason="native_window_process_or_focus_changed_or_unavailable")))\n'
              + ' return False\n'
              + 'def _qw3_send_input():\n'
              + ' ' + prefix + anchor + '\n'
              + ' if not _qw3_gate():return\n' + function + '\n'
              + '_qw3_send_input()\n')
    compile(script, 'fixed-native-input-dispatch', 'exec')
    return script, nonce


def confirmed_native_rejection(result, nonce):
    if nonce is None or result.get('returncode', 0) != 0 or result.get('status') == 'error':
        return None
    try:
        output = json.loads(result['output'])
    except (KeyError, ValueError, TypeError):
        return None
    if (not isinstance(output, dict) or set(output) != {'kind', 'nonce', 'input_sent', 'clipboard_prepared', 'reason'}
            or output.get('kind') != REJECTION_KIND or output.get('nonce') != nonce
            or output.get('input_sent') is not False or type(output.get('clipboard_prepared')) is not bool
            or output.get('reason') != 'native_window_process_or_focus_changed_or_unavailable'):
        return None
    return output
