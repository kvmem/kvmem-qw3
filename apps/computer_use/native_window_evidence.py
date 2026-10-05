"""Read-only native X11 window evidence, independent of AT-SPI availability.

These observations never authorize input by themselves. A policy must still
bind the task/device session, inspect the intended target and check dispatch.
X11's reported PID is correlated with /proc; it is not a security attestation.
"""
import inspect
import math
import re


PROVIDER = "x11-xlib-native-window-v1"


def read_native_window():
    """Executed only as fixed driver code in the owned Linux guest."""
    import os
    from pathlib import Path
    import signal
    import time

    began = time.monotonic()
    connection = None

    def expired(*_):
        raise TimeoutError("Native X11 observation deadline")

    handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expired)
    prior_timer = signal.setitimer(signal.ITIMER_REAL, 2)
    try:
        from Xlib import X, Xatom
        from Xlib.display import Display
        connection = Display()
        root = connection.screen().root

        def cardinal(window, name):
            value = window.get_full_property(connection.intern_atom(name, only_if_exists=True), Xatom.CARDINAL)
            if value is None or value.format != 32 or len(value.value) != 1:
                raise ValueError("Missing unique native PID")
            return int(value.value[0])

        def active_window():
            value = root.get_full_property(connection.intern_atom('_NET_ACTIVE_WINDOW', only_if_exists=True), Xatom.WINDOW)
            if value is None or value.format != 32 or len(value.value) != 1 or int(value.value[0]) <= 1:
                raise ValueError("No native active window")
            return connection.create_resource_object('window', int(value.value[0]))

        def snapshot():
            window = active_window()
            geometry = window.get_geometry()
            position = root.translate_coords(window, 0, 0)
            attributes = window.get_attributes()
            pid = cardinal(window, '_NET_WM_PID')
            process = Path('/proc') / str(pid)
            owner = dict(pid=pid, uid=process.stat().st_uid,
                         start_ticks=int((process / 'stat').read_text().rpartition(')')[2].split()[19]),
                         executable=os.readlink(process / 'exe'))
            title = window.get_full_text_property(connection.intern_atom('_NET_WM_NAME', only_if_exists=True))
            if title is None:
                title = window.get_wm_name()
            classes = window.get_wm_class()
            focus = connection.get_input_focus().focus
            focus_id = getattr(focus, 'id', focus)
            ancestry = []
            current = focus
            # Preserve PointerRoot/None as unavailable focus, never guess an app.
            while hasattr(current, 'id') and current.id not in ancestry and len(ancestry) < 64:
                ancestry.append(current.id)
                if current.id in (window.id, root.id):
                    break
                current = current.query_tree().parent
            root_geometry = root.get_geometry()
            return dict(window_id=window.id, root_id=root.id,
                        display_name=os.environ.get('DISPLAY', ''),
                        display_size=[root_geometry.width, root_geometry.height],
                        title=title, wm_class=list(classes) if classes else [],
                        bounds=[position.x, position.y, geometry.width, geometry.height],
                        same_screen=bool(position.same_screen), viewable=attributes.map_state == X.IsViewable,
                        override_redirect=bool(attributes.override_redirect), reported_owner=owner,
                        focus_window_id=int(focus_id), focus_ancestry=ancestry,
                        focus_within_active=bool(ancestry and ancestry[-1] == window.id))

        before = snapshot()
        after = snapshot()
        elapsed = (time.monotonic() - began) * 1000
        if before != after or elapsed >= 2000:
            return dict(provider="x11-xlib-native-window-v1", status="changed",
                        reason="Native window/focus/process changed during read", read_milliseconds=elapsed)
        return dict(provider="x11-xlib-native-window-v1", status="ok", **after,
                    captured_at=int(time.time() * 1000), read_milliseconds=elapsed,
                    consistency="two_equal_native_reads", input_sent=False)
    except Exception as error:
        return dict(provider="x11-xlib-native-window-v1", status="unavailable",
                    error_kind=type(error).__name__, read_milliseconds=(time.monotonic() - began) * 1000,
                    input_sent=False)
    finally:
        # Cancel the probe's timer before cleanup; no X11 input API is called.
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if connection is not None:
                connection.close()
        finally:
            signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, *prior_timer)


def guest_script():
    return inspect.getsource(read_native_window) + "\nimport json\nprint(json.dumps(read_native_window()))\n"


def valid_native_window(value, foreground, width, height):
    """Validate positive geometry and process/focus correlation; no input grant."""
    if not isinstance(value, dict) or not isinstance(foreground, dict):
        return None
    if (value.get('provider') != PROVIDER or value.get('status') != 'ok'
            or value.get('consistency') != 'two_equal_native_reads' or value.get('input_sent') is not False
            or type(width) is not int or type(height) is not int or width <= 0 or height <= 0
            or value.get('display_size') != [width, height]
            or not isinstance(value.get('display_name'), str) or not value['display_name']
            or type(value.get('captured_at')) is not int or value['captured_at'] <= 0
            or type(value.get('read_milliseconds')) not in (int, float)
            or not math.isfinite(value['read_milliseconds']) or not 0 <= value['read_milliseconds'] < 2000
            or value.get('same_screen') is not True or value.get('viewable') is not True
            or value.get('override_redirect') is not False):
        return None
    handle = foreground.get('handle')
    if not isinstance(handle, str) or not re.fullmatch(r'0x[0-9a-fA-F]+', handle):
        return None
    wid, root = value.get('window_id'), value.get('root_id')
    classes = value.get('wm_class')
    if (type(wid) is not int or not 1 < wid < 2**32 or int(handle, 16) != wid
            or type(root) is not int or not 1 < root < 2**32 or root == wid
            or not isinstance(value.get('title'), str) or value['title'] != foreground.get('title')
            or not isinstance(classes, list) or len(classes) != 2
            or any(not isinstance(v, str) or not v for v in classes)
            or classes[-1] != foreground.get('process')):
        return None
    bounds = value.get('bounds')
    if (not isinstance(bounds, list) or len(bounds) != 4 or any(type(v) is not int for v in bounds)
            or not 0 <= bounds[0] < bounds[0] + bounds[2] <= width
            or not 0 <= bounds[1] < bounds[1] + bounds[3] <= height):
        return None
    owner = value.get('reported_owner')
    if (not isinstance(owner, dict) or set(owner) != {'pid', 'uid', 'start_ticks', 'executable'}
            or any(type(owner.get(k)) is not int for k in ('pid', 'uid', 'start_ticks'))
            or not 0 < owner['pid'] < 2**31 or owner['uid'] < 0 or owner['start_ticks'] <= 0
            or not isinstance(owner.get('executable'), str) or not owner['executable'].startswith('/')):
        return None
    ancestry = value.get('focus_ancestry')
    if (value.get('focus_within_active') is not True or type(value.get('focus_window_id')) is not int
            or not isinstance(ancestry, list) or not 1 <= len(ancestry) <= 64
            or any(type(v) is not int or not 1 < v < 2**32 for v in ancestry)
            or len(set(ancestry)) != len(ancestry) or ancestry[0] != value['focus_window_id']
            or ancestry[-1] != wid or root in ancestry):
        return None
    return value


def native_window_context(observation):
    if not isinstance(observation, dict) or observation.get('capabilities', {}).get('platform') != 'linux':
        return None
    if any(not isinstance(observation.get(key), str) or not observation[key]
           for key in ('desktop_id', 'device_id', 'session_id', 'display_id')):
        return None
    return valid_native_window(observation.get('native_window'), observation.get('foreground'),
                               observation.get('width'), observation.get('height'))


def same_native_window(old, fresh):
    """Pair only observations bound to the same display/session/process/focus."""
    first, second = native_window_context(old), native_window_context(fresh)
    if first is None or second is None:
        return None
    identity = ('desktop_id', 'device_id', 'session_id', 'display_id', 'width', 'height', 'rotation')
    if any(old.get(k) is None or old.get(k) != fresh.get(k) for k in identity):
        return None
    fields = ('window_id', 'root_id', 'display_name', 'display_size', 'title', 'wm_class',
              'bounds', 'reported_owner', 'focus_window_id', 'focus_ancestry')
    if any(first[k] != second[k] for k in fields) or second['captured_at'] < first['captured_at']:
        return None
    x, y, width, height = first['bounds']
    return dict(basis=PROVIDER, bounds=[x, y, x + width, y + height],
                reported_owner=first['reported_owner'], focus_window_id=first['focus_window_id'],
                input_authorized=False, pixels_or_semantic_target_not_yet_compared=True)
