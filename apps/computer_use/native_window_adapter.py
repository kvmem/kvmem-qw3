"""Compose native reads with an existing Linux observer without sending input."""
import io
import hashlib
import json
from pathlib import Path
import re

from PIL import Image

from native_window_evidence import guest_script, same_native_window, valid_native_window


NATIVE_FIELDS = ('window_id', 'root_id', 'display_name', 'display_size', 'title', 'wm_class',
                 'bounds', 'reported_owner', 'focus_window_id', 'focus_ancestry')


async def observe_native_window(driver, original_observe):
    """Bracket the original screenshot with two bounded, fixed native reads.

The original image, controls, foreground and progress fields remain intact.
Missing/changed native evidence is explicit; it does not become a permission.
"""
    async def read():
        try:
            result = await driver.execute(guest_script(), read_timeout=3)
            value = json.loads(result['output'])
            if not isinstance(value, dict):
                raise ValueError('Native observer returned a non-object')
            return value
        except Exception as error:
            return dict(status='unavailable', error_kind=type(error).__name__, input_sent=False)

    before = await read()
    raw = await original_observe(driver)
    after = await read()
    with Image.open(io.BytesIO(raw['png'])) as image:
        width, height = image.size
    foreground = raw.get('foreground')
    valid = (valid_native_window(before, foreground, width, height) is not None
             and valid_native_window(after, foreground, width, height) is not None
             and all(before[key] == after[key] for key in NATIVE_FIELDS)
             and before['captured_at'] <= after['captured_at'])
    evidence = after if valid else dict(status='unavailable_or_changed', input_sent=False,
                                        reason='Native reads do not bind the original screenshot window')
    return raw | dict(native_window=evidence,
                      native_window_capture=dict(before=before, after=after,
                                                 screenshot_bracket_matched=valid,
                                                 input_authorized=False))


def exact_native_window_pixels(runtime, old, fresh):
    """Compare actual client-window pixels; do not mask arbitrary rectangles."""
    for observation in (old, fresh):
        capture = observation.get('native_window_capture', {})
        before, after = capture.get('before'), capture.get('after')
        if (capture.get('screenshot_bracket_matched') is not True
                or valid_native_window(before, observation.get('foreground'), observation.get('width'), observation.get('height')) is None
                or valid_native_window(after, observation.get('foreground'), observation.get('width'), observation.get('height')) is None
                or observation.get('native_window') != after
                or before['captured_at'] > after['captured_at']
                or any(before[key] != after[key] for key in NATIVE_FIELDS)
                or not isinstance(observation.get('sha256'), str)
                or not re.fullmatch(r'[a-f0-9]{64}', observation['sha256'])):
            return None
    proof = same_native_window(old, fresh)
    if proof is None:
        return None
    paths = [Path(runtime) / 'screenshots' / (observation['sha256'] + '.png')
             for observation in (old, fresh)]
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != observation['sha256']
           for path, observation in zip(paths, (old, fresh))):
        return None
    with Image.open(paths[0]) as first, Image.open(paths[1]) as second:
        if first.format != 'PNG' or second.format != 'PNG' or first.size != (old['width'], old['height']) or second.size != first.size:
            return None
        if first.crop(proof['bounds']).convert('RGBA').tobytes() != second.crop(proof['bounds']).convert('RGBA').tobytes():
            return None
    return proof | dict(pixels_unchanged=True, requires_semantic_target_revalidation=True,
                        input_authorized=False)
