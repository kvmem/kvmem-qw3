import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from native_window_adapter import exact_native_window_pixels, observe_native_window
from test_native_window_evidence import observation


def small_observation():
    value = observation()
    value.update(width=8, height=8)
    value['native_window'].update(display_size=[8, 8], bounds=[2, 2, 4, 4])
    value['native_window_capture'] = dict(before=copy.deepcopy(value['native_window']),
                                         after=copy.deepcopy(value['native_window']),
                                         screenshot_bracket_matched=True, input_authorized=False)
    return value


class NativeCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def capture(self, change=None, failure=None):
        native = small_observation()
        buffer = io.BytesIO()
        Image.new('RGB', (8, 8), 'white').save(buffer, format='PNG')
        raw = dict(png=buffer.getvalue(), foreground=native['foreground'],
                   accessibility=dict(status='foreground_unmatched', controls=[]),
                   progress_sha256='original-progress')

        class Driver:
            calls = 0
            async def execute(self, script, *, read_timeout):
                self.calls += 1
                if failure and self.calls == failure:
                    raise TimeoutError('Read-only native timeout')
                value = copy.deepcopy(native['native_window'])
                if self.calls == 2 and change:
                    value.update(change)
                return {'output': json.dumps(value)}

        async def original(driver):
            driver.original_observations = getattr(driver, 'original_observations', 0) + 1
            return raw

        driver = Driver()
        result = await observe_native_window(driver, original)
        self.assertEqual(driver.calls, 2)
        self.assertEqual(driver.original_observations, 1)
        for key, value in raw.items():
            self.assertEqual(result[key], value)
        self.assertFalse(result['native_window_capture']['input_authorized'])
        return result

    async def test_native_bracket_preserves_original_image_and_missing_controls(self):
        result = await self.capture()
        self.assertTrue(result['native_window_capture']['screenshot_bracket_matched'])
        self.assertEqual(result['accessibility']['status'], 'foreground_unmatched')

    async def test_focus_change_between_native_reads_invalidates_bracket(self):
        result = await self.capture(change={'focus_window_id': 0x30000f8, 'focus_ancestry': [0x30000f8]})
        self.assertFalse(result['native_window_capture']['screenshot_bracket_matched'])

    async def test_native_timeouts_do_not_replay_or_replace_original_observation(self):
        for failed_read in (1, 2):
            with self.subTest(failed_read=failed_read):
                result = await self.capture(failure=failed_read)
                self.assertFalse(result['native_window_capture']['screenshot_bracket_matched'])


class NativePixelsTests(unittest.TestCase):
    def compare(self, changed_pixel=None, change=None):
        old = small_observation()
        fresh = copy.deepcopy(old)
        if change:
            change(fresh)
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / 'screenshots').mkdir()
            for index, metadata in enumerate((old, fresh)):
                image = Image.new('RGB', (8, 8), 'white')
                if index == 1 and changed_pixel:
                    image.putpixel(changed_pixel, (0, 0, 0))
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                metadata['sha256'] = hashlib.sha256(buffer.getvalue()).hexdigest()
                (runtime / 'screenshots' / (metadata['sha256'] + '.png')).write_bytes(buffer.getvalue())
            return exact_native_window_pixels(runtime, old, fresh)

    def test_external_clock_change_does_not_change_native_client_pixels(self):
        proof = self.compare(changed_pixel=(0, 0))
        self.assertTrue(proof['pixels_unchanged'])
        self.assertFalse(proof['input_authorized'])
        self.assertTrue(proof['requires_semantic_target_revalidation'])

    def test_change_anywhere_inside_window_remains_visible(self):
        self.assertIsNone(self.compare(changed_pixel=(3, 3)))

    def test_window_focus_or_process_change_refuses_equal_pixels(self):
        for change in [lambda v:v['native_window']['reported_owner'].update(start_ticks=999),
                       lambda v:v['native_window'].update(focus_within_active=False),
                       lambda v:v.update(session_id='new')]:
            self.assertIsNone(self.compare(change=change))

    def test_unbound_native_snapshot_does_not_prove_screenshot_geometry(self):
        self.assertIsNone(self.compare(change=lambda v:v.pop('native_window_capture')))
        self.assertIsNone(self.compare(change=lambda v:v['native_window_capture'].update(screenshot_bracket_matched=False)))


if __name__ == '__main__':
    unittest.main()
