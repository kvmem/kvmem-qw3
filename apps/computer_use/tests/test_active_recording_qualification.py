"""Bounds and native evidence checks; no GUI, model or VM."""
import unittest
from unittest.mock import patch

import numpy as np
import qualify_active_linux_recording as qualification


class NativeEvidenceTests(unittest.TestCase):
    def test_marker_requires_real_positive_color_and_tolerates_codec_rounding(self):
        pixels=np.zeros((160,240,3),dtype=np.uint8);pixels[30:70,50:90]=(252,2,1)
        self.assertTrue(qualification.marker_matches(pixels,[50,30,40,40],[255,0,0]))
        self.assertFalse(qualification.marker_matches(pixels,[50,30,40,40],[0,255,0]))

    def test_off_screen_or_malformed_rectangle_cannot_prove_native_event(self):
        for rectangle in ([200,20,60,40],[-1,20,20,20],[20,20,0,40],[20.,20,40,40],[True,20,40,40]):
            with self.subTest(rectangle=rectangle):
                self.assertFalse(qualification.valid_rectangle(rectangle,240,160))

    def test_small_colored_patch_does_not_prove_a_full_marker(self):
        pixels=np.zeros((160,240,3),dtype=np.uint8);pixels[30:35,50:55]=(255,0,0)
        self.assertFalse(qualification.marker_matches(pixels,[50,30,40,40],[255,0,0]))

    def test_dedicated_vm_name_and_full_id_both_required(self):
        import json
        good=dict(Id='a'*64,Name='/qw3-recording-active-test',Image=qualification.environment.IMAGE,
                  State={'Running':True},Config={'Labels':{'qw3.goal':'cross-device-20261003'}})
        with patch.object(qualification.subprocess,'check_output',return_value=json.dumps([good])):
            qualification.owned_vm('qw3-recording-active-test','a'*64)
            with self.assertRaises(RuntimeError):qualification.owned_vm('qw3-recording-active-test','b'*64)
            with self.assertRaises(RuntimeError):qualification.owned_vm('qw3-computer-linux','a'*64)

    def test_marker_location_uses_visible_pixels_within_owned_window(self):
        pixels=np.zeros((240,320,3),dtype=np.uint8);pixels[90:150,80:140]=(255,0,0)
        self.assertEqual(qualification.locate_marker(pixels,[40,40,240,180],[255,0,0]),[90,100,40,40])
        with self.assertRaises(ValueError):qualification.locate_marker(pixels,[200,40,100,180],[255,0,0])

    def test_small_or_split_marker_cannot_supply_an_input_target(self):
        pixels=np.zeros((240,320,3),dtype=np.uint8);pixels[90:100,80:90]=(255,0,0)
        with self.assertRaises(ValueError):qualification.locate_marker(pixels,[40,40,240,180],[255,0,0])
        pixels[180:190,180:190]=(255,0,0)
        with self.assertRaises(ValueError):qualification.locate_marker(pixels,[40,40,240,180],[255,0,0])

    def test_visibility_wait_uses_new_pixels_before_returning_a_target(self):
        hidden=np.zeros((240,320,3),dtype=np.uint8);visible=hidden.copy();visible[90:150,80:140]=(255,0,0)
        samples=iter([(hidden,[40,40,240,180],[255,0,0]),(visible,[40,40,240,180],[255,0,0])])
        with patch.object(qualification.time,'monotonic',side_effect=[0,0,.01,.1,.11]),patch.object(qualification.time,'sleep') as sleep:
            rectangle,wait=qualification.wait_for_marker(lambda:next(samples))
        self.assertEqual(rectangle,[90,100,40,40]);self.assertEqual(wait['attempts'],2);sleep.assert_called_once_with(.1)

    def test_visibility_timeout_never_returns_an_input_target(self):
        hidden=np.zeros((240,320,3),dtype=np.uint8)
        with patch.object(qualification.time,'monotonic',side_effect=[0,0,16]),patch.object(qualification.time,'sleep') as sleep:
            with self.assertRaises(TimeoutError):qualification.wait_for_marker(lambda:(hidden,[40,40,240,180],[255,0,0]))
        sleep.assert_not_called()

    def test_visibility_wait_does_not_retry_invalid_window_bounds(self):
        hidden=np.zeros((240,320,3),dtype=np.uint8)
        with patch.object(qualification.time,'sleep') as sleep:
            with self.assertRaisesRegex(ValueError,'Invalid owned window bounds'):
                qualification.wait_for_marker(lambda:(hidden,[-1,40,240,180],[255,0,0]))
        sleep.assert_not_called()

    def test_late_positive_pixels_do_not_extend_the_visibility_deadline(self):
        visible=np.zeros((240,320,3),dtype=np.uint8);visible[90:150,80:140]=(255,0,0)
        with patch.object(qualification.time,'monotonic',side_effect=[0,0,16]):
            with self.assertRaises(TimeoutError):qualification.wait_for_marker(lambda:(visible,[40,40,240,180],[255,0,0]))


if __name__=='__main__':unittest.main()
