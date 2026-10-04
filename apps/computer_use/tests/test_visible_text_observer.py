import asyncio
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from visible_text_observer import identity, observe_text, positive_text_check, region_bounds, valid_lines, _read_bounded


def observation():
    return dict(desktop_id="device:session", observation_id="capture-one", session_id="session",
                display_id="primary", sha256="a" * 64, width=1080, height=2400, rotation=0)


def evidence(lines=None):
    return dict(status="ok", identity=identity(observation()), region=[0, 330, 1080, 2010],
                lines=lines or [dict(text="Request saved", confidence=0.99, bounds=[44, 748, 515, 814])])


class VisibleTextChecks(unittest.TestCase):
    def test_positive_match_is_separate_from_business_and_absence(self):
        result = positive_text_check("Request saved", evidence(), observation())
        self.assertTrue(result["passed"])
        self.assertFalse(result["business_acceptance"])
        self.assertFalse(result["absence_supported"])

    def test_exact_text_does_not_match_negation_case_punctuation_or_substring(self):
        for text in ("Not Request saved", "request saved", "Request saved!", "saved"):
            self.assertFalse(positive_text_check(text, evidence(), observation())["passed"])

    def test_wrapped_text_joins_only_adjacent_aligned_lines(self):
        lines = [dict(text="Prepare Python for the local demo. Keep", confidence=0.99,
                      bounds=[43, 1665, 894, 1724]),
                 dict(text="existing projects.", confidence=0.98, bounds=[48, 1731, 401, 1776])]
        expected = "Prepare Python for the local demo. Keep existing projects."
        self.assertTrue(positive_text_check(expected, evidence(lines), observation())["passed"])
        for bounds in ([650, 1731, 900, 1776], [48, 2100, 401, 2150], [48, 1690, 401, 1730]):
            changed = copy.deepcopy(lines)
            changed[1]["bounds"] = bounds
            self.assertFalse(positive_text_check(expected, evidence(changed), observation())["passed"])

    def test_exact_phrase_allows_whitespace_only(self):
        self.assertTrue(positive_text_check("  Request\n saved  ", evidence(), observation())["passed"])
        self.assertFalse(positive_text_check("+12025550186", evidence([
            dict(text="+1 202-555-0186", confidence=0.99, bounds=[40, 500, 900, 550])]), observation())["passed"])

    def test_any_changed_observation_identity_invalidates_evidence(self):
        for field, value in dict(desktop_id="different", observation_id="later", session_id="restarted",
                                 display_id="external", sha256="b" * 64, width=1200, height=2500, rotation=90).items():
            changed = observation() | {field:value}
            self.assertFalse(positive_text_check("Request saved", evidence(), changed)["passed"])

    def test_low_confidence_and_unavailable_are_not_positive_proof(self):
        low = evidence()
        low["lines"][0]["confidence"] = 0.94
        self.assertFalse(positive_text_check("Request saved", low, observation())["passed"])
        for status in ("timeout", "unavailable", "partial", None):
            self.assertFalse(positive_text_check("Request saved", evidence() | {"status":status}, observation())["passed"])

    def test_out_of_region_or_nonfinite_geometry_cannot_be_used(self):
        for bounds in ([44, 100, 515, 200], [44, 500, 1200, 550], [44, 500, float('nan'), 550], [515, 500, 44, 550]):
            changed = evidence()
            changed["lines"][0]["bounds"] = bounds
            self.assertFalse(positive_text_check("Request saved", changed, observation())["passed"])
        for value in (float('nan'), float('inf'), True, -0.1, 1.1):
            changed = evidence()
            changed["lines"][0]["confidence"] = value
            self.assertFalse(positive_text_check("Request saved", changed, observation())["passed"])

    def test_region_and_dimensions_are_explicit_bounded_integers(self):
        for region in ([], [0, 0, 1080.0, 2400], [True, 0, 100, 100], [0, 0, 1080, 2401], [-1, 0, 100, 100]):
            with self.assertRaises(ValueError):
                region_bounds(region, observation())
        for dimensions in ((10000, 10000), (True, 2400), (1080, 0)):
            with self.assertRaises(ValueError):
                identity(observation() | dict(zip(("width", "height"), dimensions)))


class VisibleTextWorkerBoundaries(unittest.IsolatedAsyncioTestCase):
    async def test_reader_rejects_excess_output(self):
        stream = asyncio.StreamReader()
        stream.feed_data(b'x' * 101)
        stream.feed_eof()
        with self.assertRaises(ValueError):
            await _read_bounded(stream, 100)

    async def test_bad_image_digest_is_unavailable_without_ocr_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.png'
            path.write_bytes(b'not the observed screenshot')
            result = await observe_text(path, observation(), Path(directory)/'missing-recipe.json')
            self.assertEqual(result['status'], 'unavailable')
            self.assertEqual(result['error_type'], 'ValueError')
            self.assertEqual(result['lines'], [])

    async def test_tiny_deadline_kills_only_the_owned_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.png'
            path.write_bytes(b'bounded worker input')
            source = observation() | {'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            result = await observe_text(path, source, Path(directory)/'missing-recipe.json', timeout_seconds=0.000001)
            self.assertEqual(result['status'], 'timeout')
            self.assertEqual(result['lines'], [])

    async def test_invalid_crop_is_rejected_before_worker_admission(self):
        with self.assertRaises(ValueError):
            await observe_text('/never-read', observation(), '/never-read', region=[0, 0, 1080, 2401])
        with self.assertRaises(ValueError):
            await observe_text('/never-read', observation(), '/never-read', region=[])


if __name__ == '__main__':
    unittest.main()
