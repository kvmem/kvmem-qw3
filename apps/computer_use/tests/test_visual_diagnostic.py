"""Diagnostic guardrails, not a test of model visual competence."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from visual_diagnostic_run import request_for, score
from visual_coordinate_run import grade
from qw3_computer.coordinates import normalized_point


class VisualDiagnosticTests(unittest.TestCase):
    def test_coordinate_contract_is_explicit_and_dimensions_are_applied_once(self):
        case = {"kind": "point", "expected_box": [68, 504, 172, 547]}
        self.assertFalse(grade(case, '{"x":93,"y":654}', "pixels", (1280, 800))["passed"])
        self.assertFalse(grade(case, '{"x":93,"y":654}', "normalized_1000", (1280, 800))["passed"])
        self.assertTrue(grade(case, '{"coordinate_space":"normalized_1000","x":93,"y":654}',
                              "normalized_1000", (1280, 800))["passed"])
        self.assertEqual(normalized_point(1000, 1000, 1280, 800), {"x": 1279, "y": 799})
        self.assertEqual(normalized_point(500, 500, 1920, 1080), {"x": 960, "y": 540})
        for value in (True, -1, 1001, 1.5, "93"):
            with self.assertRaises(ValueError): normalized_point(value, 654, 1280, 800)

    def test_ground_truth_never_enters_prompt_and_crop_is_target_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screen.png"; Image.new("RGB", (1280, 800), "white").save(path)
            first = {"question": "Read the selected label.", "answer_schema": {"text": "string"},
                     "expected": {"text": "HELDOUT_ANSWER_NEVER_SENT"}, "expected_box": [1, 2, 3, 4]}
            original, original_areas = request_for(first, path, "original")
            enlarged, areas = request_for(first, path, "quadrants")
            changed, other_areas = request_for(first | {"expected_box": [500, 400, 600, 500]}, path, "quadrants")
            self.assertNotIn("HELDOUT_ANSWER_NEVER_SENT", json.dumps(enlarged))
            self.assertNotIn("HELDOUT_ANSWER_NEVER_SENT", json.dumps(original))
            self.assertEqual(enlarged, changed)
            self.assertEqual(areas, other_areas)
            self.assertEqual(sum(a["input_pixels"] for a in areas), 4 * original_areas[0]["input_pixels"])

    def test_invalid_and_wrong_coordinate_answers_cannot_pass(self):
        case = {"kind": "point", "expected_box": [300, 200, 400, 250]}
        self.assertTrue(score(case, '{"x":350,"y":220}')["passed"])
        for text in ('{"x":0.5,"y":0.4}', '{"x":true,"y":220}', '{"x":350,"y":NaN}',
                     '{"x":350,"y":250}', 'I would click 350,220.', '{"x":350,"y":220,"extra":1}'):
            with self.subTest(text=text): self.assertFalse(score(case, text)["passed"])

    def test_text_and_state_errors_remain_scored_failures(self):
        text = {"kind": "text", "expected": {"text": "18.75"}}
        self.assertTrue(score(text, '{"text":"18.75"}')["passed"])
        self.assertFalse(score(text, '{"text":"18.15"}')["passed"])
        self.assertEqual(score(text, '{"text":"18.15"}')["character_edits"], 1)
        state = {"kind": "state", "expected": {"enabled": True}}
        self.assertFalse(score(state, '{"enabled":1}')["passed"])


if __name__ == "__main__":
    unittest.main()
