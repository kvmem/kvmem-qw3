"""Host-only checks for ordinary Office export and unchanged task criteria."""
from pathlib import Path
import tempfile
import unittest

from capability_regrade import grade_presentation
from osworld_windows.capabilities import public_inputs


class PresentationRegradeTests(unittest.TestCase):
    def test_reindexed_body_passes_but_missing_or_extra_body_fails(self):
        from pptx import Presentation
        from pptx.util import Inches
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'input'; public_inputs('cap07', root)
            deck = Presentation(root / 'briefing.pptx')
            deck.slides[1].shapes.title.text = 'Results'
            last = deck.slides._sldIdLst[-1]
            deck.part.drop_rel(last.rId); deck.slides._sldIdLst.remove(last)
            for slide in deck.slides:
                slide.placeholders[1]._element.xpath('.//p:ph')[0].set('idx', '0')
            output = Path(temp) / 'export.pptx'; deck.save(output)
            self.assertTrue(grade_presentation(output)['strict_success'])
            deck.slides[1].shapes[1].text = ''
            deck.save(output)
            self.assertFalse(grade_presentation(output)['strict_success'])
            deck.slides[1].shapes[1].text = 'Atlas completed 3 milestones.'
            deck.slides[0].shapes.add_textbox(Inches(1), Inches(1), Inches(1), Inches(1)).text = 'extra'
            deck.save(output)
            self.assertFalse(grade_presentation(output)['strict_success'])

    def test_unchanged_input_and_missing_output_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'input'; public_inputs('cap07', root)
            self.assertFalse(grade_presentation(root / 'briefing.pptx')['strict_success'])
            self.assertFalse(grade_presentation(root / 'missing.pptx')['strict_success'])


if __name__ == '__main__':
    unittest.main()
