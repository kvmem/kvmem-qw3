"""Host-only presentation evaluator correction; never staged into the VM.

LibreOffice can export title and body placeholders with the same idx. Grade the
actual non-title text shapes, not an internal placeholder number. Original run
and grade files remain unchanged; this writes a provenance-bound derived grade.
"""
import argparse
import json
from pathlib import Path

from osworld_windows.scenarios import verify_collection
from osworld_windows.source import sha256


def grade_presentation(path):
    from pptx import Presentation
    path = Path(path)
    checks = {'output_saved': path.is_file()}
    report = dict(task='cap07', suite='custom_capability_calibration', formal_run=False,
                  strict_success=False, checks=checks, grader_revision='presentation-text-v2')
    if not path.is_file():
        return report
    try:
        deck = Presentation(path)
        titles, bodies = [], []
        for slide in deck.slides:
            title = slide.shapes.title
            titles.append(title.text if title is not None else None)
            bodies.append([shape.text for shape in slide.shapes
                if shape.has_text_frame and shape.text.strip()
                and (title is None or shape.shape_id != title.shape_id)])
        checks.update(two_slides=len(deck.slides) == 2,
            titles=titles == ['Briefing', 'Results'],
            body_preserved=bodies == [['Owner: Mei'], ['Atlas completed 3 milestones.']])
        report.update(actual_titles=titles, actual_body_texts=bodies,
                      output_sha256=sha256(path), strict_success=all(checks.values()),
                      partial_check_fraction=sum(checks.values()) / len(checks))
    except Exception as exc:
        report.update(failure='invalid_saved_output', error=f'{type(exc).__name__}: {exc}')
    return report


def regrade(directory):
    directory = Path(directory).resolve()
    original_path = directory / 'result.json'
    original = json.loads(original_path.read_text())
    if original['task'] != 'cap07':
        raise ValueError('This correction applies only to the presentation evaluator')
    artifacts = directory / 'guest/artifacts'
    verify_collection('cap07', artifacts)
    report = grade_presentation(artifacts / 'cap07.pptx')
    report.update(original_result_sha256=sha256(original_path), original_status=original['status'],
        original_grade=original['independent_grade'], original_run=original['run'],
        grader_source_sha256=sha256(Path(__file__)),
        reason='LibreOffice renumbers body placeholders; actual slide text requirements unchanged')
    output = directory / 'independent-grade-v2.json'
    if output.exists():
        raise ValueError('Derived grade already exists; do not overwrite it')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    args = parser.parse_args()
    for directory in args.run:
        result = regrade(directory)
        print(json.dumps(dict(run=result['original_run'], strict_success=result['strict_success'],
                              original_status=result['original_status'])))
