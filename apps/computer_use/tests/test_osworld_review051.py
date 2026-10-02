"""Evidence-binding controls using synthetic fixtures, never model results."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from osworld_windows.review051 import CHECKS, apply, prepare
from osworld_windows.scenarios import collection
from osworld_windows.source import sha256


class ProcessReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.case = self.root / 'synthetic-case'
        desktop = self.case / 'guest/app/desktop'
        (desktop / 'screenshots').mkdir(parents=True)
        image = desktop / 'screenshots/temp.png'
        image.write_bytes(b'synthetic screenshot binding fixture')
        self.image = image.with_name(sha256(image) + '.png');image.rename(self.image)
        self.db = sqlite3.connect(str(desktop / 'desktop.sqlite'))
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE actions (id TEXT, task TEXT, request TEXT, status TEXT, result TEXT, created REAL)')
        self.db.execute('CREATE TABLE observations (id TEXT, task TEXT, metadata TEXT, created REAL)')
        metadata = dict(observation_id='observation-1', sha256=sha256(self.image))
        self.db.execute('INSERT INTO observations VALUES (?,?,?,?)', ('observation-1','fixture-task',json.dumps(metadata),1))
        request = dict(action_id='fixture-action', observation_id='observation-1', action={'kind':'click'})
        self.db.execute('INSERT INTO actions VALUES (?,?,?,?,?,?)',
            ('fixture-action','fixture-task',json.dumps(request),'completed',json.dumps({'observation':metadata}),2))
        self.db.commit()
        (self.case / 'guest/app/codex-events.jsonl').write_text('{"fixture_only":true}\n')
        artifacts = self.case / 'guest/artifacts';artifacts.mkdir()
        saved = artifacts / 'final.xcf';saved.write_bytes(b'fixture only; not an XCF')
        receipt = dict(infrastructure_ok=True, complete=True, files=[collection('051')['files'][0] |
            dict(status='copied', bytes=saved.stat().st_size, sha256=sha256(saved))])
        (artifacts / 'receipt.json').write_text(json.dumps(receipt))
        video = self.case / 'video';video.mkdir()
        (video / 'screen.mp4').write_bytes(b'fixture only; not a real video')
        (video / 'capture.jsonl').write_text('{"fixture_only":true}\n')
        self.ffmpeg = self.root / 'fixture-decoder';self.ffmpeg.write_bytes(b'not an executable')
        self.recording = dict(complete=True, video='screen.mp4', started_monotonic_ns=0,
            stopped_monotonic_ns=100, video_duration_seconds=10, ffmpeg_sha256=sha256(self.ffmpeg),
            artifacts={name:dict(bytes=(video/name).stat().st_size,sha256=sha256(video/name)) for name in ('screen.mp4','capture.jsonl')})
        self.report = dict(task='051',formal_run=False,pipeline_passed=True,
            native_result={'desktop_stopped':True,'task':{'task_id':'fixture-task'}},recording=self.recording,
            events=[dict(stage=name,host_monotonic_ns=number) for name,number in
                [('recording_started_before_task_launch',10),('native_task_launched',20),('native_task_terminal',90)]])
        self.write_report()

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def write_report(self):
        (self.case / 'result.json').write_text(json.dumps(self.report))
        (self.case / 'video/recording.json').write_text(json.dumps(self.recording))

    def package(self):
        self.output = self.root / 'package'
        return prepare(self.case, self.output)

    def full_decision_fixture(self):
        # Schema fixture only; it must never be applied to real trials.
        decision = json.loads((self.output / 'review-template.json').read_text())
        decision.update(reviewer='synthetic-binding-test',complete_trajectory_reviewed=True,
                        reviewed_video_end_seconds=10,checks={k:True for k in CHECKS},
                        supporting_evidence={k:'Synthetic schema fixture, no actual visual review.' for k in CHECKS})
        return decision

    def apply_decision(self, decision):
        path = self.root / 'decision.json';path.write_text(json.dumps(decision))
        return apply(self.output,path,self.root/'grade',self.ffmpeg)

    def test_all_wal_actions_are_packaged_without_claiming_review(self):
        manifest = self.package()
        self.assertEqual((manifest['actions'],manifest['observations']),(1,1))
        self.assertFalse(manifest['complete_trajectory_reviewed'])
        self.assertEqual(manifest['review_status'],'pending_full_visual_review')
        self.assertIn('guest/app/desktop/desktop.sqlite-wal',manifest['source_sha256'])
        rows = [json.loads(line) for line in (self.output/'trajectory.jsonl').read_text().splitlines()]
        self.assertEqual([r['kind'] for r in rows],['observation','action'])
        with self.assertRaisesRegex(ValueError,'complete evidence-bound review'):
            self.apply_decision(json.loads((self.output/'review-template.json').read_text()))

    def test_incomplete_recording_and_missing_terminal_coverage_rejected(self):
        self.recording['complete']=False;self.write_report()
        with self.assertRaises(ValueError):self.package()
        self.recording['complete']=True;self.recording['stopped_monotonic_ns']=80;self.write_report()
        with self.assertRaisesRegex(ValueError,'complete native launch/terminal'):self.package()

    def test_tampered_media_and_missing_screenshot_rejected(self):
        video=self.case/'video/screen.mp4';original=video.read_bytes();video.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Recording evidence changed'):self.package()
        video.write_bytes(original);self.image.unlink()
        with self.assertRaises(FileNotFoundError):self.package()

    def test_foreign_and_uncertain_action_rejected(self):
        self.db.execute("UPDATE actions SET status='uncertain'");self.db.commit()
        with self.assertRaisesRegex(ValueError,'Foreign or unresolved'):self.package()
        self.db.execute("UPDATE actions SET status='completed',task='another-task'");self.db.commit()
        with self.assertRaisesRegex(ValueError,'Foreign or unresolved'):self.package()

    def test_short_video_review_cannot_pass(self):
        self.package();decision=self.full_decision_fixture();decision['reviewed_video_end_seconds']=9
        with self.assertRaisesRegex(ValueError,'whole recording'):self.apply_decision(decision)

    def test_stale_package_and_changed_sources_cannot_pass(self):
        self.package();decision=self.full_decision_fixture();wrong=copy.deepcopy(decision);wrong['package_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'complete evidence-bound review'):self.apply_decision(wrong)
        (self.case/'guest/app/codex-events.jsonl').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Original process evidence changed'):self.apply_decision(decision)


if __name__ == '__main__':
    unittest.main()
