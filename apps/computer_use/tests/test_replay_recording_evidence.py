"""Recording provenance, fresh-frame availability and partial-media boundaries."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from cross_device_replay import export
from replay_recording_evidence import digest, recording_evidence, retained_mkv


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.directory = self.root/'video'; self.directory.mkdir()
        self.media = self.directory/'screen.mp4'; self.media.write_bytes(b'original terminal media')
        self.start = 10_000_000_000
        captures = [dict(capture=0, available_from_frame=1, captured_monotonic_ns=self.start+200_000_000),
                    dict(capture=1, available_from_frame=6, captured_monotonic_ns=self.start+2_200_000_000)]
        self.capture_path = self.directory/'capture.jsonl'
        self.capture_path.write_text(''.join(json.dumps(c)+'\n' for c in captures))
        self.receipt = self.directory/'recoverable-recording.json'
        self.data = dict(kind='isolated_bounded_recoverable_vnc_recording',clock='host_monotonic_ns',
            complete=True,fresh_sampling_uninterrupted=False,fps=2,video_frames=10,captures=2,
            started_monotonic_ns=self.start,console_host='private do not publish',
            encoder_command=['private token do not publish'],model_calls=0,
            artifacts={'screen.mp4':{'sha256':digest(self.media),'bytes':self.media.stat().st_size},
                       'capture.jsonl':{'sha256':digest(self.capture_path),'bytes':self.capture_path.stat().st_size}},
            transport_gaps=[dict(from_monotonic_ns=self.start+200_000_000,
                until_monotonic_ns=self.start+2_100_000_000,recovered=True,fresh_capture=1,
                held_frames_are_new_observations=False,missing_pixels_reconstructed=False)])
        self.save()

    def save(self): self.receipt.write_text(json.dumps(self.data))
    def tearDown(self): self.temp.cleanup()

    def evidence(self): return recording_evidence(self.root,self.media,5.0)

    def test_gap_remains_until_new_image_is_visible_in_encoded_video(self):
        result = self.evidence()
        self.assertEqual(result['recording_evidence_status'],'verified_recovery_timeline')
        self.assertEqual(result['video_observation_gaps'],[dict(start_seconds=.2,end_seconds=3.,
            recovered=True,displayed_frames_are_stale=True)])
        self.assertFalse(result['video_usable_for_full_task_trace'])
        self.assertNotIn('private',json.dumps(result))
        self.assertNotIn('model_task_success',result)

    def test_changed_movie_cannot_qualify_the_original_timeline(self):
        self.media.write_bytes(b'changed media')
        result = self.evidence()
        self.assertEqual(result['recording_evidence_status'],'invalid')
        self.assertEqual(result['video_observation_gaps'],[])

    def test_changed_capture_log_is_not_used(self):
        self.capture_path.write_text(self.capture_path.read_text()+'{}\n')
        self.assertEqual(self.evidence()['recording_evidence_status'],'invalid')

    def test_wrong_clock_and_early_fresh_frame_are_rejected(self):
        self.data['clock']='guest clock'; self.save()
        self.assertEqual(self.evidence()['recording_evidence_status'],'invalid')
        self.data['clock']='host_monotonic_ns'
        self.data['transport_gaps'][0]['until_monotonic_ns']=self.start+3_000_000_000;self.save()
        self.assertEqual(self.evidence()['recording_evidence_status'],'invalid')

    def test_encoding_success_does_not_hide_a_gap(self):
        self.data['fresh_sampling_uninterrupted']=True;self.save()
        self.assertEqual(self.evidence()['recording_evidence_status'],'invalid')

    def test_unsupported_or_excessive_gap_records_fail_closed(self):
        self.data['transport_gaps']*=12;self.save()
        self.assertEqual(self.evidence()['recording_evidence_status'],'invalid')
        self.data['transport_gaps']=[];self.data['kind']='unknown';self.save()
        self.assertEqual(self.evidence()['recording_evidence_status'],'unsupported')

    def test_decoded_clock_disagreement_is_not_a_verified_gap(self):
        self.assertEqual(recording_evidence(self.root,self.media,50.)['recording_evidence_status'],'invalid')

    def test_partial_mkv_requires_original_failed_receipt_and_hash(self):
        self.receipt.unlink()
        mkv=self.directory/'screen.mkv';mkv.write_bytes(b'original failed source')
        data=dict(kind='continuous_read_only_vnc_recording',complete=False,
                  artifacts={'screen.mkv':{'sha256':digest(mkv),'bytes':mkv.stat().st_size}})
        (self.directory/'recording.json').write_text(json.dumps(data))
        self.assertEqual(retained_mkv(self.root)[0],mkv)
        mkv.write_bytes(b'changed source')
        self.assertIsNone(retained_mkv(self.root))


class ExportMediaTests(unittest.TestCase):
    def test_real_partial_remux_preserves_failed_grade_and_original_mkv(self):
        import imageio_ffmpeg
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'failed-trial';(root/'video').mkdir(parents=True)
            mkv=root/'video/screen.mkv'
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-hide_banner','-loglevel','error','-n',
                '-f','lavfi','-i','color=c=green:s=16x16:r=2','-t','1','-c:v','libx264','-pix_fmt','yuv420p',str(mkv)],
                check=True,timeout=10)
            original=digest(mkv)
            (root/'video/recording.json').write_text(json.dumps(dict(kind='continuous_read_only_vnc_recording',
                complete=False,artifacts={'screen.mkv':{'bytes':mkv.stat().st_size,'sha256':original}})))
            result=dict(accepted=False,artifact_passed=False,app_status='stopped',seconds=60,video_bytes=0,
                        private_request='credential must stay private')
            (root/'integration-result.json').write_text(json.dumps(result))
            (root/'pilot-manifest.json').write_text(json.dumps({'case':'linux_browser','video_mode':'vnc'}))
            # A failed native recorder can leave an empty destination. It
            # must not hide its independently verified retained MKV fragment.
            (root/'continuous.mp4').touch()
            result_sha=digest(root/'integration-result.json')
            summary=export(None,[root],Path(folder)/'public',18156)
            row=summary['attempts'][0]
            self.assertTrue(row['partial_video']);self.assertTrue(row['video'].endswith('-partial.mp4'))
            self.assertFalse(row['accepted']);self.assertFalse(row['artifact_passed'])
            self.assertEqual(row['video_bytes'],0)
            self.assertEqual(row['original_recording_video_sha256'],original)
            self.assertEqual(digest(mkv),original)
            self.assertEqual(digest(root/'integration-result.json'),result_sha)
            self.assertNotIn('credential must stay private',json.dumps(summary))

    def test_component_probe_does_not_enter_model_or_formal_denominators(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'probe';(root/'recording').mkdir(parents=True)
            media=root/'recording/screen.mp4';media.write_bytes(b'\0\0\0\x20ftypxxxx')
            (root/'recording/recoverable-recording.json').write_text(json.dumps(dict(
                artifacts={'screen.mp4':{'bytes':media.stat().st_size}},kind='unqualified fixture')))
            result=dict(started_at=10,finished_at=12,status='passed',passed=True,model_calls=0,
                native_device_inputs=0,native_device_task_or_qw3_retest=False,private='must not publish')
            (root/'state.json').write_text(json.dumps(result))
            summary=export(None,[],Path(folder)/'public',18156,recording_probes=[root])
            self.assertEqual(summary['recording_qualification_attempts'],1)
            self.assertEqual(summary['formal_summary']['terminal_attempts'],0)
            self.assertEqual(summary['development_attempts'],0)
            self.assertEqual(summary['development_osworld_attempts'],0)
            self.assertTrue(summary['attempts'][0]['component_passed'])
            self.assertIsNone(summary['attempts'][0]['accepted'])
            self.assertNotIn('must not publish',json.dumps(summary))


if __name__=='__main__':unittest.main()
