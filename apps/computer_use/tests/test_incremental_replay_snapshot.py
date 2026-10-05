import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from PIL import Image
import incremental_replay_snapshot as q


class SnapshotTests(unittest.TestCase):
    def test_additions_do_not_change_old_scores_fields_or_formal_denominator(self):
        old=dict(attempts=[dict(kind="formal-osworld",attempt="013-old",original_score=0,video_sha256="old"),
                           dict(kind="development-osworld",cohort="old",attempt="013-new",original_score=None)],
                 formal_summary=dict(terminal_attempts=1,available_original_scores=1,original_passes=0))
        original=copy.deepcopy(old); new=[dict(kind="development-osworld",cohort="new",attempt="035-off",original_score=None)]
        data=q.combine(old,new);self.assertEqual(old,original);self.assertEqual(data["formal_summary"],old["formal_summary"])
        for row in old["attempts"]:self.assertIn(row,data["attempts"])
        self.assertEqual(data["development_osworld_attempts"],2)

    def test_duplicate_attempt_cannot_replace_an_existing_failed_grade(self):
        row=dict(kind="development-osworld",cohort="old",attempt="013",original_score=0)
        with self.assertRaises(ValueError):q.combine(dict(attempts=[row]),[row|dict(original_score=1)])

    def test_startup_event_cannot_enter_model_attempt_denominator(self):
        with self.assertRaises(ValueError):q.combine(dict(attempts=[]),[dict(kind="engine-startup",attempt="035-on")])

    def test_only_video_basename_assets_can_be_served(self):
        for value in ("../private/token.mp4","/tmp/token.mp4","trial-1.mp4/../token","mcp-token.txt",None):
            with self.subTest(value=value),self.assertRaises(ValueError):q.video_name(value)
        self.assertEqual(q.video_name("trial-42-partial.mp4"),"trial-42-partial.mp4")

    def test_changed_or_symlinked_movie_is_rejected_before_link(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);previous=root/"previous";output=root/"output";previous.mkdir();output.mkdir()
            (previous/"private.mp4").write_bytes(b"private");(previous/"trial-1.mp4").symlink_to(previous/"private.mp4")
            with self.assertRaises(ValueError):q.retain_movies(previous,output,[dict(video="trial-1.mp4",video_sha256=q.replay.digest(previous/"private.mp4"))])

    def test_qualification_must_match_actual_browser_and_both_public_files(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);data=dict(kind="public-terminal-attempts-snapshot",contains_private_requests=False,contains_credentials=False,attempts=[])
            (root/"summary.json").write_text(json.dumps(data));(root/"index.html").write_text("public")
            qual=dict(passed=True,actual_browser=True,checks_passed=24,model_calls=0,device_inputs=0,page_errors=[],
                      snapshot_summary_sha256=q.replay.digest(root/"summary.json"),snapshot_html_sha256=q.replay.digest(root/"index.html"))
            path=root/"qualification.json";path.write_text(json.dumps(qual));self.assertEqual(q.qualified_previous(root,path),data)
            (root/"index.html").write_text("modified")
            with self.assertRaises(ValueError):q.qualified_previous(root,path)

    def test_startup_notes_use_text_nodes_and_keep_existing_pair_handler(self):
        html='<p id="summary"></p><script>all=data.attempts;const s=data.formal_summary;function pairNotice(){}</script>'
        updated=q.annotate_startup(html)
        self.assertIn("p.textContent=",updated);self.assertNotIn("innerHTML",updated)
        self.assertIn("function pairNotice()",updated);self.assertIn("未计入模型成功率",updated)

    def geometry(self,root,source_size,movie_size):
        desktop=root/"app/desktop";screens=desktop/"screenshots";screens.mkdir(parents=True)
        source=screens/"temp.png";Image.new("RGB",source_size).save(source);sha=q.replay.digest(source);source.rename(screens/(sha+".png"))
        c=sqlite3.connect(desktop/"desktop.sqlite");c.execute("CREATE TABLE observations(metadata TEXT)")
        c.execute("INSERT INTO observations VALUES(?)",(json.dumps(dict(sha256=sha,width=source_size[0],height=source_size[1])),));c.commit();c.close()
        row=dict(video_dimensions=list(movie_size),original_score=0);proof=q.observed_geometry(root,row)
        return row,proof

    def test_actual_retained_png_size_mismatch_marks_incomplete_trace_without_regrading(self):
        with tempfile.TemporaryDirectory() as name:
            row,proof=self.geometry(Path(name),(32,24),(16,12));self.assertTrue(proof["mismatch"])
            self.assertFalse(row["full_screen_coverage"]);self.assertEqual(row["original_score"],0)

    def test_matching_sizes_do_not_claim_full_task_pixel_alignment(self):
        with tempfile.TemporaryDirectory() as name:
            row,proof=self.geometry(Path(name),(32,24),(32,24));self.assertTrue(row["retained_observation_dimensions_match_video"])
            self.assertNotIn("full_screen_coverage",row);self.assertFalse(proof["full_task_pixel_alignment_proven"])

    def test_changed_retained_png_is_not_accepted_as_geometry_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.geometry(root,(32,24),(32,24));source=next((root/"app/desktop/screenshots").glob("*.png"));source.write_bytes(b"changed")
            with self.assertRaises(ValueError):q.observed_geometry(root,dict(video_dimensions=[32,24]))


if __name__=="__main__":unittest.main()
