"""Compose the three isolated fixes into a new unqualified Windows bundle.

The successful SDK49 archive remains immutable. A new byte inventory and
archive must pass native qualification before this candidate may be deployed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

from durable_interrupt_candidate import patch_app_bytes
from foreground_visual_candidate import patch_desktop_bytes
from linux_dialog_candidate import patch_driver_bytes
from osworld_windows.bundle import verify_bundle


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def visual_contract_tests(app):
    path = app / "tests" / "test_desktop.py"
    text = path.read_text()
    before = '''    async def test_footer_extension_is_never_used_for_visual_points(self):
        action = await self.footer_control()
        action.update(coordinate_space='normalized_1000',
                      action={'kind': 'click', 'x': 750, 'y': 1000, 'button': 'left', 'clicks': 1})
        with self.assertRaises(DesktopRejected):
            await self.desktop.act(action)
        self.assertEqual(self.effects, 0)
        self.execution.close()
'''
    after = '''    async def test_visual_footer_requires_positive_whole_window_evidence(self):
        action = await self.footer_control()
        action.update(coordinate_space='normalized_1000',
                      action={'kind': 'click', 'x': 750, 'y': 1000, 'button': 'left', 'clicks': 1})
        try:
            result = await self.desktop.act(action)
            self.assertTrue(result['observation_revalidation']['visual_foreground_context_unchanged'])
            self.assertEqual(self.effects, 1)
        finally:
            self.execution.close()
'''
    if text.count(before) != 1:
        raise ValueError("Qualified original visual guard test changed")
    path.unlink()
    path.write_text(text.replace(before, after, 1))
    # Run the full set of new window-proof boundary cases in native Windows,
    # against the installed Desktop itself, without an in-test source patch.
    fixture = Path(__file__).parent / "tests" / "test_foreground_visual_candidate.py"
    boundaries = fixture.read_text().split("class VisualForegroundTests", 1)[1].split("class CandidateBuildTests", 1)[0]
    header = '''import types
import unittest
from unittest.mock import patch
from qw3_computer.desktop import Desktop, DesktopRejected
import test_desktop as fixtures
module = types.SimpleNamespace(Desktop=Desktop, DesktopRejected=DesktopRejected)
'''
    native_tests = header + "\nclass VisualForegroundTests" + boundaries
    compile(native_tests, "native-visual-window-contracts.py", "exec")
    (app / "tests" / "test_visual_window_contract.py").write_text(native_tests)


def build(source, output, archive):
    source, output, archive = (Path(p).resolve() for p in (source, output, archive))
    proof = verify_bundle(source)
    if output == source or output.is_relative_to(source):
        raise ValueError("Candidate must be outside the immutable qualified bundle")
    if output.exists() or archive.exists():
        raise FileExistsError("Do not replace a candidate or archive")
    # Share immutable runtime binaries. Unlink a destination before changing
    # its bytes; writing into a hardlink would corrupt the original bundle.
    shutil.copytree(source, output, copy_function=os.link,
                    ignore=shutil.ignore_patterns("archive.json", "qualification.json", "native-client49-*", "__pycache__"))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    patches = {"app.py": patch_app_bytes, "desktop.py": patch_desktop_bytes,
               "linux_driver.py": patch_driver_bytes}
    for name, patch in patches.items():
        path = output / "app" / "qw3_computer" / name
        before = path.read_bytes()
        after = patch(before)
        path.unlink()
        path.write_bytes(after)
        assert (source / "app" / "qw3_computer" / name).read_bytes() == before
    app = output / "app"
    visual_contract_tests(app)
    manifest.update(qualified=False, candidate_kind="combined-isolated-execution-fixes",
        qualified_base_archive_sha256=proof["archive"]["sha256"],
        qualified_base_manifest_sha256=proof["manifest_sha256"],
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        recipe_sha256=digest(__file__), candidate_created_at=time.time(),
        candidate_changes=["durable_model_turn_status", "gtk_file_chooser_observation", "positive_linux_visual_window_context"],
        candidate_test_changes=["replace_original_region_only_visual_rule_with_whole_window_proof", "add_twelve_window_proof_boundary_tests"],
        source_files={str(p.relative_to(app)): digest(p) for p in app.rglob("*") if p.is_file()})
    manifest_path.unlink()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as writer:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                writer.write(path, str(path.relative_to(output)))
    record = {"path": str(archive), "sha256": digest(archive), "bytes": archive.stat().st_size}
    (output / "archive.json").write_text(json.dumps(record, indent=2) + "\n")
    (archive.with_suffix(".sha256")).write_text(record["sha256"] + "\n")
    # Revalidate the unchanged original after patching and staging the archive.
    verify_bundle(source)
    return {"qualified": False, "real_model_trials": 0, "device_inputs": 0,
            "output": str(output), "archive": record, "changed_core_files": sorted(patches)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.archive)))


if __name__ == "__main__":
    main()
