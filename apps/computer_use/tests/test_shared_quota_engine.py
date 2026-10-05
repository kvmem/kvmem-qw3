import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import shared_quota_engine as q


SOURCE_PROTOCOL = q.ROOT / "apps/computer_use/.runtime/linux/kvmem-capacity-continuation-20261005/protocol.json"


class ProfileTests(unittest.TestCase):
    def test_common_quota_is_the_only_command_change_and_original_is_unchanged(self):
        before = {"command": ["binary", "serve", "--ctx", "524288"], "qw3_env": {"fixed": "7"}, "binary_sha256": "pin"}
        saved = copy.deepcopy(before)
        after = q.profile_identity(before)
        self.assertEqual(before, saved)
        self.assertEqual(after["command"], saved["command"] + q.RATIO_FLAG)
        self.assertEqual({k: v for k, v in after.items() if k != "command"}, {k: v for k, v in saved.items() if k != "command"})

    def test_existing_quota_is_not_silently_replaced(self):
        with self.assertRaises(ValueError):
            q.profile_identity({"command": ["binary", "--kvmem-gpu-memory-ratio", "0.5"]})

    def test_changed_source_shape_is_not_patched(self):
        with self.assertRaises(ValueError):
            q.patch_source("def main(): pass")

    def test_false_boolean_quota_is_rejected(self):
        with self.assertRaises(ValueError):
            q.qualify_profile({"quota_ratio": True})


@unittest.skipUnless(SOURCE_PROTOCOL.exists(), "Requires the original frozen local profile; qualification cannot skip")
class ActualLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.profile = q.prepare(SOURCE_PROTOCOL, self.root / "profile.json", self.root / "cpu.json")

    def tearDown(self):
        self.tmp.cleanup()

    def launch(self, mode, extra=()):
        out = self.root / (mode + str(len(list(self.root.iterdir()))))
        argv = [str(q.ROOT / "apps/computer_use/run_engine.py"), "--run-dir", str(out), "--method", "dflash2",
                "--kvmem", mode, "--prefix-reuse", "off", "--visual-history", "pin-all", "--guided-refresh", "off",
                "--context-tokens", "524288", "--kvmem-cpu-gib", "48", "--gpu-sharing", "spare-capacity",
                "--min-free-mib", "56000", *extra]
        def output(command, **kwargs):
            if command[0] == "git":
                return "fixture-source-commit\n"
            if "--query-compute-apps=pid" in command:
                return "99999\n"
            if "--query-gpu=memory.free" in command:
                return "60000\n"
            return "fixture-gpu,fixture,97887,37887\n"
        captured = []
        namespace = dict(vars(q.original))
        namespace.update(RATIO_FLAG=q.RATIO_FLAG, quota_profile=self.profile, quota_profile_sha="declared-profile",
                         quota_wrapper_sha="declared-wrapper")
        exec(compile(q.patch_source(inspect.getsource(q.original.main)), "cpu-actual-launcher", "exec"), namespace)
        listener = types.SimpleNamespace(bind=lambda address: None)
        socket_context = types.SimpleNamespace(__enter__=lambda s: listener, __exit__=lambda *a: None)
        class Socket:
            def __enter__(self):
                return listener
            def __exit__(self, *args):
                pass
        with patch.object(sys, "argv", argv), patch.object(q.original.subprocess, "check_output", side_effect=output), \
                patch.object(q.original.socket, "socket", return_value=Socket()), \
                patch.object(q.original.fcntl, "flock"), patch.object(q.original.os, "set_inheritable"), \
                patch.object(q.original.os, "dup2"), patch.object(q.original.os, "execve", side_effect=lambda *a: captured.append(a)):
            namespace["main"]()
        return json.loads((out / "manifest.json").read_text()), captured

    def test_actual_off_launcher_appends_the_same_quota_and_retains_environment(self):
        manifest, captured = self.launch("off")
        expected = self.profile["engine_identities"]["off"]
        self.assertEqual(manifest["command"], expected["command"])
        self.assertEqual(manifest["qw3_env"], expected["qw3_env"])
        self.assertEqual(len(captured), 1)
        self.assertEqual(manifest["shared_quota_profile_sha256"], "declared-profile")

    def test_actual_on_launcher_preserves_both_windows_and_generation_reserve(self):
        manifest, _ = self.launch("on")
        self.assertEqual(manifest["command"], self.profile["engine_identities"]["on"]["command"])
        for flag, value in (("--kvmem-budget", "131072"), ("--kvmem-prefill-budget", "131072"), ("--kvmem-gen-budget", "65536")):
            self.assertEqual(manifest["command"][manifest["command"].index(flag) + 1], value)
        self.assertEqual(manifest["min_free_mib"], 56000)

    def test_gpu_gate_cannot_be_lowered_or_trace_flags_added(self):
        for change in (("--min-free-mib", "1"), ("--trace-memory",), ("--trace-retrieval",)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.launch("on", change)

    def test_changed_context_history_prefix_or_cpu_capacity_is_rejected(self):
        for change in (("--context-tokens", "131072"), ("--prefix-reuse", "auto"),
                       ("--visual-history", "retrievable"), ("--kvmem-cpu-gib", "24")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.launch("on", change)

    def test_manifest_identity_changes_or_protocol_changes_are_rejected(self):
        for key in ("command", "binary_sha256", "qw3_env"):
            bad = copy.deepcopy(self.profile)
            bad["engine_identities"]["on"][key] = [] if key == "command" else "changed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                q.qualify_profile(bad)


if __name__ == "__main__":
    unittest.main()
