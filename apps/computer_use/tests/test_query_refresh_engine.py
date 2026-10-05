import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import query_refresh_engine as engine


COMMON_PROFILE = engine.ROOT / 'apps/computer_use/.runtime/shared-quota-profile-20261005/profile.json'


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.path = self.root / 'profile.json'
        self.profile = engine.prepare(COMMON_PROFILE, self.path, self.root / 'qualification.json')

    def tearDown(self):
        self.tmp.cleanup()

    def launch(self, mode, extra=()):
        output = self.root / ('run-' + str(len(list(self.root.iterdir()))))
        argv = ['original-launcher', '--run-dir', str(output), '--method', 'dflash2', '--kvmem', 'on',
                '--prefix-reuse', 'off', '--visual-history', 'pin-all', '--guided-refresh', mode,
                '--context-tokens', '524288', '--gpu-sharing', 'spare-capacity', '--min-free-mib', '56000', *extra]
        captured = []
        def result(command, **kwargs):
            if command[0] == 'git': return 'fixture-commit\n'
            if '--query-compute-apps=pid' in command: return '99999\n'
            if '--query-gpu=memory.free' in command: return '60000\n'
            return 'fixture-gpu,fixture,97887,37887\n'
        class Socket:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def bind(self, address): self.address = address
        runtime = engine.namespace(self.profile, mode, engine.common.digest(self.path))
        original = engine.common.original
        with patch.object(sys, 'argv', argv), patch.object(original.socket, 'socket', return_value=Socket()), \
                patch.object(original.subprocess, 'check_output', side_effect=result), \
                patch.object(original.fcntl, 'flock'), patch.object(original.os, 'set_inheritable'), \
                patch.object(original.os, 'dup2'), patch.object(original.os, 'execve', side_effect=lambda *args: captured.append(args)):
            runtime['main']()
        return json.loads((output / 'manifest.json').read_text()), captured

    def test_original_on_profile_is_retained_for_control(self):
        parent = engine.validate(self.profile)
        self.assertEqual(self.profile['engine_identities']['off'], parent['engine_identities']['on'])
        manifest, calls = self.launch('off')
        self.assertEqual(manifest['command'], parent['engine_identities']['on']['command'])
        self.assertEqual(len(calls), 1)

    def test_both_mode_is_the_only_engine_difference(self):
        before = self.profile['engine_identities']['off']
        after = self.profile['engine_identities']['both']
        command = list(after['command']); index = command.index('--kvmem-guided-reselect')
        self.assertEqual(command[index + 1], 'both'); del command[index:index + 2]
        self.assertEqual(command, before['command'])
        self.assertEqual(after['qw3_env'], before['qw3_env'])
        manifest, _ = self.launch('both')
        self.assertEqual(manifest['command'], after['command'])
        self.assertEqual(manifest['refresh_mode'], 'both')
        self.assertTrue(manifest['saved_input_only'])

    def test_resource_admission_cannot_be_lowered(self):
        for extra in (('--min-free-mib', '1'), ('--kvmem-cpu-gib', '24'), ('--context-tokens', '131072')):
            with self.subTest(extra=extra), self.assertRaises(ValueError): self.launch('both', extra)

    def test_history_cache_or_trace_changes_are_rejected(self):
        for extra in (('--prefix-reuse', 'auto'), ('--visual-history', 'retrievable'), ('--trace-retrieval',), ('--trace-memory',)):
            with self.subTest(extra=extra), self.assertRaises(ValueError): self.launch('both', extra)

    def test_selected_mode_and_on_state_cannot_be_overridden(self):
        for extra in (('--guided-refresh', 'off'), ('--kvmem', 'off')):
            with self.subTest(extra=extra), self.assertRaises(ValueError): self.launch('both', extra)

    def test_incoming_qw3_environment_does_not_change_the_effective_profile(self):
        with patch.dict(os.environ, {'QW3_KVMEM_RECOMPUTE_QUERY': '0', 'QW3_DFLASH_PRIVATE_MAX_DRAFT_TOKENS': '1'}):
            manifest, calls = self.launch('both')
        self.assertEqual(manifest['qw3_env'], self.profile['engine_identities']['both']['qw3_env'])
        self.assertNotIn('QW3_KVMEM_RECOMPUTE_QUERY', calls[0][2])

    def test_changed_window_environment_or_binary_identity_is_rejected(self):
        for key, value in (('command', ['changed']), ('qw3_env', {}), ('binary_sha256', 'changed')):
            profile = copy.deepcopy(self.profile); profile['engine_identities']['both'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): engine.validate(profile)

    def test_existing_profile_is_never_overwritten(self):
        with self.assertRaises(FileExistsError): engine.prepare(COMMON_PROFILE, self.path, self.root / 'qualification.json')

    def test_source_anchor_changes_and_incomplete_qualification_stop_before_launch(self):
        with patch.object(engine.inspect, 'getsource', return_value='def main(): pass'), self.assertRaises(ValueError):
            engine.patched_source()
        qualification = dict(kind='query_refresh_engine_cpu_qualification', passed=True, skipped=0,
                             tests_run=9, model_calls=0, device_inputs=0, source_sha256={})
        (self.root / 'qualification.json').write_text(json.dumps(qualification))
        with patch.object(engine, 'namespace') as dispatched, self.assertRaises(ValueError):
            engine.launch(self.path, 'both', [])
        dispatched.assert_not_called()


if __name__ == '__main__':
    unittest.main()
