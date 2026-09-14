"""Readiness retries must not alter the agent's runtime limits or frozen files."""
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import run_startup_retry_arm as retry


class StartupRetryChecks(unittest.TestCase):
    def test_only_readiness_constant_changes_and_arguments_are_forwarded(self):
        original=retry.aligned.arm
        path=Path(original.__file__)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        captured={}
        def inspect():
            captured['module']=retry.aligned.arm
            captured['argv']=sys.argv[:]
        argv=['retry','--startup-timeout-s','900','--out','preserved-attempt','--shared-gpu']
        with patch.object(retry.aligned,'arm',original), patch.object(retry.aligned,'main',inspect), patch.object(sys,'argv',argv):
            retry.main()
        self.assertIn(240,original.main.__code__.co_consts)
        self.assertIn(900,captured['module'].main.__code__.co_consts)
        self.assertNotIn(240,captured['module'].main.__code__.co_consts)
        self.assertEqual(captured['argv'],['retry','--out','preserved-attempt','--shared-gpu'])
        self.assertEqual(captured['module'].ROOT,original.ROOT)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)

    def test_readiness_override_is_bounded(self):
        for value in ['0','240','1801']:
            with patch.object(sys,'argv',['retry','--startup-timeout-s',value]):
                with self.assertRaises(ValueError):retry.main()


if __name__=='__main__':unittest.main()
