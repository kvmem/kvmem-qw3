import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from qw3_computer.desktop import DesktopError
from qw3_computer.software import Software, atomic_json, sha256
from qw3_computer.software_latex import missing_file, package_for_file


class LatexTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.software = Software(self.root / 'workspace')
        self.env = patch.dict(os.environ, SYSTEMROOT=str(self.root / 'Windows'))
        self.env.start()
        fonts = self.root / 'Windows/Fonts'; fonts.mkdir(parents=True)
        for asset in self.software.latex_catalog['fonts']:
            (fonts / asset['file']).write_bytes(b'neutral fixture font')
        self.software.download = AsyncMock(return_value={'cache_hit': True})
        self.project = self.software.path('projects', 'paper')
        self.project.mkdir()
        self.source = self.project / 'main.tex'
        self.source.write_text('existing user document', encoding='utf-8')

    async def asyncTearDown(self):
        await self.software.close()
        self.env.stop()
        self.temp.cleanup()

    def start(self, operation='setup_latex', **changes):
        args = {'operation': operation, 'action_id': 'latex-attempt-001', 'profile': 'latex-chinese-math', 'project': 'paper'} | changes
        with patch('qw3_computer.software.sys.platform', 'win32'):
            return self.software.start('task-1', args)

    def managed(self, status='ready'):
        from qw3_computer import software_latex
        owner = dict(version=1, profile='latex-chinese-math', catalog_sha256=sha256(Path(software_latex.__file__).with_name('latex_catalog.json')))
        atomic_json(self.project / '.qw3-latex-project.json', owner)
        atomic_json(self.software.path('latex', 'paper.json'), owner | {'status': status})
        bins = self.software.path('latex', 'paper', 'texmfs', 'install', 'miktex', 'bin', 'x64')
        bins.mkdir(parents=True)
        (bins / 'xelatex.exe').write_bytes(b'neutral fake executable')
        return bins

    def archive(self, package='ctex', filename='ctexart.cls'):
        path = self.software.path('downloads', package + '.tar.lzma')
        with tarfile.open(path, 'w:xz') as archive:
            info = tarfile.TarInfo('texmf/tex/latex/' + filename); info.size = 4
            archive.addfile(info, io.BytesIO(b'test'))
        return {'package': package, 'filename': path.name, 'sha256': sha256(path)}

    async def test_operation_profile_mismatch_cannot_install(self):
        with self.assertRaises(DesktopError): self.start(profile='python-web')
        with self.assertRaises(DesktopError): self.start(operation='setup_python')
        self.assertEqual(self.software.jobs, {})

    async def test_installer_exit_zero_without_executable_is_uncertain(self):
        self.software.command = AsyncMock(return_value='')
        first = self.start(); await self.software.tasks[first['id']]
        job = self.software.jobs[first['id']]
        self.assertEqual(job['status'], 'uncertain')
        self.assertTrue(job['needs_reconciliation'])
        self.assertEqual(self.source.read_text(), 'existing user document')
        with self.assertRaises(DesktopError):
            await self.software.invoke('task-1', {'operation': 'resume_job', 'job_id': first['id']})
        self.assertEqual(self.software.command.await_count, 1)

    async def test_incomplete_install_receipt_cannot_be_restarted(self):
        self.managed('installing')
        self.software.command = AsyncMock()
        first = self.start(); await self.software.tasks[first['id']]
        self.assertEqual(self.software.jobs[first['id']]['status'], 'uncertain')
        self.software.command.assert_not_called()

    async def test_unknown_project_repair_preserves_source_and_never_dispatches(self):
        self.software.command = AsyncMock()
        first = self.start('repair_latex'); await self.software.tasks[first['id']]
        self.assertEqual(self.software.jobs[first['id']]['status'], 'failed')
        self.software.command.assert_not_called()
        self.assertEqual(self.source.read_text(), 'existing user document')

    async def test_repair_uses_observed_missing_file_and_fresh_pdf_outputs(self):
        self.managed()
        self.software.latex_catalog['repository']['assets'] = [self.archive()]
        self.software.latex_catalog['repository']['metadata'] = []
        seen = []
        async def command(job, tag, argv, **kwargs):
            seen.append((tag, list(map(str, argv))))
            job['last_process'] = {'exit_code': 0}
            if tag == 'latex-version': return 'MiKTeX-XeTeX 4.16 (MiKTeX 25.12 Portable)\n'
            if tag == 'diagnose-latex':
                job['last_process']['exit_code'] = 1
                return "! LaTeX Error: File `ctexart.cls' not found."
            if tag.startswith('repair-state'): return 'ctex|true\n'
            if tag.startswith(('repair-check', 'verify-latex')):
                folder = Path(next(str(x).split('=', 1)[1] for x in argv if str(x).startswith('--output-directory=')))
                (folder / 'main.pdf').write_bytes(b'%PDF-1.5\n' + b'neutral test placeholder ' * 30)
                (folder / 'main.log').write_text('normal compiler fixture')
            return ''
        self.software.command = command
        first = self.start('repair_latex'); await self.software.tasks[first['id']]
        job = self.software.jobs[first['id']]
        self.assertEqual(job['status'], 'completed', job.get('error'))
        self.assertEqual(job['diagnosis']['missing_file'], 'ctexart.cls')
        self.assertTrue(job['verification']['passed'])
        self.assertEqual(self.source.read_text(), 'existing user document')
        self.assertEqual(len({r['pdf'] for r in job['verification']['fresh_compilations']}), 2)
        self.assertTrue((self.project / 'open-latex-terminal.cmd').is_file())
        self.assertTrue((self.project / 'compile-latex.cmd').is_file())
        self.assertEqual([a[-1] for t, a in seen if t.startswith(('restore-package', 'remove-damaged'))], ['ctex', 'ctex'])
        self.assertFalse(any(t == 'install-latex' for t, _ in seen))

    async def test_unknown_missing_package_fails_without_package_changes(self):
        self.managed()
        self.software.latex_catalog['repository']['assets'] = [self.archive()]
        async def command(job, tag, argv, **kwargs):
            job['last_process'] = {'exit_code': 0 if tag == 'latex-version' else 1}
            return 'MiKTeX 25.12 Portable' if tag == 'latex-version' else "! LaTeX Error: File `unknown.cls' not found."
        self.software.command = AsyncMock(side_effect=command)
        first = self.start('repair_latex'); await self.software.tasks[first['id']]
        self.assertEqual(self.software.jobs[first['id']]['status'], 'failed')
        self.assertEqual(self.software.command.await_count, 2)

    async def test_missing_file_mapping_rejects_ambiguity_and_changed_archive(self):
        first = self.archive(); second = self.archive('ambiguous')
        self.software.latex_catalog['repository']['assets'] = [first, second]
        self.assertIsNone(package_for_file(self.software, 'ctexart.cls'))
        self.assertIsNone(package_for_file(self.software, '../ctexart.cls'))
        self.software.latex_catalog['repository']['assets'] = [first]
        self.assertEqual(package_for_file(self.software, 'ctexart.cls'), 'ctex')
        self.software.path('downloads', first['filename']).write_bytes(b'changed')
        with self.assertRaises(DesktopError): package_for_file(self.software, 'ctexart.cls')
        self.assertIsNone(missing_file('generic compilation failure'))


if __name__ == '__main__':
    unittest.main()
