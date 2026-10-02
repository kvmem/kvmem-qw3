"""Independent Windows/host acceptance for the LaTeX software product workflow."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from software_product_guest import digest, registry_paths, save, stop_process_tree


async def grade(project, baseline, output):
    result = {"kind": "independent_latex_environment_acceptance", "passed": False, "checks": {}}
    try:
        expected = baseline.get("input_sha256", baseline.get("fixture", {}).get("source_sha256", {}))
        checks = result["checks"]
        checks["source_unchanged"] = bool(expected) and all(digest(project / n) == v for n, v in expected.items())
        checks["registry_paths_unchanged"] = registry_paths() == baseline["registry_paths"]
        checks["unrelated_project_unchanged"] = all(digest(Path(p)) == value
            for p, value in baseline.get("fixture", {}).get("unrelated_files", {}).items())
        marker = project / ".qw3-latex-project.json"
        checks["project_managed"] = marker.is_file() and json.loads(marker.read_text())["profile"] == "latex-chinese-math"
        workspace = project.parent.parent
        compiler = workspace / 'latex' / project.name / 'texmfs/install/miktex/bin/x64/xelatex.exe'
        fresh = output / 'independent-build'; fresh.mkdir(exist_ok=False)
        process = None
        try:
            process = await asyncio.create_subprocess_exec(str(compiler), '--disable-installer', '--no-shell-escape',
                '--dont-parse-first-line', '--interaction=nonstopmode', '--halt-on-error',
                '--output-directory=' + str(fresh), 'main.tex', cwd=project,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await asyncio.wait_for(process.communicate(), 180)
            result['independent_compile'] = {'exit_code': process.returncode,
                'stdout': stdout.decode('utf-8', errors='replace'), 'stderr': stderr.decode('utf-8', errors='replace')}
            checks['fresh_independent_compile'] = process.returncode == 0 and (fresh / 'main.pdf').is_file()
            if (fresh / 'main.pdf').is_file():
                shutil.copyfile(fresh / 'main.pdf', output / 'independent.pdf')
        finally:
            await stop_process_tree(process)
        jobs = [json.loads(p.read_text(encoding="utf-8")) for p in (workspace / 'jobs').glob('*/job.json')]
        owned = [j for j in jobs if j['project'] == project.name and j.get('verification', {}).get('kind') == 'latex' and j['status'] == 'completed']
        latest = max(owned, key=lambda j: j['finished_at']) if owned else None
        checks["completed_job_present"] = latest is not None
        if latest:
            verified = latest['verification']
            checks['two_distinct_compilations'] = len({p['pdf'] for p in verified['fresh_compilations'] if p['passed']}) == 2
            checks['pdf_files_match_receipts'] = all(digest(Path(p['pdf'])) == p['pdf_sha256'] for p in verified['fresh_compilations'])
            checks['source_matches_job'] = digest(project / 'main.tex') == verified['source_sha256']
            # Copy the actual delivered result for a host parser/rendering pass.
            shutil.copyfile(verified['pdf'], output / 'delivered.pdf')
            result['verification'] = verified
        result['passed'] = all(checks.values())
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    save(output / 'independent-grade.json', result)
    return result


async def grade_launcher(project, output):
    """Run both shipped launchers in new cmd processes, without inherited PATH."""
    result = {'kind': 'delivered_latex_launcher_acceptance', 'passed': False, 'checks': {}}
    process = None
    try:
        env = {k: v for k, v in os.environ.items() if k.upper() not in {'PATH', 'MIKTEX_BIN_DIR', 'TEXINPUTS', 'VIRTUAL_ENV'}}
        env['PATH'] = str(Path(os.environ['SYSTEMROOT']) / 'System32')
        cmd = Path(os.environ['SYSTEMROOT']) / 'System32/cmd.exe'
        process = await asyncio.create_subprocess_exec(str(cmd), '/d', '/q', '/c', 'open-latex-terminal.cmd',
            cwd=project, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(process.communicate(b'where xelatex\r\nxelatex --version\r\nexit\r\n'), 60)
        text = stdout.decode('utf-8', errors='replace')
        expected = project.parent.parent / 'latex' / project.name / 'texmfs/install/miktex/bin/x64/xelatex.exe'
        result.update(terminal_exit_code=process.returncode,terminal_stdout=text,terminal_stderr=stderr.decode('utf-8', errors='replace'))
        result['checks']['selected_project_compiler'] = str(expected).lower() in text.lower() and 'MiKTeX 25.12 Portable' in text
        await stop_process_tree(process)
        started = time.time()
        process = await asyncio.create_subprocess_exec(str(cmd), '/d', '/q', '/c', 'compile-latex.cmd',
            cwd=project, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(process.communicate(), 180)
        result.update(compile_exit_code=process.returncode,compile_stdout=stdout.decode('utf-8', errors='replace'),
                      compile_stderr=stderr.decode('utf-8', errors='replace'))
        pdf = project / 'build/manual/main.pdf'
        result['checks'].update(compile_succeeded=process.returncode == 0,
            fresh_pdf=pdf.is_file() and pdf.stat().st_mtime >= started and pdf.read_bytes().startswith(b'%PDF-'))
        if pdf.is_file():
            result['pdf_sha256'] = digest(pdf);shutil.copyfile(pdf, output / 'launcher.pdf')
        result['passed'] = all(result['checks'].values())
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        await stop_process_tree(process)
        save(output / 'launcher-grade.json', result)
    return result


def host_pdf_grade(root):
    """Use a separate PDF implementation to inspect actual exported artifacts."""
    report = {'kind': 'host_independent_latex_pdf_acceptance', 'passed': False, 'artifacts': []}
    try:
        for relative in ('delivered.pdf', 'independent.pdf', 'launcher.pdf',
                         'reopened/delivered.pdf', 'reopened/independent.pdf', 'reopened/launcher.pdf'):
            pdf = root / 'guest' / relative
            info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
            text = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-'], text=True)
            fonts = subprocess.check_output(['pdffonts', str(pdf)], text=True)
            checks = {'one_a4_page': 'Pages:           1' in info and '(A4)' in info,
                      'chinese_text': all(s in text for s in ['本地工具链验收', '中文环境配置完成', '数学公式']),
                      'document_marker': 'QW3-LATEX-20261003' in text,
                      'math_content': all(s in text for s in ['55', 'mc', '∫', '∑']),
                      'all_fonts_embedded': len(fonts.splitlines()) > 2 and all(line.split()[-5] == 'yes'
                            for line in fonts.splitlines()[2:] if line.strip())}
            item = {'file': relative, 'sha256': digest(pdf), 'checks': checks, 'passed': all(checks.values())}
            stem = root / ('pdf-' + relative.replace('/', '-').removesuffix('.pdf'))
            stem.with_suffix('.txt').write_text(text, encoding='utf-8')
            subprocess.run(['pdftoppm', '-f', '1', '-singlefile', '-scale-to', '1200', '-png', str(pdf), str(stem)], check=True)
            report['artifacts'].append(item)
        report['passed'] = all(item['passed'] for item in report['artifacts'])
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    save(root / 'pdf-grade.json', report)
    return report
