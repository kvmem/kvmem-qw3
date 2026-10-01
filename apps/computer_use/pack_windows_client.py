"""Assemble an offline Windows x64 client from verified downloaded assets.

The bundle contains the app, CPython and pinned native Codex, not model weights
or an executor credential. It must still be qualified inside Windows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unzip(path, output):
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            parts = Path(member.filename).parts
            if member.filename.startswith("/") or ".." in parts or ":" in member.filename or "\\" in member.filename:
                raise ValueError("Unsafe zip member: " + member.filename)
        archive.extractall(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    assets = args.assets.resolve()
    metadata = json.loads((assets / "codex-release.json").read_text())
    assert metadata["tag_name"] == "rust-v0.159.2"
    codex = assets / "codex-package-x86_64-pc-windows-msvc.tar.gz"
    expected = next(a["digest"] for a in metadata["assets"] if a["name"] == codex.name)
    assert "sha256:" + digest(codex) == expected, "Codex package differs from official release digest"
    with tarfile.open(codex) as archive:
        archive.extractall(output / "codex", filter="data")
    python = assets / "python-3.13.13-embed-amd64.zip"
    unzip(python, output / "python")
    wheels = sorted((assets / "wheels").glob("*.whl"))
    assert any(p.name.startswith("aiohttp-3.14.0-") for p in wheels)
    assert any(p.name.startswith("pillow-12.1.0-") for p in wheels)
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            assert not any(".data/" in n for n in archive.namelist()), "Unsupported wheel installation scheme"
        unzip(wheel, output / "python/Lib/site-packages")
    # Embeddable CPython intentionally ignores PYTHONPATH. Include the app and
    # native wheels in its explicit isolated path configuration instead.
    (output / "python/python313._pth").write_text(
        "python313.zip\n.\nLib/site-packages\n../app\nimport site\n", encoding="utf-8")
    app = output / "app"
    shutil.copytree(root / "qw3_computer", app / "qw3_computer", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(root / "tests", app / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    (app / "vm").mkdir()
    shutil.copyfile(root / "vm/model_relay.py", app / "vm/model_relay.py")
    shutil.copyfile(root / "windows_task.py", app / "windows_task.py")
    shutil.copyfile(root / "windows_long_task.py", app / "windows_long_task.py")
    shutil.copyfile(root / "probe_compaction.py", app / "probe_compaction.py")
    shutil.copyfile(root / "task_driver.py", app / "task_driver.py")
    shutil.copyfile(root / "collect_artifacts.py", app / "collect_artifacts.py")
    (output / "start.ps1").write_text('''param(
    [string]$Backend = 'http://127.0.0.1:18080/v1',
    [string]$BackendToken = '',
    [string]$Desktop = 'http://127.0.0.1:5000',
    [string]$DesktopToken = 'C:\\Qw3Computer\\executor-token.txt',
    [string]$Runtime = '',
    [ValidateRange(1,10000)][int]$ToolCallLimit = 100,
    [ValidateSet('outputs','task')][string]$OutputPolicy = 'outputs'
)
$ErrorActionPreference = 'Stop'
$env:PATH = "$PSScriptRoot\\codex\\bin;$env:PATH"
$env:PYTHONUTF8 = '1'
if (!$Runtime) { $Runtime = Join-Path $PSScriptRoot 'runtime' }
$clientArgs = @('-X','utf8','-m','qw3_computer.app','--backend',$Backend,'--desktop',$Desktop,'--desktop-token',$DesktopToken,'--runtime',$Runtime)
$clientArgs += @('--tool-call-limit',"$ToolCallLimit",'--output-policy',$OutputPolicy)
if ($BackendToken) { $clientArgs += @('--backend-token',$BackendToken) }
& "$PSScriptRoot\\python\\python.exe" @clientArgs
exit $LASTEXITCODE
''', encoding="utf-8")
    (output / "test.ps1").write_text('''$ErrorActionPreference = 'Stop'
$env:PATH = "$PSScriptRoot\\codex\\bin;$env:PATH"
$env:PYTHONUTF8 = '1'
$env:QW3_TEST_CODEX = '1'
& "$PSScriptRoot\\python\\python.exe" -X utf8 -m unittest discover -s "$PSScriptRoot\\app\\tests" -v
exit $LASTEXITCODE
''', encoding="utf-8")
    manifest = {
        "kind": "windows_native_client_offline_bundle", "qualified": False,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "python_version": "3.13.13", "codex_version": "0.159.2",
        "assets": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in [python, codex, *wheels]},
        "source_files": {str(p.relative_to(app)): digest(p) for p in app.rglob("*") if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Offline client assembled:", output)
    print("Windows execution and inference connectivity are not yet qualified.")


if __name__ == "__main__":
    main()
