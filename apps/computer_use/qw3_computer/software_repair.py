"""Repair a selected managed Python project without rerunning its installer.

Only the reviewed dependency profile and delivered terminal launcher are changed.
A diagnostic receipt precedes every mutation; an unknown interpreter requires
handover instead of recreating an environment over existing files.
"""
import json
from pathlib import Path

from .desktop import DesktopError
from .software import atomic_json, sha256


LAUNCHER = '@echo off\ncd /d "%~dp0"\ncall "%~dp0.venv\\Scripts\\activate.bat"\ncmd /k\n'


async def repair_python(software, job, stage):
    project = software.path("projects", job["project"])
    marker = software.path("projects", job["project"], ".qw3-project.json")
    owner = {"version": 1, "profile": job["profile"], "python": software.catalog["python"]["version"]}
    if not marker.is_file() or json.loads(marker.read_text()) != owner:
        raise DesktopError("Repair requires a matching managed project; inspect or take over an unmanaged environment")
    venv = software.path("projects", job["project"], ".venv")
    python = software.path("projects", job["project"], ".venv", "Scripts", "python.exe")
    base = software.path("python", software.catalog["python"]["version"])
    if not python.is_file():
        raise DesktopError("Project interpreter is missing; rebuilding an existing environment requires takeover")
    profile = software.catalog["profiles"][job["profile"]]
    expected = {a["name"]: a["version"] for a in profile["packages"]}
    lock = software.path("projects", job["project"], "requirements.lock")
    lock_text = "".join(f"{a['name']}=={a['version']} --hash=sha256:{a['sha256']}\n" for a in profile["packages"])
    if not lock.is_file() or lock.read_text(encoding="utf-8") != lock_text:
        raise DesktopError("Project dependency lock differs from the reviewed profile; no repair was started")
    launcher = software.path("projects", job["project"], "open-terminal.cmd")
    if launcher.exists() and (not launcher.is_file() or launcher.stat().st_size > 65536):
        raise DesktopError("Unexpected terminal launcher; no repair was started")
    # Validate descendants before pip can read/write package files. A junction
    # inside site-packages must not redirect a bounded repair to another project.
    for entry in venv.rglob("*"):
        software.path(*entry.relative_to(software.root).parts)

    script = '''import sys,json,importlib,importlib.metadata as metadata
facts=dict(version=sys.version.split()[0],executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,packages={},imports={})
for name in NAMES:
    try: facts['packages'][name]=metadata.version(name)
    except metadata.PackageNotFoundError: facts['packages'][name]=None
for name in ('requests','pytest'):
    try:
        module=importlib.import_module(name)
        facts['imports'][name]=dict(ok=True,file=module.__file__)
    except Exception as exc: facts['imports'][name]=dict(ok=False,error=type(exc).__name__+': '+str(exc)[:300])
print(json.dumps(facts))
'''.replace("NAMES", repr(list(expected)))

    async def diagnose(tag):
        facts = json.loads(await software.command(job, tag, [python, "-I", "-B", "-c", script], cwd=project, timeout=30))
        identity_ok = (Path(facts["executable"]).resolve() == python and Path(facts["prefix"]).resolve() == venv
                       and Path(facts["base_prefix"]).resolve() == base and facts["version"] == owner["python"])
        imports_ok = all(v["ok"] and Path(v["file"]).resolve().is_relative_to(venv)
                         for v in facts["imports"].values())
        return facts | {"identity_ok": identity_ok, "dependencies_ok": facts["packages"] == expected and imports_ok,
                        "launcher_ok": launcher.is_file() and launcher.read_text(encoding="utf-8") == LAUNCHER}

    before = await stage("diagnose_project", lambda: diagnose("diagnose-project"))
    job["diagnosis"] = before
    software.save(job)
    if not before["identity_ok"]:
        raise DesktopError("Project interpreter identity is inconsistent; no repair was started")

    async def dependencies():
        assets = [await software.download(job, asset) for asset in profile["packages"]]
        await software.command(job, "repair-dependencies", [python, "-I", "-m", "pip", "--isolated",
            "--disable-pip-version-check", "install", "--no-input", "--no-index", "--only-binary=:all:",
            "--find-links", software.path("downloads"), "--require-hashes", "--force-reinstall", "--no-deps", "-r", lock], cwd=project)
        return {"downloads": assets, "lock_sha256": sha256(lock), "forced_reinstall": True}

    async def terminal():
        backup = software.path("jobs", job["id"], f"{job['attempt']}-open-terminal.cmd.before")
        existing = launcher.read_bytes() if launcher.exists() else None
        if existing is not None:
            with backup.open("xb") as output:
                output.write(existing)
        receipt = {"previously_present": existing is not None,
                   "backup": str(backup) if existing is not None else None,
                   "backup_sha256": sha256(backup) if existing is not None else None}
        # Record rollback evidence before replacing the single managed launcher.
        job["launcher_backup"] = receipt
        software.save(job)
        with launcher.open("w", encoding="utf-8", newline="\n") as output:
            output.write(LAUNCHER)
        return receipt | {"new_sha256": sha256(launcher)}

    if not before["dependencies_ok"]:
        await stage("repair_dependencies", dependencies)
    if not before["launcher_ok"]:
        await stage("repair_terminal_launcher", terminal)

    async def verify():
        await software.command(job, "repair-pip-check", [python, "-I", "-m", "pip", "--isolated", "check"], cwd=project)
        processes = [await diagnose(f"verify-repair-{index}") for index in (1, 2)]
        if not all(p["identity_ok"] and p["dependencies_ok"] and p["launcher_ok"] for p in processes):
            raise DesktopError("Repair failed fresh-process verification")
        result = {"passed": True, "project": str(project), "fresh_processes": processes,
                  "scope": "managed_project_dependencies_and_terminal_launcher", "system_installer_run": False}
        atomic_json(software.path("jobs", job["id"], "repair-verification.json"), result)
        return result
    return await stage("verify_repair", verify)
