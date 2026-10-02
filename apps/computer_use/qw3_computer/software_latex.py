"""Managed project-local MiKTeX installation, compilation and bounded repair.

All installer/package arguments come from the reviewed catalog. Existing main.tex
is preserved. The compiler cannot auto-install packages or execute shell escapes.
"""
import json
import os
from pathlib import Path
import re
import tarfile
import time

from .desktop import DesktopError
from .software import atomic_json, sha256

FLAGS = ["--disable-installer", "--no-shell-escape", "--dont-parse-first-line",
         "--interaction=nonstopmode", "--halt-on-error"]


def missing_file(log):
    match = re.search(r"File\s+[`']([^`'\r\n]+)'\s+not found", log)
    return match[1] if match else None


def package_for_file(software, name):
    """Resolve a compiler diagnostic only against the verified local assets."""
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        return None
    matches = set()
    for asset in software.latex_catalog["repository"]["assets"]:
        path = software.path("downloads", asset["filename"])
        if not path.is_file() or sha256(path) != asset["sha256"]:
            raise DesktopError("LaTeX repair archive is missing or changed")
        with tarfile.open(path, "r:xz") as archive:
            if any(m.isfile() and m.name.replace("\\", "/").split("/")[-1] == name for m in archive):
                matches.add(asset["package"])
    return next(iter(matches)) if len(matches) == 1 else None


def launchers(name):
    relative = '..\\..\\latex\\' + name + '\\texmfs\\install\\miktex\\bin\\x64'
    setup = '@echo off\ncd /d "%~dp0"\nset "PATH=%~dp0' + relative + ';%SystemRoot%\\System32;%SystemRoot%"\n'
    compile_text = (setup + 'if not exist "build\\manual" mkdir "build\\manual"\n'
        + 'xelatex ' + ' '.join(FLAGS) + ' --output-directory="build\\manual" main.tex\nexit /b %errorlevel%\n')
    return {"open-latex-terminal.cmd": setup + 'cmd /k\n', "compile-latex.cmd": compile_text}


async def latex_workflow(software, job, stage):
    catalog = software.latex_catalog
    catalog_sha = sha256(Path(__file__).with_name("latex_catalog.json"))
    repair = job["operation"] == "repair_latex"
    name = job["project"]
    project = software.path("projects", name)
    marker = software.path("projects", name, ".qw3-latex-project.json")
    install = software.path("latex", name)
    install_receipt = software.path("latex", name + ".json")
    bins = install / "texmfs/install/miktex/bin/x64"
    source = software.path("projects", name, "main.tex")
    owner = {"version": 1, "profile": job["profile"], "catalog_sha256": catalog_sha}
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != owner:
            raise DesktopError("LaTeX project profile changed; inspect before upgrading")
    elif repair or install.exists() or install_receipt.exists():
        raise DesktopError("LaTeX repair requires a matching managed project and toolchain")
    else:
        project.mkdir(exist_ok=True)
        atomic_json(marker, owner)
    for entry in project.rglob("*"):
        software.path(*entry.relative_to(software.root).parts)
    if not source.exists() and not repair:
        software.put(source, Path(__file__).with_name("latex_example.tex").read_text(encoding="utf-8"))
    if not source.is_file() or not 0 < source.stat().st_size <= 1024 * 1024:
        raise DesktopError("LaTeX project needs main.tex (up to 1 MiB)")
    source_sha = sha256(source)
    job["source_sha256"] = source_sha
    job["source_file"] = str(source)
    software.save(job)
    fonts = Path(os.environ.get("SYSTEMROOT", "C:/Windows")) / "Fonts"
    if not all((fonts / f["file"]).is_file() for f in catalog["fonts"]):
        raise DesktopError("This Chinese profile needs installed SimSun and Microsoft YaHei fonts")

    async def downloads():
        assets = [catalog["installer"], *catalog["repository"]["metadata"], *catalog["repository"]["assets"]]
        return {"assets": [await software.download(job, a) for a in assets]}

    async def version():
        executable = software.path("latex", name, "texmfs", "install", "miktex", "bin", "x64", "xelatex.exe")
        if not executable.is_file():
            raise DesktopError("Installed xelatex is absent; installation needs reconciliation")
        output = await software.command(job, "latex-version", [executable, "--version"], cwd=project, timeout=45)
        if "MiKTeX 25.12 Portable" not in output:
            raise DesktopError("LaTeX executable identity differs from the selected profile")
        return {"version": output.splitlines()[0], "executable": str(executable), "sha256": sha256(executable)}

    async def install_base():
        if install_receipt.exists():
            receipt = json.loads(install_receipt.read_text())
            if receipt.get("catalog_sha256") != catalog_sha or receipt.get("status") not in {"base_verified", "ready"}:
                job["needs_reconciliation"] = True
                raise DesktopError("Prior LaTeX installation has not been reconciled; not restarted")
            return (await version()) | {"reused": True}
        if install.exists():
            raise DesktopError("An unmanaged LaTeX installation already occupies this project directory")
        atomic_json(install_receipt, owner | {"status": "installing", "job_id": job["id"]})
        await software.command(job, "install-latex", [software.path("downloads", catalog["installer"]["filename"]),
            "--unattended", "--private", "--portable=" + str(install), "--no-registry", "--no-additional-roots"],
            cwd=project, timeout=600)
        try:
            facts = await version()
        except BaseException:
            job["needs_reconciliation"] = True
            raise
        atomic_json(install_receipt, owner | {"status": "base_verified", "version": facts})
        return facts | {"reused": False}

    async def package_state(tag):
        output = await software.command(job, tag, [bins / "miktex.exe", "packages", "list",
            "--template", "{id}|{isInstalled}\n"], cwd=project)
        states = dict(line.strip().split("|") for line in output.splitlines() if "|" in line)
        names = [a["package"] for a in catalog["repository"]["assets"]]
        if any(states.get(n) not in {"true", "false"} for n in names):
            raise DesktopError("MiKTeX did not report the reviewed package states")
        return {n: states[n] == "true" for n in names}

    async def configure():
        await software.command(job, "latex-database", [bins / "miktex.exe", "--disable-installer", "packages",
            "update-package-database", "--repository", software.path("downloads")], cwd=project)
        states = await package_state("latex-package-state")
        missing = [n for n, installed in states.items() if not installed]
        present = [n for n, installed in states.items() if installed]
        if missing:
            await software.command(job, "latex-install-packages", [bins / "miktex.exe", "--disable-installer", "packages",
                "install", "--repository", software.path("downloads"), *missing], cwd=project)
        if present:
            await software.command(job, "latex-update-packages", [bins / "miktex.exe", "--disable-installer", "packages",
                "update", "--repository", software.path("downloads"), *present], cwd=project)
        await software.command(job, "latex-format", [bins / "miktex.exe", "--disable-installer", "formats", "build", "xelatex"], cwd=project)
        atomic_json(install_receipt, owner | {"status": "ready", "configured_at": time.time()})
        return {"installed": missing, "updated": present}

    async def compile_once(tag):
        folder = software.path("projects", name, "build", job["id"] + '-' + str(job["attempt"]) + '-' + tag)
        folder.mkdir(parents=True, exist_ok=False)  # A stale PDF can never pass.
        output = await software.command(job, tag, [bins / "xelatex.exe", *FLAGS,
            "--output-directory=" + str(folder), "main.tex"], cwd=project, timeout=180, allowed_exit_codes=(0, 1))
        code = job["last_process"]["exit_code"]
        pdf = folder / "main.pdf"
        actual_log = (folder / "main.log").read_text(encoding="utf-8", errors="replace") if (folder / "main.log").is_file() else output
        passed = (code == 0 and pdf.is_file() and pdf.stat().st_size > 500
                  and pdf.read_bytes().startswith(b"%PDF-") and "Missing character:" not in actual_log)
        return {"passed": passed, "exit_code": code, "pdf": str(pdf) if passed else None,
                "pdf_sha256": sha256(pdf) if passed else None, "missing_file": missing_file(output),
                "log": str(folder / "main.log"), "warnings": [l for l in actual_log.splitlines() if "Warning:" in l][:10]}

    if repair:
        receipt = json.loads(install_receipt.read_text()) if install_receipt.is_file() else {}
        if receipt.get("status") != "ready" or any(receipt.get(k) != v for k, v in owner.items()):
            raise DesktopError("LaTeX toolchain is not a verified managed installation")
        await stage("check_latex_version", version)
        # Validate the private installation before a package tool can mutate it.
        for entry in install.rglob("*"):
            software.path(*entry.relative_to(software.root).parts)
        first = await stage("diagnose_latex", lambda: compile_once("diagnose-latex"))
        job["diagnosis"] = first; software.save(job)
        if not first["passed"]:
            await stage("download_latex_assets", downloads)
            for attempt in (1, 2):
                package = package_for_file(software, first["missing_file"])
                if package is None:
                    raise DesktopError("Compile failure is not a uniquely mapped reviewed missing package; inspect the saved log")
                async def repair_package():
                    states = await package_state(f"repair-state-{attempt}")
                    if states[package]:
                        await software.command(job, f"remove-damaged-package-{attempt}", [bins / "miktex.exe", "packages", "remove", package], cwd=project)
                    await software.command(job, f"restore-package-{attempt}", [bins / "miktex.exe", "--disable-installer", "packages",
                        "install", "--repository", software.path("downloads"), package], cwd=project)
                    return {"package": package, "missing_file": first["missing_file"], "previously_installed": states[package]}
                await stage("repair_latex_package", repair_package)
                first = await stage("check_latex_repair", lambda: compile_once(f"repair-check-{attempt}"))
                if first["passed"]:
                    break
            if not first["passed"]:
                raise DesktopError("LaTeX did not recover within two package repairs; take over with the saved logs")
    else:
        await stage("download_latex_assets", downloads)
        await stage("install_latex", install_base)
        await stage("configure_latex_packages", configure)

    async def verify():
        results = [await compile_once(f"verify-latex-{i}") for i in (1, 2)]
        if not all(r["passed"] for r in results) or sha256(source) != source_sha:
            raise DesktopError("LaTeX failed fresh-process compilation or source preservation")
        for filename, content in launchers(name).items():
            software.put(software.path("projects", name, filename), content)
        facts = {"passed": True, "kind": "latex", "project": str(project), "source_sha256": source_sha,
                 "pdf": results[-1]["pdf"], "fresh_compilations": results, "catalog_sha256": catalog_sha,
                 "terminal_launcher": "open-latex-terminal.cmd", "compile_launcher": "compile-latex.cmd",
                 "fonts": [{"file": f["file"], "sha256": sha256(fonts / f["file"])} for f in catalog["fonts"]]}
        atomic_json(software.path("jobs", job["id"], "latex-verification.json"), facts)
        return facts
    return await stage("verify_latex", verify)
