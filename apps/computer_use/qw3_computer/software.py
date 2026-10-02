"""Opt-in, native Windows software workflows with durable, bounded jobs.

The model selects a reviewed profile and a project name, never a command or URL.
Installation, project isolation and verification are separate recorded stages.
"""
import asyncio
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import subprocess
import sys
import time
from urllib.parse import urlparse

import aiohttp
import certifi

from .desktop import DesktopError


ACTIVE = {"starting", "running"}
PROJECT = re.compile(r"[a-z0-9][a-z0-9-]{0,47}")
ACTION_ID = re.compile(r"[a-zA-Z0-9_-]{8,80}")
RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(10)), *(f"lpt{i}" for i in range(10))}


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def process_environment():
    # Windows known-folder expansion needs SystemDrive even for a per-user
    # installer. Do not inherit arbitrary credentials, proxy or Python settings.
    env = {k: v for k, v in os.environ.items() if k.upper() in {
        "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP", "USERPROFILE",
        "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "ALLUSERSPROFILE"}}
    env.update(PYTHONUTF8="1", PYTHONNOUSERSITE="1", PIP_CONFIG_FILE=os.devnull)
    return env


def job_receipt(job):
    """Expose the identifier accepted by job_status, not its private owner ID."""
    receipt = {k: v for k, v in job.items() if k not in {"task_id", "action_id"}}
    receipt["job_id"] = job["id"]
    if job["status"] in ACTIVE:
        receipt["next_poll"] = {"operation": "job_status", "job_id": job["id"], "wait_seconds": 10}
    return receipt


class Software:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog = json.loads(Path(__file__).with_name("software_catalog.json").read_text())
        for name in ("projects", "downloads", "jobs", "python"):
            self.path(name).mkdir(exist_ok=True)
        # The profile is a local authorization boundary, not a model argument.
        self.lock = self.path("workspace.lock").open("a+b")
        self.lock.write(b"0"); self.lock.flush(); self.lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise DesktopError("Software workspace is already controlled by another application")
        self.tasks = {}
        self.jobs = {}
        for path in self.path("jobs").glob("*/job.json"):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["status"] in ACTIVE:
                job.update(status="uncertain", needs_reconciliation=True,
                           error="Application restarted during a software job; inspect process and installation state before recovery")
                atomic_json(path, job)
            self.jobs[job["id"]] = job

    def path(self, *parts):
        candidate = self.root.joinpath(*parts)
        if not candidate.is_relative_to(self.root):
            raise DesktopError("Path is outside the selected software workspace")
        current = self.root
        for part in candidate.relative_to(self.root).parts:
            current /= part
            if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
                raise DesktopError("Software workspace paths must not contain links or junctions")
        if candidate.resolve() != candidate:
            raise DesktopError("Software workspace path changed")
        return candidate

    def save(self, job):
        atomic_json(self.path("jobs", job["id"], "job.json"), job)

    def snapshot(self, task_id=None):
        return {"root": str(self.root), "profiles": {k: {"description": v["description"], "requirements": v["requirements"]}
                for k, v in self.catalog["profiles"].items()}, "python_version": self.catalog["python"]["version"],
                "jobs": [job_receipt(j) for j in self.jobs.values() if task_id is None or j["task_id"] == task_id]}

    def inspect(self, task_id):
        projects = []
        for path in self.path("projects").iterdir():
            if not PROJECT.fullmatch(path.name) or path.name in RESERVED:
                continue
            path = self.path("projects", path.name)
            if path.is_dir():
                projects.append({"name": path.name, "path": str(path),
                                 "managed": (path / ".qw3-project.json").is_file(),
                                 "venv_present": (path / ".venv").exists()})
        return self.snapshot(task_id) | {"projects": projects, "registered_python": self.registered_python()}

    def registered_python(self):
        if sys.platform != "win32":
            return []
        import winreg
        found = []
        for hive_name, hive in (("current_user", winreg.HKEY_CURRENT_USER), ("machine", winreg.HKEY_LOCAL_MACHINE)):
            try:
                with winreg.OpenKey(hive, r"Software\Python\PythonCore\3.13\InstallPath") as key:
                    found.append({"scope": hive_name, "path": winreg.QueryValue(key, "")})
            except FileNotFoundError:
                pass
        return found

    async def invoke(self, task_id, arguments):
        if not isinstance(arguments, dict):
            raise DesktopError("Software tool needs an object")
        operation = arguments.get("operation")
        fields = {"inspect": {"operation"}, "setup_python": {"operation", "action_id", "project", "profile"},
                  "repair_python": {"operation", "action_id", "project", "profile"},
                  "job_status": {"operation", "job_id", "wait_seconds"}, "cancel_job": {"operation", "job_id"},
                  "resume_job": {"operation", "job_id"}}
        if operation not in fields or set(arguments) - fields[operation]:
            raise DesktopError("Unsupported software operation or argument")
        if operation == "inspect":
            return self.inspect(task_id)
        if operation in {"setup_python", "repair_python"}:
            return self.start(task_id, arguments)
        job_id = arguments.get("job_id")
        if not isinstance(job_id, str) or job_id not in self.jobs or self.jobs[job_id]["task_id"] != task_id:
            raise DesktopError("Software job does not belong to this task")
        job = self.jobs[job_id]
        if operation == "job_status":
            seconds = arguments.get("wait_seconds", 5)
            if type(seconds) is not int or not 0 <= seconds <= 10:
                raise DesktopError("Job wait must be 0..10 seconds")
            if task := self.tasks.get(job_id):
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(task), seconds)
        elif operation == "cancel_job":
            await self.cancel(job_id)
        elif operation == "resume_job":
            if job["status"] not in {"failed", "cancelled"} or job.get("needs_reconciliation"):
                raise DesktopError("Only a known stopped software job can resume; uncertain installation requires reconciliation")
            if any(j["status"] in ACTIVE or j["status"] == "uncertain" for j in self.jobs.values()):
                raise DesktopError("Another software job is active or unresolved")
            job.update(status="starting", attempt=job["attempt"] + 1)
            job.pop("error", None)
            self.save(job)
            self.tasks[job_id] = asyncio.create_task(self.perform(job))
        return job_receipt(job)

    def start(self, task_id, arguments):
        project, profile, action_id = (arguments.get(k) for k in ("project", "profile", "action_id"))
        if (not isinstance(project, str) or not PROJECT.fullmatch(project) or project in RESERVED
                or not isinstance(profile, str) or profile not in self.catalog["profiles"]
                or not isinstance(action_id, str) or not ACTION_ID.fullmatch(action_id)):
            raise DesktopError("Invalid project, reviewed profile, or action identity")
        if not isinstance(task_id, str) or not task_id:
            raise DesktopError("Software job needs an owning task")
        # The model's idempotency key is scoped to this task. Older clients used
        # that key directly as a global directory name. Preserve their receipts
        # and same-task retries without allowing another task to adopt them.
        identity = next((j["id"] for j in self.jobs.values()
                         if j["task_id"] == task_id and j.get("action_id", j["id"]) == action_id), None)
        if identity is None:
            encoded = json.dumps([task_id, action_id], ensure_ascii=True).encode("utf-8")
            identity = "job-" + hashlib.sha256(encoded).hexdigest()[:24]
        request = {"task_id": task_id, "project": project, "profile": profile,
                   "operation": arguments.get("operation", "setup_python"), "action_id": action_id}
        if request["operation"] not in {"setup_python", "repair_python"}:
            raise DesktopError("Unsupported software operation")
        if identity in self.jobs:
            job = self.jobs[identity]
            previous = job | {"operation": job.get("operation", "setup_python"), "action_id": job.get("action_id", job["id"])}
            if any(previous[k] != v for k, v in request.items()):
                raise DesktopError("Software action identity was reused with different arguments")
            return job_receipt(job) | {"duplicate": True}
        if sys.platform != "win32":
            raise DesktopError("Software installation currently requires the native Windows client")
        if any(j["status"] in ACTIVE or j["status"] == "uncertain" for j in self.jobs.values()):
            raise DesktopError("Another software job is active or unresolved")
        if shutil.disk_usage(self.root).free < 1024**3:
            raise DesktopError("Software workspace requires at least 1 GiB free before installation")
        self.path("jobs", identity).mkdir(exist_ok=False)
        job = request | {"id": identity, "status": "starting", "attempt": 1, "steps": [],
                         "created_at": time.time(), "needs_reconciliation": False}
        self.jobs[identity] = job
        self.save(job)  # Reserve identity durably before any installation work.
        self.tasks[identity] = asyncio.create_task(self.perform(job))
        return job_receipt(job)

    async def cancel(self, identity):
        task = self.tasks.get(identity)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            job = self.jobs[identity]
            if task.cancelled() and job["status"] == "starting":
                # Cancellation before the coroutine's first instruction does
                # not execute perform's exception/finally handlers.
                job.update(status="cancelled", finished_at=time.time(),
                           error="Software job cancelled before its worker started")
                self.save(job)

    async def pause(self, task_id):
        for identity, job in self.jobs.items():
            if job["task_id"] == task_id:
                await self.cancel(identity)

    async def close(self):
        for identity in list(self.tasks):
            await self.cancel(identity)
        self.lock.close()

    async def download(self, job, asset):
        target = self.path("downloads", asset["filename"])
        if target.exists():
            if sha256(target) != asset["sha256"]:
                raise DesktopError("Cached download hash mismatch; retained for inspection")
            return {"file": target.name, "cache_hit": True, "bytes": target.stat().st_size, "sha256": asset["sha256"]}
        url = urlparse(asset["url"])
        if url.scheme != "https" or url.hostname not in {"www.python.org", "files.pythonhosted.org"}:
            raise DesktopError("Unreviewed software download source")
        temporary = target.with_suffix(target.suffix + ".partial-" + job["id"])
        # A cancelled GET can restart safely; an installer is never retried here.
        maximum = asset.get("max_bytes", asset.get("bytes", 0))
        digest = hashlib.sha256(); count = 0; started = time.monotonic()
        try:
            # The bundled Python cannot rely on a freshly installed Windows
            # root store. Ship a versioned public CA bundle; retain TLS hostname
            # and certificate checks, then separately verify the asset hash.
            tls = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=tls)
            async with aiohttp.ClientSession(connector=connector, trust_env=False,
                    timeout=aiohttp.ClientTimeout(total=300, sock_read=30)) as client:
                async with client.get(asset["url"], allow_redirects=False) as response:
                    if response.status != 200:
                        raise DesktopError(f"Software download HTTP {response.status}")
                    with temporary.open("wb") as handle:
                        async for data in response.content.iter_chunked(262144):
                            count += len(data)
                            if count > maximum:
                                raise DesktopError("Software download exceeded its reviewed size")
                            handle.write(data); digest.update(data)
            if digest.hexdigest() != asset["sha256"]:
                raise DesktopError("Downloaded software hash differs from the reviewed catalog")
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return {"file": target.name, "cache_hit": False, "bytes": count, "sha256": digest.hexdigest(),
                "download_seconds": time.monotonic() - started}

    async def command(self, job, tag, argv, *, cwd, timeout=300):
        for path in (Path(argv[0]), Path(cwd)):
            self.path(*path.relative_to(self.root).parts)
        log = self.path("jobs", job["id"], f"{job['attempt']}-{tag}.log")
        # No shell; arguments are produced only by this reviewed workflow.
        with log.open("wb") as output:
            process = await asyncio.create_subprocess_exec(*map(str, argv), cwd=cwd, env=process_environment(),
                stdin=subprocess.DEVNULL, stdout=output, stderr=output)
            job["process"] = {"pid": process.pid, "tag": tag, "started_at": time.time()}
            self.save(job)
            try:
                async with asyncio.timeout(timeout):
                    while process.returncode is None:
                        if log.stat().st_size > 4 * 1024 * 1024:
                            raise DesktopError("Software process output exceeded its limit")
                        await asyncio.sleep(.2)
                    code = await process.wait()
                if code:
                    raise DesktopError(f"{tag} exited with {code}; inspect {log.name}")
            except BaseException:
                stopped = process.returncode is not None
                if not stopped:
                    try:
                        killer = await asyncio.create_subprocess_exec(
                            str(Path(os.environ["SYSTEMROOT"]) / "System32/taskkill.exe"),
                            "/PID", str(process.pid), "/T", "/F", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        await asyncio.wait_for(killer.wait(), 10)
                        await asyncio.wait_for(process.wait(), 10)
                        stopped = True
                    except (OSError, TimeoutError, KeyError):
                        pass
                # An installer may delegate to Windows Installer. Killing its
                # launcher does not establish rollback or completed delivery.
                if tag == "install-python" or not stopped:
                    job["needs_reconciliation"] = True
                raise
            finally:
                job["last_process"] = job.pop("process") | {"exit_code": process.returncode, "finished_at": time.time()}
                self.save(job)
        return log.read_text(encoding="utf-8", errors="replace")

    def put(self, path, text):
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise DesktopError(f"Existing project file differs; not overwritten: {path.name}")
        else:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)

    async def perform(self, job):
        job.update(status="running", started_at=time.time())
        self.save(job)

        async def stage(name, function):
            record = {"name": name, "attempt": job["attempt"], "status": "running", "started_at": time.time()}
            job["steps"].append(record); job["stage"] = name; self.save(job)
            result = await function()
            record.update(status="completed", finished_at=time.time(), result=result); self.save(job)
            return result

        try:
            if job.get("operation") == "repair_python":
                from .software_repair import repair_python
                job["verification"] = await repair_python(self, job, stage)
                job.update(status="completed", finished_at=time.time())
                return
            profile = self.catalog["profiles"][job["profile"]]
            project = self.path("projects", job["project"]); project.mkdir(exist_ok=True)
            marker = self.path("projects", job["project"], ".qw3-project.json")
            owner = {"version": 1, "profile": job["profile"], "python": self.catalog["python"]["version"]}
            if marker.exists():
                if json.loads(marker.read_text()) != owner:
                    raise DesktopError("Project belongs to a different workflow profile")
            else:
                if (project / ".venv").exists():
                    raise DesktopError("Existing unmanaged virtual environment requires inspection before repair")
                atomic_json(marker, owner)
            install = self.path("python", self.catalog["python"]["version"])
            base = install / "python.exe"
            for registered in self.registered_python():
                if Path(registered["path"]).resolve() != install:
                    raise DesktopError("An existing Python 3.13 registration would be changed; installation was not started")
            await stage("download_python", lambda: self.download(job, self.catalog["python"]))

            async def install_python():
                reused = base.exists()
                if not reused:
                    await self.command(job, "install-python", [self.path("downloads", self.catalog["python"]["filename"]),
                        "/quiet", "/norestart", "InstallAllUsers=0", f"TargetDir={install}", "Include_launcher=0",
                        "PrependPath=0", "AppendPath=0", "AssociateFiles=0", "Shortcuts=0", "Include_test=0",
                        "Include_doc=0", "Include_tcltk=0", "Include_dev=0", "Include_pip=1", "CompileAll=0",
                        "/log", self.path("jobs", job["id"], f"{job['attempt']}-installer.log")], cwd=project, timeout=600)
                facts = json.loads(await self.command(job, "verify-base", [base, "-I", "-c",
                    "import sys,json,struct;print(json.dumps(dict(version=sys.version.split()[0],bits=struct.calcsize('P')*8,executable=sys.executable)))"], cwd=project))
                if facts["version"] != self.catalog["python"]["version"] or facts["bits"] != 64:
                    raise DesktopError("Installed interpreter identity differs from the requested profile")
                return facts | {"reused": reused}
            await stage("install_python", install_python)
            venv = self.path("projects", job["project"], ".venv")
            python = venv / "Scripts/python.exe"

            async def create_venv():
                await self.command(job, "create-venv", [base, "-I", "-m", "venv", venv], cwd=project)
                return {"path": str(venv)}
            await stage("create_venv", create_venv)

            async def dependencies():
                assets = [await self.download(job, asset) for asset in profile["packages"]]
                lock = self.path("projects", job["project"], "requirements.lock")
                self.put(lock, "".join(f"{a['name']}=={a['version']} --hash=sha256:{a['sha256']}\n" for a in profile["packages"]))
                await self.command(job, "install-dependencies", [python, "-I", "-m", "pip", "--isolated",
                    "--disable-pip-version-check", "install", "--no-input", "--no-index", "--only-binary=:all:",
                    "--find-links", self.path("downloads"), "--require-hashes", "-r", lock], cwd=project)
                return {"downloads": assets, "lock_sha256": sha256(lock)}
            await stage("install_dependencies", dependencies)

            async def verify():
                await self.command(job, "pip-check", [python, "-I", "-m", "pip", "--isolated", "check"], cwd=project)
                expected = {a["name"]: a["version"] for a in profile["packages"]}
                script = ("import sys,json,importlib.metadata as m,requests,pytest;"
                          "print(json.dumps(dict(version=sys.version.split()[0],executable=sys.executable,"
                          "prefix=sys.prefix,base_prefix=sys.base_prefix,packages={n:m.version(n) for n in "
                          + repr(list(expected)) + "},request_url=requests.Request('GET','https://example.com').prepare().url)))")
                results = []
                for index in (1, 2):
                    # Two independent processes: no inherited interpreter or
                    # activated shell state can hide an unusable environment.
                    facts = json.loads(await self.command(job, f"verify-project-{index}", [python, "-I", "-c", script], cwd=project))
                    if (Path(facts["executable"]).resolve() != python.resolve() or Path(facts["prefix"]).resolve() != venv
                            or facts["base_prefix"] == facts["prefix"] or facts["packages"] != expected
                            or facts["version"] != self.catalog["python"]["version"] or facts["request_url"] != "https://example.com/"):
                        raise DesktopError("Fresh-process project verification failed")
                    results.append(facts)
                self.put(self.path("projects", job["project"], "open-terminal.cmd"),
                    '@echo off\ncd /d "%~dp0"\ncall "%~dp0.venv\\Scripts\\activate.bat"\ncmd /k\n')
                self.put(self.path("projects", job["project"], "ENVIRONMENT.txt"),
                    "Open open-terminal.cmd for this project's Python and pip.\n"
                    f"Python {self.catalog['python']['version']}\n" + "\n".join(profile["requirements"]) + "\n")
                proof = {"passed": True, "project": str(project), "fresh_processes": results,
                         "catalog_sha256": sha256(Path(__file__).with_name("software_catalog.json")), "verified_at": time.time()}
                atomic_json(self.path("projects", job["project"], "environment-verification.json"), proof)
                return proof
            job["verification"] = await stage("verify_project", verify)
            job.update(status="completed", finished_at=time.time())
        except asyncio.CancelledError:
            job.update(status="uncertain" if job["needs_reconciliation"] else "cancelled",
                       error="Software job interrupted; inspect saved checkpoints before resuming", finished_at=time.time())
        except Exception as exc:
            job.update(status="uncertain" if job["needs_reconciliation"] else "failed",
                       error=f"{type(exc).__name__}: {exc}", finished_at=time.time())
        finally:
            if job.get("steps") and job["steps"][-1]["status"] == "running":
                job["steps"][-1].update(status=job["status"], finished_at=time.time())
            self.save(job)
