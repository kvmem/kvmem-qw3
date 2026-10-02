"""Owned-VM product trial. Fixtures precede the agent; independent grading follows it."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import aiohttp

from task_driver import run_task


MAIN = '''import json, sys, requests, pytest
def total():
    return sum([12, 30])
if __name__ == '__main__':
    print(json.dumps(dict(total=total(), version=sys.version.split()[0], executable=sys.executable,
                         prefix=sys.prefix, requests=requests.__version__, pytest=pytest.__version__,
                         url=requests.Request('GET', 'https://example.com').prepare().url)))
'''
TEST = "from main import total\ndef test_total():\n    assert total() == 42\n"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def registry_paths():
    import winreg
    result = {}
    for scope, hive, path in (("user", winreg.HKEY_CURRENT_USER, "Environment"),
                             ("machine", winreg.HKEY_LOCAL_MACHINE,
                              r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")):
        with winreg.OpenKey(hive, path) as key:
            try:
                value, kind = winreg.QueryValueEx(key, "Path")
                result[scope] = {"value": value, "kind": kind}
            except FileNotFoundError:
                result[scope] = {"missing": True}
    return result


async def stop_process_tree(process):
    if process and process.returncode is None:
        killer = await asyncio.create_subprocess_exec(
            str(Path(os.environ["SYSTEMROOT"]) / "System32/taskkill.exe"),
            "/PID", str(process.pid), "/T", "/F", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        await asyncio.wait_for(killer.wait(), 15)
        await asyncio.wait_for(process.wait(), 15)


async def grade_launcher(project, output):
    """Exercise the actual delivered launcher, not a replacement activation script."""
    result = {"kind": "delivered_terminal_launcher_acceptance", "passed": False}
    process = None
    try:
        if not (project / "open-terminal.cmd").is_file():
            raise ValueError("The delivered launcher does not exist")
        # Deliberately start without an inherited activated virtual environment.
        env = {k: v for k, v in os.environ.items() if k.upper() not in {"VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"}}
        process = await asyncio.create_subprocess_exec(
            str(Path(os.environ["SYSTEMROOT"]) / "System32/cmd.exe"), "/d", "/q", "/c", "open-terminal.cmd",
            cwd=project, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        code = "import json,sys,requests,pytest;print('QW3_LAUNCHER_FACTS '+json.dumps(dict(executable=sys.executable,prefix=sys.prefix,version=sys.version.split()[0],requests=requests.__version__,pytest=pytest.__version__)))"
        stdin = ('python -c "' + code + '"\r\nexit\r\n').encode("ascii")
        stdout, stderr = await asyncio.wait_for(process.communicate(stdin), 45)
        text = stdout.decode("utf-8", errors="replace")
        facts = [json.loads(line.removeprefix("QW3_LAUNCHER_FACTS ")) for line in text.splitlines()
                 if line.startswith("QW3_LAUNCHER_FACTS ")]
        result.update(exit_code=process.returncode, stdout=text, stderr=stderr.decode("utf-8", errors="replace"), facts=facts)
        result["passed"] = (process.returncode == 0 and len(facts) == 1
            and Path(facts[0]["executable"]).resolve() == (project / ".venv/Scripts/python.exe").resolve()
            and Path(facts[0]["prefix"]).resolve() == (project / ".venv").resolve()
            and (facts[0]["version"], facts[0]["requests"], facts[0]["pytest"]) == ("3.13.16", "2.34.2", "9.1.1"))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        await stop_process_tree(process)
        save(output / "launcher-grade.json", result)
    return result


async def grade(project, baseline, output):
    result = {"kind": "independent_python_environment_acceptance", "passed": False, "checks": {}}
    try:
        result["checks"].update(
            input_files_unchanged=all(digest(project / name) == value for name, value in baseline["input_sha256"].items()),
            user_path_unchanged=registry_paths()["user"] == baseline["registry_paths"]["user"],
            machine_path_unchanged=registry_paths()["machine"] == baseline["registry_paths"]["machine"],
            terminal_launcher_present=(project / "open-terminal.cmd").is_file(),
            dependency_lock_present=(project / "requirements.lock").is_file())
        results = []
        # A new cmd process must discover the project interpreter through normal
        # activation; the application interpreter and its PATH are not used.
        for command in ('python main.py', 'python -m pytest -q test_main.py', 'python -m pip check'):
            # cmd.exe does not use the C runtime's argv quote escaping. Put
            # reviewed shell syntax in a batch file and pass its plain basename.
            # The working directory may contain spaces without any shell path.
            script = project / "qw3-acceptance.cmd"
            if script.exists():
                raise ValueError("Acceptance script already exists; refusing to overwrite")
            script.write_text('@echo off\ncall "%~dp0.venv\\Scripts\\activate.bat"\n'
                              'if errorlevel 1 exit /b %errorlevel%\n' + command + '\n', encoding="ascii")
            process = None
            try:
                process = await asyncio.create_subprocess_exec(
                    str(Path(os.environ["SYSTEMROOT"]) / "System32/cmd.exe"), "/d", "/c", script.name,
                    cwd=project, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = await asyncio.wait_for(process.communicate(), 60)
            finally:
                await stop_process_tree(process)
                script.unlink()
            results.append({"command": command, "exit_code": process.returncode,
                            "stdout": stdout.decode("utf-8", errors="replace"),
                            "stderr": stderr.decode("utf-8", errors="replace")})
        result["fresh_terminal_commands"] = results
        result["checks"]["all_commands_succeeded"] = all(r["exit_code"] == 0 for r in results)
        if results[0]["exit_code"] != 0:
            raise ValueError("Project did not run successfully in a new activated terminal")
        facts = json.loads(results[0]["stdout"])
        checks = result["checks"] | {
                  "project_interpreter": Path(facts["executable"]).resolve() == (project / ".venv/Scripts/python.exe").resolve(),
                  "isolated_environment": Path(facts["prefix"]).resolve() == (project / ".venv").resolve(),
                  "python_version": facts["version"] == "3.13.16", "requests_version": facts["requests"] == "2.34.2",
                  "pytest_version": facts["pytest"] == "9.1.1", "project_result": facts["total"] == 42,
                  "request_preparation": facts["url"] == "https://example.com/",
                  "tests_executed": "1 passed" in results[1]["stdout"],
                  "input_files_unchanged": all(digest(project / name) == value for name, value in baseline["input_sha256"].items()),
                  "user_path_unchanged": registry_paths()["user"] == baseline["registry_paths"]["user"],
                  "machine_path_unchanged": registry_paths()["machine"] == baseline["registry_paths"]["machine"]}
        result.update(checks=checks, passed=all(checks.values()))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    save(output / "independent-grade.json", result)
    return result


async def grade_repair(project, baseline, output):
    result = {"kind": "independent_python_repair_acceptance", "passed": False, "checks": {}}
    try:
        fixture = baseline["fixture"]
        reference = Path(fixture["reference_project"])
        checks = result["checks"]
        checks["input_files_unchanged"] = all(digest(Path(fixture[key]) / name) == value
            for key, label in (("target_project", "repair-target"), ("reference_project", "unrelated-project"))
            for name, value in fixture["source_hashes"][label].items())
        checks["unrelated_project_unchanged"] = fixture["unrelated_project_files"] == {
            str(p.relative_to(reference)): digest(p) for p in reference.rglob("*") if p.is_file()}
        checks["registry_paths_unchanged"] = registry_paths() == baseline["registry_paths"]
        commands = []
        for label, folder, args in (
            ("target_program", project, ["main.py"]),
            ("target_tests", project, ["-m", "pytest", "-q", "-p", "no:cacheprovider", "test_main.py"]),
            ("target_dependencies", project, ["-m", "pip", "--isolated", "check"]),
            ("reference_program", reference, ["main.py"]),
        ):
            process = None
            try:
                process = await asyncio.create_subprocess_exec(str(folder / ".venv/Scripts/python.exe"), "-I", "-B", *args,
                    cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = await asyncio.wait_for(process.communicate(), 60)
                commands.append(dict(label=label, exit_code=process.returncode, stdout=stdout.decode("utf-8", errors="replace"),
                                     stderr=stderr.decode("utf-8", errors="replace")))
            finally:
                await stop_process_tree(process)
        result["commands"] = commands
        checks["programs_and_dependencies_passed"] = all(c["exit_code"] == 0 for c in commands)
        checks["target_result"] = commands[0]["stdout"].strip() == "42"
        checks["target_test_executed"] = "1 passed" in commands[1]["stdout"]
        checks["reference_result"] = commands[3]["stdout"].strip() == "42"
        checks["unrelated_project_unchanged_after_probes"] = fixture["unrelated_project_files"] == {
            str(p.relative_to(reference)): digest(p) for p in reference.rglob("*") if p.is_file()}
        result["passed"] = all(checks.values())
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    save(output / "independent-grade.json", result)
    return result


async def run(configuration):
    config = json.loads(configuration.read_text(encoding="utf-8-sig"))
    identity = config["run"]
    output = Path("C:/Qw3Computer/product-runs") / identity
    output.mkdir(parents=True, exist_ok=False)
    share = Path("//host.lan/Data")
    repair = config.get("task") == "product-python-repair"
    fixture = None
    if repair:
        fixture_path = share / (identity + "-fixture.json")
        if digest(fixture_path) != config["repair_fixture_sha256"]:
            raise ValueError("Repair fixture evidence changed")
        fixture = json.loads(fixture_path.read_text())
        if fixture["status"] != "ready_fault_injected" or fixture["model_used"] is not False:
            raise ValueError("Repair fixture is not independently prepared")
    workspace = Path(fixture["workspace"]) if repair else Path("C:/Qw3Computer/outputs") / (identity + "-software")
    project = workspace / "projects" / ("repair-target" if repair else "python-demo")
    grader = grade_repair if repair else grade
    bundle = Path("C:/Qw3Computer/clients") / config["bundle_sha256"][:12]
    manifest = json.loads((bundle / "manifest.json").read_text())
    if "opt_in_software_workflows" not in manifest["runtime_capabilities"]:
        raise ValueError("Native bundle does not support software mode")
    for name, expected in manifest["source_files"].items():
        if digest(bundle / "app" / name) != expected:
            raise ValueError("Installed native source changed")
    if digest(Path(__file__)) != config["guest_script_sha256"]:
        raise ValueError("Guest launcher changed")
    if digest(Path("C:/Qw3Computer/executor.ps1")) != config["executor_sha256"]:
        raise ValueError("Guest executor differs from the recorded candidate")
    report = {"kind": "native_local_python_product_trial", "run": identity, "status": "preparing", "passed": False,
              "model_used": True, "advisor_used": False, "execution_mode": "local_model_with_controlled_software_tools",
              "bundle_sha256": config["bundle_sha256"], "cold_workspace": not repair, "started_at": time.time(),
              "scenario": "product-python-repair" if repair else "product-python-new", "thinking": config.get("thinking", False)}
    app = None
    owned_ready = False
    token_path = Path("C:/Qw3Computer") / (identity + "-model-token.txt")
    token_path.write_bytes((share / (identity + "-model-token.txt")).read_bytes())

    def progress(stage):
        report["stage"] = stage
        save(output / "progress.json", report)
        save(share / (identity + "-progress.json"), {"stage": stage, "time": time.time()})

    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=30)) as client:
        async def api(path, body=None):
            async with client.request("POST" if body is not None else "GET", "http://127.0.0.1:18083" + path, json=body) as response:
                response.raise_for_status()
                return await response.json()
        try:
            if repair:
                baseline = {"registry_paths": registry_paths(), "fixture": fixture}
                pre = output / "before"; pre.mkdir()
                initial_grade = await grade_repair(project, baseline, pre)
                initial_launcher = await grade_launcher(project, pre)
                if (initial_grade["passed"] or initial_launcher["passed"]
                        or not initial_grade["checks"].get("unrelated_project_unchanged")):
                    raise ValueError("Repair faults or reference preservation are not present at task start")
            else:
                if workspace.exists():
                    raise ValueError("Product workspace already exists; clean-install trial is invalid")
                project.mkdir(parents=True)
                (project / "main.py").write_text(MAIN, encoding="utf-8")
                (project / "test_main.py").write_text(TEST, encoding="utf-8")
                baseline = {"registry_paths": registry_paths(), "venv_absent": not (project / ".venv").exists(),
                            "input_sha256": {name: digest(project / name) for name in ("main.py", "test_main.py")}}
            save(output / "initial-state.json", baseline)
            try:
                await api("/api/state")
            except aiohttp.ClientConnectorError:
                pass
            else:
                raise ValueError("Another application occupies the native task port")
            env = dict(os.environ, PATH=str(bundle / "codex/bin") + os.pathsep + os.environ.get("PATH", ""), PYTHONUTF8="1")
            args = [sys.executable, "-X", "utf8", "-m", "qw3_computer.app", "--desktop", "http://127.0.0.1:5000",
                    "--desktop-token", "C:/Qw3Computer/executor-token.txt", "--backend", config["backend"],
                    "--backend-token", str(token_path), "--runtime", str(output / "app"), "--software-root", str(workspace),
                    "--tool-call-limit", "100", "--output-policy", "task", "--trace-backend"]
            if config.get("thinking"):
                args.append("--thinking")
            with (output / "app.stdout.log").open("wb") as stdout, (output / "app.stderr.log").open("wb") as stderr:
                app = await asyncio.create_subprocess_exec(*args, env=env, cwd=bundle / "app", stdout=stdout, stderr=stderr,
                                                          creationflags=subprocess.CREATE_NO_WINDOW)
            progress("waiting_for_native_application")
            async with asyncio.timeout(60):
                while True:
                    if app.returncode is not None:
                        raise RuntimeError("Native application exited at startup")
                    try:
                        if (await api("/api/state"))["ready"]:
                            owned_ready = True
                            break
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(.3)
            # Show the product's own progress UI for the continuous recording.
            edge = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft/Edge/Application/msedge.exe"
            subprocess.Popen([str(edge), "--no-first-run", "--no-default-browser-check", "--disable-sync",
                              "--user-data-dir=C:/Qw3Computer/ProductBrowserProfile", "--app=http://127.0.0.1:18083"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            await asyncio.sleep(2)
            prompt = ("请为选定软件工作目录里已有的 python-demo 项目配置可用的 Python 环境。"
                      "使用 Python 3.13.16、requests 2.34.2 和 pytest 9.1.1，创建项目专用 .venv，保留 main.py 和 test_main.py，"
                      "不修改系统 PATH 或其他 Python。使用已提供的 python-web 软件配置档。"
                      "等待安装结束并核对工具的实际验证结果，再说明项目路径和下次如何打开项目终端。"
                      "如果失败，先报告具体错误；不要仅凭开始安装就说完成。")
            if repair:
                prompt = ("请修复选定软件工作目录中的 repair-target Python 项目。现在程序不能正常运行，"
                          "打开项目终端选择的环境也不对。它原本使用 python-web 配置档。"
                          "先诊断实际问题，只修复这个项目，保留源文件、系统 PATH 和 unrelated-project 项目。"
                          "等待实际验证通过后，说明发现的问题、做了什么修复、下次如何打开正确的项目终端。")
            (output / "prompt.txt").write_text(prompt, encoding="utf-8")
            progress("model_running")
            report["task"] = await run_task("http://127.0.0.1:18083", prompt, output / "task", config["timeout_seconds"])
            report["terminal_state"] = await api("/api/state")
            progress("independent_verification")
            report["grade"] = await grader(project, baseline, output)
            report["passed"] = report["task"]["outcome"] == "agent_completed" and report["grade"]["passed"]
            report["launcher_grade"] = await grade_launcher(project, output)
            report["passed"] = report["passed"] and report["launcher_grade"]["passed"]
            if report["task"]["outcome"] == "agent_completed":
                progress("reopening_application")
                before = report["terminal_state"]
                if not before["desktop_paused"] or any(j["status"] != "completed" for j in before["software"]["jobs"]):
                    raise ValueError("Cannot close/reopen application with unresolved actions or software jobs")
                saved_jobs = {str(p.relative_to(workspace)): digest(p) for p in (workspace / "jobs").rglob("job.json")}
                prior_pid = app.pid
                await stop_process_tree(app)
                owned_ready = False
                with (output / "reopened.stdout.log").open("wb") as stdout, (output / "reopened.stderr.log").open("wb") as stderr:
                    app = await asyncio.create_subprocess_exec(*args, env=env, cwd=bundle / "app", stdout=stdout, stderr=stderr,
                                                              creationflags=subprocess.CREATE_NO_WINDOW)
                async with asyncio.timeout(60):
                    while True:
                        if app.returncode is not None:
                            raise RuntimeError("Reopened application exited at startup")
                        try:
                            reopened = await api("/api/state")
                            if reopened["ready"]:
                                owned_ready = True
                                break
                        except aiohttp.ClientError:
                            pass
                        await asyncio.sleep(.3)
                restart_output = output / "reopened"; restart_output.mkdir()
                restarted_grade = await grader(project, baseline, restart_output)
                restarted_launcher = await grade_launcher(project, restart_output)
                restart_checks = {"different_process": app.pid != prior_pid, "task_identity_preserved": reopened["id"] == before["id"],
                    "completion_preserved": reopened["status"] == "completed" and reopened.get("software_verified") is True,
                    "desktop_paused": reopened["desktop_paused"], "job_records_unchanged": bool(saved_jobs) and
                    saved_jobs == {str(p.relative_to(workspace)): digest(p) for p in (workspace / "jobs").rglob("job.json")},
                    "fresh_project_checks_passed": restarted_grade["passed"], "delivered_launcher_passed": restarted_launcher["passed"]}
                report["reopen"] = {"checks": restart_checks, "passed": all(restart_checks.values()),
                                    "prior_pid": prior_pid, "new_pid": app.pid, "state": reopened}
                report["passed"] = report["passed"] and report["reopen"]["passed"]
            report["status"] = "complete"
        except Exception as exc:
            report.update(status="halted", error=f"{type(exc).__name__}: {exc}")
        finally:
            if owned_ready:
                try:
                    stopped = await api("/api/control", {"command": "stop"})
                    report["desktop_stopped"] = stopped["desktop_paused"]
                except Exception as exc:
                    report["stop_error"] = str(exc)
            if app and app.returncode is None:
                try:
                    await stop_process_tree(app)
                except Exception as exc:
                    report.update(passed=False, cleanup_error=str(exc), needs_reconciliation=True)
            if (workspace / "jobs").exists():
                shutil.copytree(workspace / "jobs", output / "software-jobs")
            artifacts = output / "project"; artifacts.mkdir(exist_ok=True)
            for name in ("main.py", "test_main.py", "requirements.lock", "ENVIRONMENT.txt", "open-terminal.cmd", "environment-verification.json"):
                if (project / name).is_file():
                    shutil.copyfile(project / name, artifacts / name)
            report["finished_at"] = time.time()
            save(output / "result.json", report)
            # Export diagnostics, never the bridge or MCP bearer credentials.
            shutil.copytree(output, share / (identity + "-evidence"),
                            ignore=shutil.ignore_patterns("mcp-token.txt", "model-token.txt", "auth.json"))
            save(share / (identity + "-result.json"), report)
            token_path.unlink(missing_ok=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = asyncio.run(run(args.configuration))
    except Exception as exc:
        config = json.loads(args.configuration.read_text(encoding="utf-8-sig"))
        result = {"status": "halted", "passed": False, "error": f"{type(exc).__name__}: {exc}",
                  "needs_reconciliation": True}
        save(Path("//host.lan/Data") / (config["run"] + "-result.json"), result)
    print(json.dumps({k: result.get(k) for k in ("status", "passed", "error")}), flush=True)
