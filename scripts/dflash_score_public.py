#!/usr/bin/env python3
"""Score frozen public completions; report quality without asserting noninferiority."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import secrets
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--python", type=Path, required=True, help="Isolated quality environment Python")
p.add_argument("--references", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("manifests", type=Path, nargs="+")
a = p.parse_args()
assert not a.output.exists()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
provenance = json.loads((a.references / "provenance.json").read_text())
references = {}
for name, spec in provenance["datasets"].items():
    path = a.references / spec["file"]
    assert sha(path) == spec["sha256"]
    references[name] = {row["ordinal"]: row for row in map(json.loads, path.read_text().splitlines())}
python = a.python.absolute()
environment = python.parent.parent
base = Path(subprocess.check_output([str(python), "-I", "-c", "import sys;print(sys.base_prefix)"], text=True).strip())
base_alias = Path(subprocess.check_output([str(python), "-I", "-c", "import sys;print(sys._base_executable)"], text=True).strip()).parent.parent
versions = json.loads(subprocess.check_output([str(python), "-I", "-c",
    "import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ['math-verify','latex2sympy2-extended','antlr4-python3-runtime','sympy','numpy']}))"], text=True))
worker = Path(__file__).with_name("dflash_quality_worker.py")
command = ["bwrap", "--unshare-all", "--new-session", "--die-with-parent", "--cap-drop", "ALL",
           "--clearenv", "--setenv", "LANG", "C.UTF-8", "--setenv", "OPENBLAS_NUM_THREADS", "1",
           "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
           "--ro-bind", str(base), str(base), "--ro-bind", str(environment), str(environment),
           "--ro-bind", str(base), str(base_alias),
           "--ro-bind", str(worker), "/worker.py", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
           "--chdir", "/tmp", str(python), "-I", "-B", "/worker.py"]


def evaluate(job):
    nonce = secrets.token_hex(16)
    with tempfile.TemporaryFile() as output:
        try:
            process = subprocess.run(command, input=json.dumps({**job, "nonce": nonce}).encode(),
                                     stdout=output, stderr=output, timeout=15)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "passed": False}
        output.seek(0)
        text = output.read(1 << 20).decode(errors="replace")
    for line in reversed(text.splitlines()):
        try:
            result = json.loads(line)
        except ValueError:
            continue
        if isinstance(result, dict) and result.get("nonce") == nonce:
            result.pop("nonce")
            result["exit_code"] = process.returncode
            if process.returncode:
                result.update(passed=False, status="worker_exit_failure")
            return result
    return {"status": "worker_exit_failure", "passed": False,
            "exit_code": process.returncode, "diagnostic_tail": text[-500:]}


def math_job(reference, text):
    answer = reference["answer"].split("####")[-1].strip()
    gold = answer if answer.startswith("$") else "$" + answer + "$"
    final = text.rsplit("</think>", 1)[1] if "</think>" in text else ""
    return {"kind": "math", "gold": gold, "whole": text, "final": final}


def code_job(task, reference, code, canonical=False):
    prelude = ""
    if task == "humaneval":
        prompt, entry = reference["prompt"], reference["entry_point"]
        if canonical:
            code = prompt + reference["canonical_solution"]
        elif re.search(r"\bdef\s+" + re.escape(entry) + r"\s*\(", code):
            prelude = prompt.split("def " + entry, 1)[0]
        elif code and all(not line.strip() or line.startswith((" ", "\t")) for line in code.splitlines()):
            code = prompt + code
        return {"kind": "code", "prelude": prelude, "code": code, "tests": reference["test"] + "\ncheck(" + entry + ")\n"}
    imports = "\n".join(reference["test_imports"])
    return {"kind": "code", "prelude": imports, "code": reference["code"] if canonical else code,
            "tests": "\n".join(reference["test_list"])}


def extract_code(text):
    if "</think>" not in text:
        return ""
    final = text.rsplit("</think>", 1)[1]
    blocks = re.findall(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", final, re.S | re.I)
    return blocks[0].strip("\r\n") if blocks else final.strip("\r\n")


def mbpp_interface(reference, code):
    """Diagnostic alias only; never select implementations using test outcomes."""
    if not code.strip():
        return code, {"action": "no_final_code"}
    try:
        tree, gold = ast.parse(code), ast.parse(reference["code"])
        tests = ast.parse("\n".join(reference["test_list"]))
    except (SyntaxError, RecursionError):
        return code, {"action": "syntax_not_normalizable"}
    definitions = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    gold_defs = {n.name for n in gold.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    calls = {n.func.id for n in ast.walk(tests) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    expected = calls & gold_defs
    if len(expected) != 1:
        return None, {"action": "ambiguous_reference_interface"}
    name = next(iter(expected))
    if name in definitions:
        return code, {"action": "already_named", "expected": name}
    if len(definitions) != 1:
        return None, {"action": "ambiguous_candidate_interface", "expected": name}
    actual = next(iter(definitions))
    return code + "\n" + name + " = " + actual + "\n", {"action": "single_function_alias", "expected": name, "actual": actual}


# Verify the sandbox/evaluator before accepting any model score, including
# early exits that would otherwise look like successful test processes.
self_checks = [
    ({"kind": "code", "code": "def f(): return 7", "tests": "assert f() == 7"}, True),
    ({"kind": "code", "code": "def f(): return 8", "tests": "assert f() == 7"}, False),
    ({"kind": "code", "code": "import os;os._exit(0)", "tests": "assert False"}, False),
    ({"kind": "code", "code": "import os", "tests": "assert not os.path.exists('/home/chaidi/qw3');assert not os.path.exists('/data')"}, True),
    ({"kind": "code", "prelude": "from math import sqrt", "code": "from __future__ import annotations\ndef f(): return sqrt(4)", "tests": "assert f() == 2"}, True),
    ({"kind": "math", "gold": "$1/2$", "whole": "$0.5$", "final": "$0.5$"}, True),
    ({"kind": "math", "gold": "$1/2$", "whole": "$0.6$", "final": "$0.6$"}, False),
]
checks = [evaluate(job) for job, _ in self_checks]
data = {"status": "running", "quality_qualified": False, "versions": versions,
        "reference_provenance_sha256": sha(a.references / "provenance.json"),
        "source_sha256": {str(Path(__file__).relative_to(root)): sha(Path(__file__)),
                          str(worker.relative_to(root)): sha(worker)},
        "sandbox_command": command, "self_checks": checks, "canonical_checks": [], "runs": [],
        "scoring_rules": ["All measured responses retained, including length-truncated outputs.",
                          "Math reports both whole-text Math-Verify and usable final-answer scores after </think>.",
                          "Code uses the first Python/untagged fence after </think>, or the unmodified unfenced final text.",
                          "HumanEval preserves prompt imports or appends body-only completions to the prompt; MBPP uses official test imports.",
                          "Canonical solutions must pass the same official tests before model scores are accepted.",
                          "No code repair, choosing among candidates by tests, or LLM judging.",
                          "MBPP additionally reports a separate interface diagnostic: alias one uniquely defined function to the single official tested name; preserve strict scores.",
                          "MT-Bench requires separate judging; not automatically scored or marked passed.",
                          "These scores do not establish statistical noninferiority or public performance qualification."]}


def save():
    a.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


save()
if not all(result["passed"] == expected for result, (_, expected) in zip(checks, self_checks)):
    data.update(status="failed", error="sandbox/evaluator self-check failed")
    save()
    raise RuntimeError(data["error"] + ": " + repr(checks))
canonical = {}
try:
    for manifest in a.manifests:
        source = json.loads(manifest.read_text())
        assert source["status"] == "passed_scoped"
        assert source["input_manifest_sha256"] == provenance["input_manifest_sha256"]
        for run in source["runs"]:
            path = root / run["responses"]
            assert sha(path) == run["responses_sha256"]
            scored = {"manifest": str(manifest), "manifest_sha256": sha(manifest), "route": run["route"], "records": []}
            data["runs"].append(scored)
            for record in map(json.loads, path.read_text().splitlines()):
                if record["warmup"]:
                    continue
                task, ordinal = record["task"], record["ordinal"]
                reference = references[task][ordinal]["reference"]
                text = record["response"]["choices"][0]["text"]
                row = {"task": task, "ordinal": ordinal, "answer_sha256": digest(text),
                       "finish_reason": record["finish_reason"], "thinking_closed": "</think>" in text}
                key = (task, ordinal)
                if task in ("humaneval", "mbpp"):
                    if key not in canonical:
                        canonical[key] = evaluate(code_job(task, reference, "", canonical=True))
                        data["canonical_checks"].append({"task": task, "ordinal": ordinal, **canonical[key]})
                    assert canonical[key]["passed"], f"canonical test failed: {key} {canonical[key]}"
                    code = extract_code(text)
                    row["code_sha256"] = digest(code)
                    grade = evaluate(code_job(task, reference, code)) if code.strip() else {"status": "no_final_code", "passed": False}
                    if task == "mbpp":
                        adapted, action = mbpp_interface(reference, code)
                        row["interface_normalization"] = action
                        row["interface_grade"] = (evaluate(code_job(task, reference, adapted))
                            if adapted and adapted != code else grade if adapted is not None else
                            {"status": "ambiguous_interface", "passed": None})
                elif task in ("gsm8k", "math500"):
                    grade = evaluate(math_job(reference, text))
                    assert grade["status"] != "reference_parse_failure", key
                else:
                    grade = {"status": "unscored_requires_judge", "passed": None}
                row["grade"] = grade
                scored["records"].append(row)
                save()
            scored["summary"] = {}
            for task in dict.fromkeys(r["task"] for r in scored["records"]):
                rows = [r for r in scored["records"] if r["task"] == task]
                graded = [r for r in rows if r["grade"]["passed"] is not None]
                summary = {"requests": len(rows), "scored": len(graded),
                           "final_correct": sum(r["grade"]["passed"] for r in graded),
                           "thinking_not_closed": sum(not r["thinking_closed"] for r in rows)}
                if task in ("gsm8k", "math500"):
                    summary["whole_text_correct"] = sum(r["grade"].get("whole_correct", False) for r in rows)
                if task == "mbpp":
                    summary["interface_correct"] = sum(r["interface_grade"]["passed"] is True for r in rows)
                    summary["interface_unresolved"] = sum(r["interface_grade"]["passed"] is None for r in rows)
                scored["summary"][task] = summary
                print(run["route"], task, summary, flush=True)
    data["status"] = "closed_scoped_quality_diagnostic"
except BaseException as error:
    data.update(status="failed", error=repr(error))
    raise
finally:
    save()
