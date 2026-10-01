"""Load reviewed pure grading code, without importing Linux setup controllers."""
import ast
import logging
from functools import lru_cache

from .source import ROOT, asset, scenario, sha256


def source_asset(key):
    task,name=key.removeprefix("task_").split("/",1)
    return str(asset(task,name))


def get_cloud_file(_env, specification):
    # The selected evaluator's asset() already resolved and hash-checked this
    # file. Reject any attempt to turn grading into a network download.
    from pathlib import Path
    path=Path(specification["path"]).resolve()
    prefix=(ROOT/"third_party/OSWorld-V2/cache/osworld_v2_assets").resolve()
    relative=path.relative_to(prefix)
    checked=source_asset(str(relative))
    if path!=Path(checked):raise ValueError("Unexpected evaluator material")
    return checked


@lru_cache(maxsize=None)
def task004_class():
    specification=scenario("004");path=ROOT/specification["source_file"]
    tree=ast.parse(path.read_text())
    body=[]
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and (node.module or "").startswith(("desktop_env", "evaluation_examples")):
            continue
        if isinstance(node,ast.ClassDef):
            node.body=[n for n in node.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name not in ("setup","evaluate")]
        body.append(node)
    namespace={"BaseTask":dict,"asset":source_asset,"get_cloud_file":get_cloud_file}
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),"exec"),namespace)
    return namespace["Task004"]


def pure_function(relative,name,namespace):
    path=ROOT/"third_party/OSWorld-V2"/relative
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    environment={"logger":logging.getLogger("osworld-windows-grader"),**namespace}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),environment)
    return environment[name],sha256(path)


@lru_cache(maxsize=None)
def pure_task_module(task):
    """Reviewed module-level graders only; no task setup/evaluate methods."""
    if task not in {"038", "043", "080"}:
        raise ValueError("Task has not been reviewed for pure module loading")
    specification = scenario(task)
    path = ROOT / specification["source_file"]
    body = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(("desktop_env", "evaluation_examples")):
            continue
        if isinstance(node, ast.ClassDef):
            continue
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"TASK_CLASS", "HF_BASE"} for t in node.targets):
            continue
        body.append(node)
    namespace = {"__name__": "osworld_windows.upstream_task_" + task, "__file__": str(path), "asset": source_asset}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace
