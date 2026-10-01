"""Resolve only the ten task/assets pinned in the reviewed selection manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SELECTION = ROOT / "docs/computer_use_osworld_windows_10_20261001.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scenario(task):
    data = json.loads(SELECTION.read_text())
    item = next(s for s in data["scenarios"] if s["source_task_id"] == task)
    if sha256(ROOT/item["source_file"]) != item["source_sha256"]:
        raise ValueError("Selected task source differs from pinned manifest")
    return item


def asset(task, name):
    item = scenario(task)
    key = f"task_{task}/{name}"
    expected = item["assets"][key]
    path = ROOT / "third_party/OSWorld-V2/cache/osworld_v2_assets" / key
    if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        raise ValueError("Task material differs from pinned manifest: " + key)
    return path
