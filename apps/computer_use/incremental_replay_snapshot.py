"""Append terminal public replays while retaining the qualified old snapshot.

Only public summary fields and hash-checked movies enter the served directory.
Engine startup interruptions remain separate from model task denominators.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import sqlite3
import time
from unittest.mock import patch

from PIL import Image
import cross_device_replay as replay


def identity(row):
    return row["kind"], row.get("cohort"), row["attempt"]


def qualified_previous(directory, qualification):
    data, q = replay.read(directory / "summary.json"), replay.read(qualification)
    if (q.get("passed") is not True or q.get("actual_browser") is not True or q.get("checks_passed", 0) < 24
            or q.get("model_calls") != 0 or q.get("device_inputs") != 0 or q.get("page_errors") != []
            or q.get("snapshot_summary_sha256") != replay.digest(directory / "summary.json")
            or q.get("snapshot_html_sha256") != replay.digest(directory / "index.html")
            or data.get("kind") != "public-terminal-attempts-snapshot"
            or data.get("contains_private_requests") is not False or data.get("contains_credentials") is not False):
        raise ValueError("Previous public snapshot differs from its actual browser qualification")
    rows = data["attempts"]
    if len({identity(r) for r in rows}) != len(rows): raise ValueError("Previous attempt identity is ambiguous")
    return data


def video_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"trial-[0-9]+(?:-partial)?\.mp4", value):
        raise ValueError("Only named public video assets may be served")
    return value


def retain_movies(previous, output, rows):
    for row in rows:
        if row.get("video") is None: continue
        name = video_name(row["video"]); source = previous / name
        if not source.is_file() or source.is_symlink() or replay.digest(source) != row["video_sha256"]:
            raise ValueError("Previous movie changed or is not a regular public asset")
        replay.terminal_media(source, output / name)


def combine(previous, additions):
    data = copy.deepcopy(previous); rows = data["attempts"]; seen = {identity(r) for r in rows}
    for row in additions:
        if row["kind"] != "development-osworld" or identity(row) in seen:
            raise ValueError("Append only new terminal development attempts")
        rows.append(copy.deepcopy(row)); seen.add(identity(row))
    rows.sort(key=lambda r: (r.get("started_at") or 0, r["attempt"]))
    data.update(exported_at=time.time(), development_osworld_attempts=sum(r["kind"] == "development-osworld" for r in rows))
    return data


def observed_geometry(root, row):
    """Compare retained desktop PNGs with movie size; never infer hidden pixels."""
    database = root / "app/desktop/desktop.sqlite"
    dimensions = row.get("video_dimensions")
    if not database.is_file() or not dimensions: return {"status": "unavailable"}
    counts = 0; sizes = set()
    connection = sqlite3.connect("file:" + str(database.resolve()) + "?mode=ro&immutable=1", uri=True)
    try:
        for (metadata,) in connection.execute("SELECT metadata FROM observations"):
            m = json.loads(metadata); sha = m.get("sha256")
            if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha): raise ValueError("Invalid retained screenshot identity")
            source = root / "app/desktop/screenshots" / (sha + ".png")
            if replay.digest(source) != sha: raise ValueError("Retained screenshot hash changed")
            with Image.open(source) as picture: size = picture.size
            if size != (m.get("width"), m.get("height")): raise ValueError("Source screenshot and observation dimensions differ")
            counts += 1; sizes.add(size)
    finally: connection.close()
    mismatch = any(size != tuple(dimensions) for size in sizes)
    if mismatch:
        row.update(full_screen_coverage=False, video_usable_for_full_task_trace=False,
                   time_coverage="录像尺寸与保留的桌面观察不同，不能作为完整桌面轨迹或延迟对照")
    row.update(retained_desktop_observation_count=counts, retained_desktop_observation_dimensions=[list(s) for s in sorted(sizes)],
               retained_observation_dimensions_match_video=bool(counts) and not mismatch)
    return dict(status="verified_retained_sources", count=counts, sizes=[list(s) for s in sorted(sizes)],
                mismatch=mismatch, database_sha256=replay.digest(database), full_task_pixel_alignment_proven=False)


def startup_note(audit):
    a = replay.read(audit); f = a["capacity_failure"]
    if (a.get("kind") != "actual_kvmem_capacity_and_successors_progress" or f.get("model_trial_dispatched") is not False
            or f.get("original_failure_preserved") is not True or f.get("required_blocks") != 1536
            or f.get("allowed_blocks") != 1304 or replay.digest(Path(f["log"])) != f.get("sha256")):
        raise ValueError("Startup interruption does not match its preserved pre-task evidence")
    return dict(title="035 · KVMem ON 启动中断", reason="固定配置需要1536个显存块，当时容量检查只允许1304个",
                engine_log_sha256=f["sha256"], model_task_started=False, business_grade=None)


def annotate_startup(html):
    before = '<p id="summary"></p>'
    anchor = "all=data.attempts;const s=data.formal_summary;"
    if html.count(before) != 1 or html.count(anchor) != 1: raise ValueError("Unexpected qualified replay template")
    html = html.replace(before, before + '<details id="startupNotes"><summary>启动中断记录（未开始模型任务）</summary><div id="startupNotesBody"></div></details>')
    html = html.replace(anchor, anchor + "const notes=data.pre_task_interruptions||[];"
        "for(const item of notes){const p=document.createElement('p');p.textContent=item.title+'：'+item.reason+'。未开始模型任务，未计入模型成功率。';el('startupNotesBody').append(p)}")
    return html


def export(previous, qualification, campaigns, output, work, port, capacity_audit):
    previous, output, work = map(lambda p: Path(p).resolve(), (previous, output, work))
    if output.exists() or work.exists() or output == work or output in work.parents:
        raise ValueError("Use fresh public and separate private work directories")
    data = qualified_previous(previous, qualification); existing = {identity(r): r for r in data["attempts"]}
    roots = {}; original_rows = replay.matched_rows
    def new_rows(campaign):
        memory = replay.read(campaign / "protocol.json").get("kind") == "declared_local_qw3_kvmem_osworld_comparison"
        for root, row in original_rows(campaign):
            if memory: row["task"] = "OSWorld KVMem 对照 " + replay.read(root / "manifest.json")["task_id"]
            key = identity(row)
            if key in existing:
                # Old geometry/media annotations remain intact. All original
                # task facts must still equal this newly read source record.
                if any(existing[key].get(k) != value for k, value in row.items()):
                    raise ValueError("Previously exported original task facts changed")
                continue
            if key in roots: raise ValueError("Duplicate new campaign attempt")
            roots[key] = root; yield root, row
    work.mkdir(parents=True, exist_ok=False)
    with patch.object(replay, "matched_rows", new_rows):
        new = replay.export(None, [], work / "additions", port, matched_campaigns=campaigns)
    output.mkdir(parents=True, exist_ok=False); retain_movies(previous, output, data["attempts"])
    next_index = 1 + max((int(re.search(r"trial-([0-9]+)", r["video"]).group(1))
                         for r in data["attempts"] if r.get("video")), default=-1)
    geometry = {}
    for i, row in enumerate(new["attempts"]):
        geometry[str(identity(row))] = observed_geometry(roots[identity(row)], row)
        if row.get("video"):
            source = work / "additions" / video_name(row["video"])
            name = f"trial-{next_index+i}" + ("-partial.mp4" if row["video"].endswith("-partial.mp4") else ".mp4")
            if replay.digest(source) != row["video_sha256"]: raise ValueError("New exported movie changed")
            replay.terminal_media(source, output / name); row["video"] = name
    merged = combine(data, new["attempts"]); merged["pre_task_interruptions"] = [startup_note(capacity_audit)]
    (output / "summary.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    (output / "index.html").write_text(annotate_startup((previous / "index.html").read_text()))
    (output / "serve.py").write_text((work / "additions/serve.py").read_text())
    report = dict(kind="qualified_incremental_public_replay_export", passed=True, old_rows=len(data["attempts"]),
                  appended=len(new["attempts"]), total=len(merged["attempts"]), source_geometry=geometry,
                  original_formal_summary_unchanged=merged["formal_summary"] == data["formal_summary"],
                  old_rows_and_video_hashes_unchanged=True, startup_events_excluded_from_model_denominators=True,
                  source_sha256=replay.digest(Path(__file__)), previous_summary_sha256=replay.digest(previous / "summary.json"),
                  summary_sha256=replay.digest(output / "summary.json"), html_sha256=replay.digest(output / "index.html"),
                  model_calls=0, device_inputs=0)
    (work / "export.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "qualification", "output", "work", "capacity-audit"): p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--campaigns", type=Path, nargs="+", required=True); p.add_argument("--port", type=int, default=18159)
    args = p.parse_args()
    if not 1024 <= args.port <= 65535: p.error("Use an unprivileged local port")
    report = export(args.previous, args.qualification, args.campaigns, args.output, args.work, args.port, args.capacity_audit)
    print(json.dumps({k: report[k] for k in ("old_rows", "appended", "total", "model_calls", "device_inputs")}))


if __name__ == "__main__": main()
