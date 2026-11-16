#!/usr/bin/env python3
"""Summarize completed BFCL Query-history pairs and matched-input selections."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

from run_bfcl_query_history_ab import input_identity


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def csv_file(path, records):
    if not records:
        path.write_text("")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in record.items()})


def selections(path):
    result = {}
    if not path.exists():
        return result
    current = None
    for row in read_jsonl(path):
        if row.get("type") == "meta":
            current = row["trace_tag"]
            result[current] = {"meta": row, "selected": []}
        elif current is not None and row.get("sel"):
            result[current]["selected"].append((row["p0"], row["nt"]))
    return result


def group_metrics(rows):
    requests = [s for r in rows for t in r["turns"] for s in t["steps"]]
    commits = [c for s in requests for c in s["query_history_commits"]]
    later = [(t, t["steps"][0]) for r in rows for t in r["turns"]
             if t["turn_index"] > 0 and t["steps"][0]["usage"]["prompt_tokens"] > 32768]
    return {"completed_conversations": len(rows),
        "official_conversation_passes": sum(r["official_grade"]["valid"] for r in rows),
        "diagnostic_turn_conditions_passed": sum(r["local_conditions_passed"] for r in rows),
        "diagnostic_turn_state_checks_passed": sum(t["state_check"]["valid"] for r in rows for t in r["turns"]),
        "diagnostic_turn_response_checks_passed": sum(t["response_check"]["valid"] for r in rows for t in r["turns"]),
        "turns_completed": sum(len(r["turns"]) for r in rows),
        "total_requests": len(requests),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "tool_calls": sum(r["tool_calls"] for r in rows),
        "tool_execution_errors": sum(r["tool_execution_errors"] for r in rows),
        "requests_finishing_at_length_limit": sum(s["finish_reason"] == "length" for s in requests),
        "turns_with_truncated_natural_answer": sum(t.get("answer_truncated", False) for r in rows for t in r["turns"]),
        "conversations_exhausting_context_limit": sum(any(t.get("context_limit_reached", False)
            for t in r["turns"]) for r in rows),
        "query_history_commits": len(commits),
        "positive_query_history_commits": sum("observed=1" in c for c in commits),
        "later_queries_starting_over_32k": len(later),
        "diagnostic_conditions_passed_on_later_queries_over_32k": sum(t["local_conditions_valid"] for t, _ in later),
        "diagnostic_response_checks_passed_on_later_queries_over_32k": sum(t["response_check"]["valid"] for t, _ in later),
        "request_latency_seconds_sum": sum(s["latency_s"] for s in requests),
        "episode_wall_seconds_sum": sum(r["wall_seconds_including_startup"] for r in rows)}


def pair_metrics(lookup, ids):
    outcome_ids = {"on_wins": [], "on_losses": [], "ties": []}
    for index in ids:
        sid = f"multi_turn_long_context_{index}"
        off, on = lookup.get((sid, "off")), lookup.get((sid, "on"))
        if off is None or on is None:
            continue
        a, b = off["official_grade"]["valid"], on["official_grade"]["valid"]
        outcome_ids["ties" if a == b else "on_wins" if b else "on_losses"].append(index)
    return {**{k: len(v) for k, v in outcome_ids.items()}, "sample_ids": outcome_ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    plan = json.loads((root / "plan.json").read_text())
    reference_calls = {}
    source_manifest = root / "bfcl_source_manifest.json"
    if source_manifest.exists():
        for entry in json.loads(source_manifest.read_text())["files"]:
            if entry["path"].endswith("/possible_answer/BFCL_v4_multi_turn_long_context.json"):
                source_path = Path(entry["path"])
                assert hashlib.sha256(source_path.read_bytes()).hexdigest() == entry["sha256"]
                reference_calls = {r["id"]: r["ground_truth"] for r in read_jsonl(source_path)}
    results = []
    paths = {}
    failures = []
    paused_attempts = []
    for path in sorted(root.glob("sample_*/*/result.json")):
        result = json.loads(path.read_text())
        if result["status"] != "completed":
            if result["status"] == "paused":
                paused_attempts.append(str(path.parent))
                continue
            failures.append({"path": str(path), "error": result.get("error")})
            continue
        key = result["id"], result["group"]
        assert key not in paths, f"Multiple completed attempts for {key}"
        paths[key] = path.parent
        results.append(result)
    groups = {}
    for group in plan["groups"]:
        rows = [r for r in results if r["group"] == group]
        groups[group] = group_metrics(rows)
    per_sample, per_turn, comparisons = [], [], []
    failure_analysis = []
    lookup = {(r["id"], r["group"]): r for r in results}
    wins = losses = ties = 0
    for index in plan["ids"]:
        sid = f"multi_turn_long_context_{index}"
        for group in plan["groups"]:
            r = lookup.get((sid, group))
            if not r:
                continue
            commits = [c for t in r["turns"] for s in t["steps"] for c in s["query_history_commits"]]
            rounds = [int(re.search(r"committed_round=(\d+)", c).group(1)) for c in commits]
            per_sample.append({"id": sid, "group": group, "official_pass": r["official_grade"]["valid"],
                "official_first_error_type": r["official_grade"].get("error_type", ""),
                "query_count": r["query_count"], "diagnostic_turns_passed": r["local_conditions_passed"],
                "requests": r["requests"], "completion_tokens": r["completion_tokens"],
                "max_prompt_tokens": r["max_prompt_tokens"], "tool_calls": r["tool_calls"],
                "tool_execution_errors": r["tool_execution_errors"],
                "query_history_commits": len(commits), "max_committed_round": max(rounds, default=0),
                "committed_rounds": rounds, "wall_seconds_including_startup": r["wall_seconds_including_startup"]})
            if not r["official_grade"]["valid"]:
                failed_turn = next((t for t in r["turns"] if not t["local_conditions_valid"]), None)
                failure_kind = "diagnostic_condition"
                if failed_turn is None:
                    failed_turn = next((t for t in r["turns"]
                        if t.get("context_limit_reached") or t["step_limit_reached"]), None)
                    failure_kind = "context_limit" if failed_turn and failed_turn.get("context_limit_reached") else "tool_step_limit"
                if failed_turn is not None:
                    decoded = json.loads((paths[sid, group] / "decoded_calls.json").read_text())
                    failure_analysis.append({"id": sid, "group": group,
                        "first_failed_user_turn_index": failed_turn["turn_index"],
                        "failure_before_second_distinct_user_query": failed_turn["turn_index"] == 0,
                        "failure_kind": failure_kind,
                        "query": failed_turn["query"],
                        "actual_calls_in_first_failed_turn": decoded[failed_turn["turn_index"]],
                        "reference_calls_in_first_failed_turn": reference_calls.get(sid, [[]] * r["query_count"])[failed_turn["turn_index"]],
                        "state_check": failed_turn["state_check"], "response_check": failed_turn["response_check"]})
            for t in r["turns"]:
                first = t["steps"][0]
                per_turn.append({"id": sid, "group": group, "turn_index": t["turn_index"],
                    "query": t["query"], "actual_prompt_tokens_before_answer": first["usage"]["prompt_tokens"],
                    "state_valid": t["state_check"]["valid"], "response_valid": t["response_check"]["valid"],
                    "diagnostic_conditions_valid": t["local_conditions_valid"],
                    "requests": len(t["steps"]), "step_limit_reached": t["step_limit_reached"],
                    "context_limit_reached": t.get("context_limit_reached", False),
                    "tool_execution_errors": t["tool_execution_errors"]})
        off, on = lookup.get((sid, "off")), lookup.get((sid, "on"))
        if not off or not on:
            continue
        a, b = off["official_grade"]["valid"], on["official_grade"]["valid"]
        wins += int(b and not a)
        losses += int(a and not b)
        ties += int(a == b)
        request_maps = {}
        selection_maps = {}
        for group in ("off", "on"):
            directory = paths[sid, group]
            request_maps[group] = {(r["turn_index"], r["step"]): r for r in read_jsonl(directory / "requests.jsonl")}
            selection_maps[group] = selections(directory / "retrieval.jsonl")
        for key, a_req in request_maps["off"].items():
            b_req = request_maps["on"].get(key)
            if not b_req or a_req["input_digest"] != b_req["input_digest"]:
                continue
            a_sel = selection_maps["off"].get(a_req["label"])
            b_sel = selection_maps["on"].get(b_req["label"])
            a_events, b_events = a_req["selection_events"], b_req["selection_events"]
            a_hashes = [m.group(1) for event in a_events
                        if (m := re.search(r"selected_hash=([a-f0-9]+)", event))]
            b_hashes = [m.group(1) for event in b_events
                        if (m := re.search(r"selected_hash=([a-f0-9]+)", event))]
            a_hash = a_hashes[-1] if a_hashes else None
            b_hash = b_hashes[-1] if b_hashes else None
            a_set = set(map(tuple, a_sel["selected"])) if a_sel else None
            b_set = set(map(tuple, b_sel["selected"])) if b_sel else None
            a_message = a_req["response"]["choices"][0]["message"]
            b_message = b_req["response"]["choices"][0]["message"]
            decision_changed = input_identity([{"role": "assistant", "tool_calls": a_message.get("tool_calls", [])}], []) != \
                input_identity([{"role": "assistant", "tool_calls": b_message.get("tool_calls", [])}], [])
            comparisons.append({"id": sid, "turn_index": key[0], "step": key[1],
                "input_digest": a_req["input_digest"], "prompt_tokens": a_req["usage"]["prompt_tokens"],
                "off_last_selection_hash": a_hash, "on_last_selection_hash": b_hash,
                "both_selection_hashes_present": a_hash is not None and b_hash is not None,
                "selection_hash_changed": a_hash != b_hash if a_hash is not None and b_hash is not None else None,
                "both_block_snapshots_present": a_set is not None and b_set is not None,
                "selected_block_sets_changed": a_set != b_set if a_set is not None and b_set is not None else None,
                "tool_call_decision_changed": decision_changed,
                "only_off_selected_blocks": sorted(a_set - b_set) if a_set is not None and b_set is not None else [],
                "only_on_selected_blocks": sorted(b_set - a_set) if a_set is not None and b_set is not None else [],
                "off_response_digest": input_identity([a_req["response"]["choices"][0]["message"]], []),
                "on_response_digest": input_identity([b_req["response"]["choices"][0]["message"]], [])})
    summary = {"planned_conversations_per_group": len(plan["ids"]),
        "planned_user_queries_per_group": sum(r["user_query_count"] for r in plan["samples"]), "groups": groups,
        "paired_outcomes": pair_metrics(lookup, plan["ids"]),
        "matched_input_requests": len(comparisons),
        "matched_input_requests_with_both_hashes": sum(c["both_selection_hashes_present"] for c in comparisons),
        "matched_input_requests_with_different_hashes": sum(c["selection_hash_changed"] is True for c in comparisons),
        "matched_input_requests_with_different_block_snapshots": sum(c["selected_block_sets_changed"] is True for c in comparisons),
        "matched_input_requests_with_different_tool_call_decisions": sum(c["tool_call_decision_changed"] for c in comparisons),
        "infrastructure_errors": failures,
        "paused_attempts": paused_attempts,
        "all_planned_episodes_completed": all(groups[g]["completed_conversations"] == len(plan["ids"]) for g in plan["groups"])}
    pause_path = root / "pause.json"
    paused = pause_path.exists() and json.loads(pause_path.read_text()).get("resume_requires_user_request", False)
    summary["paused_by_user"] = paused
    context_cases = []
    for r in results:
        for t in r["turns"]:
            if not t.get("context_limit_reached"):
                continue
            match = re.search(r"prompt_tokens=(\d+)", t["context_limit_error"])
            calls = json.loads((paths[r["id"], r["group"]] / "decoded_calls.json").read_text())
            context_cases.append({"id": r["id"], "group": r["group"], "turn_index": t["turn_index"],
                "query": t["query"], "accepted_requests_in_turn": len(t["steps"]),
                "rejected_prompt_tokens": int(match.group(1)) if match else None,
                "official_pass": r["official_grade"]["valid"], "actual_calls_in_turn": calls[t["turn_index"]]})
    summary["context_limit_cases"] = context_cases
    reuse_path = root / "reuse_manifest.json"
    if reuse_path.exists():
        reuse = json.loads(reuse_path.read_text())
        summary["cohorts"] = {}
        for name, ids in (("reused", reuse["reused_ids"]), ("additional", reuse["new_ids"])):
            sids = {f"multi_turn_long_context_{i}" for i in ids}
            summary["cohorts"][name] = {"ids": ids,
                "groups": {g: group_metrics([r for r in results if r["group"] == g and r["id"] in sids])
                           for g in plan["groups"]},
                "paired_outcomes": pair_metrics(lookup, ids)}
    summary["first_turn_failures_by_group"] = {g: sum(f["group"] == g and f["failure_before_second_distinct_user_query"]
        for f in failure_analysis) for g in plan["groups"]}
    if (root / "validation.json").exists():
        summary["validation"] = json.loads((root / "validation.json").read_text())
    (root / "failure_analysis.json").write_text(json.dumps(failure_analysis, ensure_ascii=False, indent=2) + "\n")
    (root / "context_limit_cases.json").write_text(json.dumps(context_cases, ensure_ascii=False, indent=2) + "\n")
    for name, records in [("per_sample", per_sample), ("per_turn", per_turn), ("matched_input_selection_comparison", comparisons)]:
        csv_file(root / f"{name}.csv", records)
        (root / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    diagnostic_path = root / "changed_blocks_diagnostics.json"
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text())
        if diagnostic.get("comparison_sha256") == hashlib.sha256((root / "matched_input_selection_comparison.jsonl").read_bytes()).hexdigest():
            summary["changed_block_diagnostics"] = diagnostic
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if "off" in groups and "on" in groups:
        off_group, on_group = groups["off"], groups["on"]
        completed = summary["all_planned_episodes_completed"]
        if paused:
            pairs_done = summary["paired_outcomes"]["on_wins"] + summary["paired_outcomes"]["on_losses"] + summary["paired_outcomes"]["ties"]
            paired = [i for i in plan["ids"] if all((f"multi_turn_long_context_{i}", g) in lookup for g in ("off", "on"))]
            paired_passes = {g: sum(lookup[f"multi_turn_long_context_{i}", g]["official_grade"]["valid"] for i in paired) for g in ("off", "on")}
            conclusion = (f"用户要求暂停，推理进程已停止。已完成 {pairs_done} 对中，关闭通过 {paired_passes['off']} 条，开启通过 {paired_passes['on']} 条；"
                "单边已完成的样本不参与配对结论，下表仅列已保存结果。")
        elif not completed:
            conclusion = "实验进行中，以下只统计已经完成的会话，不能作为最终对照结论。"
        elif on_group["official_conversation_passes"] == off_group["official_conversation_passes"]:
            conclusion = "本轮开启与关闭的整条会话通过数相同，未观察到任务通过率提升。"
        else:
            conclusion = (f"本轮整条会话通过数：关闭 {off_group['official_conversation_passes']}，"
                          f"开启 {on_group['official_conversation_passes']}。这是小样本结果。")
        sample_lines = []
        for index in plan["ids"]:
            sid = f"multi_turn_long_context_{index}"
            a, b = lookup.get((sid, "off")), lookup.get((sid, "on"))
            if not a or not b:
                continue
            sample_lines.append(f"| {index} | {a['query_count']} | {'通过' if a['official_grade']['valid'] else '失败'} | "
                f"{'通过' if b['official_grade']['valid'] else '失败'} | {a['local_conditions_passed']} / {b['local_conditions_passed']} | "
                f"{a['max_prompt_tokens']:,} / {b['max_prompt_tokens']:,} |")
        metrics = [
            ("已完成会话", "completed_conversations"),
            ("官方整条会话通过数", "official_conversation_passes"),
            ("已进入并完成子检查的用户 Query", "turns_completed"),
            ("逐轮状态和返回检查同时通过（诊断）", "diagnostic_turn_conditions_passed"),
            ("逐轮工具返回检查通过（诊断）", "diagnostic_turn_response_checks_passed"),
            ("输入已超过 32K 的后续 Query", "later_queries_starting_over_32k"),
            ("其中状态和返回检查同时通过（诊断）", "diagnostic_conditions_passed_on_later_queries_over_32k"),
            ("推理请求数（包括工具返回后继续生成）", "total_requests"),
            ("生成 tokens（包括 thinking）", "completion_tokens"),
            ("模拟工具调用次数", "tool_calls"),
            ("工具显式错误返回（执行异常或 error 字段）", "tool_execution_errors"),
            ("触及输出上限的请求", "requests_finishing_at_length_limit"),
            ("耗尽 128K 逻辑上下文的会话", "conversations_exhausting_context_limit"),
            ("Query 历史提交次数", "query_history_commits"),
            ("其中有正分数的提交", "positive_query_history_commits")]
        metric_lines = [f"| {label} | {off_group[key]} | {on_group[key]} |" for label, key in metrics]
        cohort_note = ""
        cohort_timing = ""
        if "cohorts" in summary:
            old, extra = summary["cohorts"]["reused"], summary["cohorts"]["additional"]
            cohort_note = (f"此前 {len(old['ids'])} 条直接复用原始结果，本次新增 {len(extra['ids'])} 条；"
                "原始记录逐文件核对摘要后复制，未重跑或挑选答案。新增名单是原审计候选池的全部剩余样本。\n\n"
                "| 批次 | 关闭：通过 / 完成会话 | 开启：通过 / 完成会话 | 开启胜 / 负 / 平 |\n"
                "|---|---:|---:|---:|\n")
            for label, cohort in (("复用上一轮", old), ("本次新增", extra)):
                a, b, p = cohort['groups']['off'], cohort['groups']['on'], cohort['paired_outcomes']
                cohort_note += (f"| {label} | {a['official_conversation_passes']} / {a['completed_conversations']} | "
                    f"{b['official_conversation_passes']} / {b['completed_conversations']} | "
                    f"{p['on_wins']} / {p['on_losses']} / {p['ties']} |\n")
            elapsed = sum(g['episode_wall_seconds_sum'] for g in extra['groups'].values())
            cohort_timing = (f"本次新增样本的会话耗时合计 {elapsed / 60:.2f} 分钟，"
                "下列总体耗时还包含复用的上一轮历史运行时间，不等于本次补测用时。\n\n")
        validation_note = (f"从保存调用独立复现全部 {len(results)} 个官方评分，并核验历史轮次、32K 选块预算、"
            "token 计数、源文件摘要和两组参数。" if summary.get("validation", {}).get("passed")
            else "完成后独立复核；当前尚未生成最终校验结果。")
        diagnostic_notes = []
        if "changed_block_diagnostics" in summary:
            d = summary["changed_block_diagnostics"]
            diagnostic_notes.append(f"匹配输入的选块差集中，共 {d['changed_block_occurrences']} 个块次，"
                f"其中 {d['numeric_array_fragment_occurrences']} 个呈数字数组片段形态。这里按每次快照中的块出现计数，"
                "不是唯一块数；文本形态不能证明该块无用，也未覆盖每个中间压力窗口。详见 `changed_blocks_diagnostics.json` 和 `changed_block_texts.jsonl`。")
        if (root / "sample_109/same_input_price_and_changed_blocks_diagnostic.json").exists():
            diagnostic_notes.append("已复核的错误例子：109 两组在第三问都以 310.00 下单，工具返回的正确值为 310.23；该次下单前的选块快照中，两组都保留了正确价格，被替换的 18 个块均为数字数组片段。这说明该检查点的错误不能简单归因为正确价格 token 被丢弃，也不能由此推断所有中间窗口或数值状态都正确。详细证据见 `sample_109/same_input_price_and_changed_blocks_diagnostic.json`。")
        if (root / "sample_110/login_workflow_diagnostic.json").exists():
            diagnostic_notes.append("次要行为差异：110 的最后一问在相同输入下，关闭组先建工单、收到未登录错误、再登录重试；开启组先查登录状态、登录、再建工单，两边都是三次工具调用。该处开启组避免了一次显式错误返回，但该会话两组此前都因下单价格错误而失败，不能据此声称整条会话通过率提升。见 `sample_110/login_workflow_diagnostic.json`。")
        if (root / "sample_131/receiver_workflow_diagnostic.json").exists():
            diagnostic_notes.append("另一处局部改善：131 的第六问在相同输入下，关闭组直接把 Alice 当作 receiver_id 发送，报错后查询用户 ID 再发送；开启组先查 ID 再发送，少一次失败工具调用和一次推理请求。两组此前已在交易步骤失败，整条会话得分未因此改变。见 `sample_131/receiver_workflow_diagnostic.json`。")
        report = f"""# BFCL 32K：Query 累计分数开启 / 关闭

状态：{'已暂停' if paused else '全部完成' if completed else '进行中'}。{conclusion}

本轮为固定 {len(plan['ids'])} 条 BFCL 长上下文会话的闭环对照，名单在模型运行前确定。候选都来自交易场景，不代表完整 BFCL。每条会话分别在两个开关状态下完整运行，工具返回由模型实际调用产生，参考调用仅用于评分。

每组计划 {summary['planned_user_queries_per_group']} 个用户 Query；提前耗尽上下文或工具步数时，剩余未执行的问题仍影响整条会话官方评分，但不计入已进入问题的逐轮诊断分母。

{cohort_note}

## 结果

| 指标 | 关闭 | 开启 |
|---|---:|---:|
{chr(10).join(metric_lines)}

配对结果：开启胜 {wins} 条、负 {losses} 条、持平 {ties} 条。

“整条会话通过”使用固定版本 BFCL 官方 `multi_turn_checker`，检查每轮工具状态与所需工具返回。它不评价最终自然语言回答。“逐轮诊断”调用官方状态和返回子检查，允许后轮恢复，不是另一项官方准确率。前轮下单参数错误会影响后轮环境状态，因此不能把所有后续状态失败都归因为新的记忆错误。

| 样本 ID 后缀 | Query 数 | 关闭：整条会话 | 开启：整条会话 | 逐轮同时通过：关闭 / 开启 | 实际最大输入 tokens：关闭 / 开启 |
|---:|---:|---|---|---|---|
{chr(10).join(sample_lines)}

## 功能是否实际触发

两组找到 {summary['matched_input_requests']} 次内容相同的输入请求。忽略不进入模型提示词的随机工具调用 ID，并将工具参数 JSON 键顺序规范化后比较输入；不将不同模型轨迹的输入强行配对。

其中 {summary['matched_input_requests_with_both_hashes']} 次同时有选块哈希，{summary['matched_input_requests_with_different_hashes']} 次哈希不同；完整块快照中，{summary['matched_input_requests_with_different_block_snapshots']} 次选块集合不同。具体差集见 `matched_input_selection_comparison.csv`。这些是累计开关影响保留内容的证据，不等于任务准确率收益。

相同输入下，工具调用名称、参数或是否调用工具发生改变的请求共 {summary['matched_input_requests_with_different_tool_call_decisions']} 次；仅自然语言措辞或 thinking 改变不计入这个数，也不据此认定调用正确与否。

{(chr(10) * 2).join(diagnostic_notes)}

累计开关按每次逻辑输入请求更新，工具返回后的继续请求也会更新。用户 Query 数与历史提交次数因此不同。本实验评估现有功能整体，没有单独隔离不同用户问题之间、与同一问题内多次工具交互之间的贡献。

## 参数与执行条件

以之前 DSH action-pressure 测试的 `runtime/dsh_tool_guided_actionpressure50_20261010/launch.template.json` 为基础，两组共享以下配置：

| 参数 | 本轮配置 |
|---|---|
| 模型 | Qwen3.8-27B Q8_0；同一冻结二进制和权重文件 |
| 草稿模型 | DFlash2 BF16 |
| KV dtype / 块大小 | FP8 / 32 tokens |
| 检索预算 / prefill 预算 | 32,768 / 32,768 tokens |
| 生成预留 / thinking 上限 | 16,384 / 4,096 tokens |
| 单次请求输出上限 | {plan['max_tokens']} tokens，包含 thinking |
| Query 相关性 | mean-k、TopK、原始最后用户问题 |
| 原有 attention 融合 | quota，50% Query / 50% action attention，衰减 0.5，间隔 16 |
| 容量回收 attention / 输入 attention | 开启 / 关闭 |
| sink / recent | 1,312 / 0 tokens |
| 硬保留部分 | system、root、query、policy；tool-call/tool-result 项关闭，仍有执行器必要的尾部边界块 |
| 语义扩展 / Query 重算 | 关闭 / 关闭 |
| guided / continuous batching | 关闭 / 关闭 |
| prefill chunk / 逻辑上下文上限 | 1,024 / 131,072 tokens |
| CPU tier / NVMe | 64 GiB / 关闭 |
| temperature / seed | 0 / {plan['seed']} |
| 对照变量 | `--kvmem-query-history off` 与 `on`；开启衰减 0.5 |

32K 是检索窗口预算，16K 是额外生成空间预留。每次记录的 `usage.prompt_tokens` 是完整逻辑输入长度，不能当成当时全部驻留的 KV tokens；选块快照用于检查实际保留窗口。

相较旧测试，检索预算从 36K 改为用户要求的 32K；guided boundary 与当前累计实现不兼容，所以两组都关闭 guided；temperature 固定为 0；上下文上限设为足以容纳本批输入的 128K。因而本轮可比较开启/关闭两组，不用于直接比较旧 DSH 分数。

每个样本使用新服务进程，清空跨样本的 Query 与 action 历史；配对顺序交替为 off→on、on→off。跨轮使用完整消息历史和稳定 thinking 包装。工具仅执行 BFCL 本地模拟器，`long_context=True`；没有将 `initial_config`、参考答案或诊断状态注入模型输入。

本轮使用 qw3 服务端的工具包装和标准化的 OpenAI 工具 schema，实际输入长度以服务端 usage 为准；它与前次静态审计使用的 HF chat template、参考调用轨迹长度不同。

触及统一输出上限的自然语言回答不保证完整；官方评分仍只检查已执行工具状态和返回，不能把工具任务通过率当作自然语言回答完整率。

模型自行重复调用导致完整消息输入超过统一的 131,072-token 上限时，该会话因额度耗尽而终止，保留已执行调用并由官方检查器评分，不重跑抽取答案。此类会话未执行的后续用户 Query 不纳入已进入问题的逐轮诊断分母。详见 `context_limit.json`；若运行期间修正了此项记账处理，版本与原始错误保存在 `protocol_amendment.json` 和对应会话目录。

## 时间

{cohort_timing}

关闭组推理请求耗时合计 {off_group['request_latency_seconds_sum']:.2f} 秒，开启组 {on_group['request_latency_seconds_sum']:.2f} 秒；包含服务启动的会话耗时分别为 {off_group['episode_wall_seconds_sum']:.2f} 秒和 {on_group['episode_wall_seconds_sum']:.2f} 秒。输出内容、请求次数和加载缓存可能不同，这些实测值不能直接解释为累计算法的纯开销或加速。

## 明细与限制

- `per_sample.csv`：逐样本结果、长度、耗时及历史提交轮次。
- `per_turn.csv`：全部用户 Query、实际输入长度、逐轮状态/返回检查。
- `matched_input_selection_comparison.csv`：相同输入的选块差集。
- `failure_analysis.json`：首个未通过子检查的用户轮，以及实际调用与参考调用。
- `context_limit_cases.json`：耗尽逻辑上下文的会话、所在用户轮和已执行调用；终止前未通过子检查的情况另见失败分析。
- `sample_<id>/<off|on>/`：启动参数、完整请求与返回、解码调用、官方评分、选块快照、服务日志。
- `plan.json`、`pre_run_selection.json`：运行前名单和协议；`model_identity.json`、`build_manifest.json`、`bfcl_source_manifest.json`：冻结版本和 SHA256。
- `validation.json`：{validation_note}

没有根据失败分数重跑挑选答案。基础设施错误保留独立尝试，本次记录 {len(failures)} 次。{len(plan['ids'])} 个定向选择的样本、单一场景和单一 seed 只能说明本配置下的观察，不能据此判断广泛任务上的稳定收益。

重新运行可使用仓库中的 `scripts/kvmem_eval/run_bfcl_query_history_ab.py`，传入 `plan.json` 中冻结的二进制路径、`launch_template_snapshot.json`、`--ids` 的完整名单和一个新的输出目录。已完成目录用 `--resume` 会跳过已完成样本，不会自动重抽失败答案。汇总入口为 `scripts/kvmem_eval/summarize_bfcl_query_history_ab.py --output-dir <输出目录>`；独立评分复核入口为 `scripts/kvmem_eval/verify_bfcl_query_history_ab.py --output-dir <输出目录>`。原始源数据和模拟器已固定保存在前次审计目录中，不需再次下载数据。
"""
        (root / "report.md").write_text(report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
