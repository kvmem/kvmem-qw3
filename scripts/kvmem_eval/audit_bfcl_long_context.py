#!/usr/bin/env python3
"""Audit BFCL long-context queries and tokenize an official reference trajectory.

No model inference is performed. Tool results are obtained by executing the
pinned BFCL simulator and its published reference calls, with long_context=True.
The resulting lengths describe that reference path, not a model rollout, and
exclude generated reasoning and natural-language assistant answers.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_query(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def distribution(values):
    values = sorted(values)
    def percentile(p):
        pos = (len(values) - 1) * p
        lo = int(pos)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.fmean(values), "p90": percentile(.9),
            "p95": percentile(.95), "max": max(values)}


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                         else v for k, v in row.items()} for row in records)


def parse_reference_call(text, allowed_names):
    node = ast.parse(text, mode="eval").body
    # All published reference calls must be a named simulator method and
    # literal arguments. Never execute arbitrary Python from a dataset.
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"Non-literal reference call: {text}")
    if node.func.id not in allowed_names:
        raise ValueError(f"Unknown simulator function: {node.func.id}")
    if any(k.arg is None for k in node.keywords):
        raise ValueError("Expanded keyword arguments are not allowed")
    args = [ast.literal_eval(arg) for arg in node.args]
    kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    return node.func.id, args, kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    upstream = args.source_root / "upstream"
    sys.path.insert(0, str(upstream.resolve()))
    from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    def render(messages, tools):
        return tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=False, preserve_thinking=False)
    def prompt_count(messages, tools):
        return count(render(messages, tools))

    source_file = args.source_root / "BFCL_v4_multi_turn_long_context.json"
    rows = read_jsonl(source_file)
    gold = {r["id"]: r for r in read_jsonl(upstream / "bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_long_context.json")}
    assert len(rows) == len(gold) == 200
    if args.limit:
        rows = rows[:args.limit]
    conversations, turns, pairs = [], [], []
    reference_path = args.output_dir / "reference_trajectories.jsonl"
    markers = re.compile(r"\b(it|its|them|those|that|these|previous|previously|earlier|above|same|again|also|instead|remaining|rest|other|another|finally)\b", re.I)
    quoted = re.compile(r"['\"]([^'\"\n]{2,80})['\"]")
    with reference_path.open("w", encoding="utf-8") as reference_file:
        for index, row in enumerate(rows):
            sid = row["id"]
            query_turns = row["question"]
            query_texts = ["\n".join(m["content"] for m in turn if m["role"] == "user") for turn in query_turns]
            assert all(len(t) == 1 and t[0]["role"] == "user" for t in query_turns)
            assert len(query_turns) == len(gold[sid]["ground_truth"])
            functions = []
            for klass in row["involved_classes"]:
                functions.extend(read_jsonl(upstream / "bfcl_eval/data/multi_turn_func_doc" / MULTI_TURN_FUNC_DOC_FILE_MAPPING[klass]))
            assert len({f["name"] for f in functions}) == len(functions)
            schemas = {f["name"]: f for f in functions}
            tools = [{"type": "function", "function": f} for f in functions]
            messages = []
            # This obtains the SAME simulator instances used for every later
            # turn, including seeded distractor state from long_context.py.
            _, instances = execute_multi_turn_func_call([], row["initial_config"],
                row["involved_classes"], "qw3_static_audit", sid, long_context=True)
            method_owners = {name: instance for instance in instances.values()
                             for name in dir(instance) if not name.startswith("_")
                             and callable(getattr(instance, name))}
            events = []
            call_records = []
            tool_errors = []
            per_turn = []
            last_literals = set()
            all_earlier_literals = set()
            for ti, (text, ref_calls) in enumerate(zip(query_texts, gold[sid]["ground_truth"])):
                messages.extend(query_turns[ti])
                before = prompt_count(messages, tools)
                query_tokens = count(text)
                turn_calls = []
                literals = set()
                output_tokens = 0
                for ci, ref_call in enumerate(ref_calls):
                    name, positionals, kwargs = parse_reference_call(ref_call, schemas)
                    # Use Python's actual signature to check positional mapping;
                    # schema order can differ from simulator signature order.
                    sig = inspect.signature(getattr(method_owners[name], name))
                    bound = sig.bind(*positionals, **kwargs)
                    function_args = dict(bound.arguments)
                    for value in ast.walk(ast.parse(ref_call, mode="eval")):
                        if isinstance(value, ast.Constant) and isinstance(value.value, str) and len(value.value.strip()) >= 3:
                            literals.add(value.value.strip())
                    results, _ = execute_multi_turn_func_call([ref_call], row["initial_config"],
                        row["involved_classes"], "qw3_static_audit", sid, long_context=True)
                    assert len(results) == 1
                    result = results[0]
                    # The simulator's official executor stringifies outputs;
                    # retain it verbatim instead of reserializing JSON.
                    messages.append({"role": "assistant", "content": "", "tool_calls": [
                        {"type": "function", "function": {"name": name, "arguments": function_args}}]})
                    messages.append({"role": "tool", "content": result})
                    after = prompt_count(messages, tools)
                    result_tokens = count(result)
                    output_tokens += result_tokens
                    record = {"turn_index": ti, "call_index": ci, "reference_call": ref_call,
                              "tool_result_tokens": result_tokens, "prompt_tokens_after_tool": after}
                    call_records.append(record)
                    turn_calls.append(name)
                    events.append({"turn_index": ti, "phase": "after_tool", "tokens": after})
                    try:
                        parsed = json.loads(result)
                    except (ValueError, TypeError):
                        parsed = None
                    if result.startswith("Error during execution: ") or (isinstance(parsed, dict) and "error" in parsed):
                        tool_errors.append({**record, "result": result})
                end = prompt_count(messages, tools)
                events.append({"turn_index": ti, "phase": "query_start", "tokens": before})
                tr = {"id": sid, "turn_index": ti, "query": text, "query_tokens": query_tokens,
                      "reference_prompt_tokens_before_query_answer": before,
                      "reference_prompt_tokens_after_turn_tools": end,
                      "reference_tool_calls": len(ref_calls), "reference_tools": turn_calls,
                      "reference_tool_output_tokens": output_tokens}
                turns.append(tr)
                per_turn.append(tr)
                if ti:
                    current_quotes = set(quoted.findall(text))
                    previous_quotes = set(quoted.findall(query_texts[ti - 1]))
                    pair = {"id": sid, "from_turn": ti - 1, "to_turn": ti,
                            "previous_query": query_texts[ti - 1], "current_query": text,
                            "explicit_reference_markers": markers.findall(text),
                            "shared_quoted_literals": sorted(current_quotes & previous_quotes),
                            "shared_reference_argument_strings": sorted(literals & last_literals),
                            "reused_reference_argument_strings_from_any_earlier_turn": sorted(literals & all_earlier_literals)}
                    pair["has_rule_based_relation_evidence"] = bool(pair["explicit_reference_markers"] or
                        pair["shared_quoted_literals"] or pair["shared_reference_argument_strings"])
                    pairs.append(pair)
                last_literals = literals
                all_earlier_literals.update(literals)

            full_tokens = prompt_count(messages, tools)
            basic = {"id": sid, "user_query_count": len(query_texts),
                "distinct_user_queries_exact": len(set(query_texts)),
                "distinct_user_queries_nfc_whitespace": len(set(map(normalize_query, query_texts))),
                "total_user_query_tokens": sum(x["query_tokens"] for x in per_turn),
                "user_only_chat_tokens": prompt_count([m for t in query_turns for m in t], []),
                "tool_schema_json_tokens": count(json.dumps(tools, ensure_ascii=False)),
                "initial_config_json_tokens_not_model_input": count(json.dumps(row["initial_config"], ensure_ascii=False)),
                "first_query_prompt_tokens": per_turn[0]["reference_prompt_tokens_before_query_answer"],
                "reference_full_prompt_tokens": full_tokens,
                "reference_max_prompt_tokens": max(e["tokens"] for e in events),
                "reference_total_tool_output_tokens": sum(x["reference_tool_output_tokens"] for x in per_turn),
                "reference_tool_call_count": len(call_records), "reference_tool_error_count": len(tool_errors),
                "involved_classes": row["involved_classes"],
                "has_rule_based_relation_evidence": any(p["has_rule_based_relation_evidence"] for p in pairs if p["id"] == sid),
                "queries": query_texts}
            for threshold in [4096,8192,16384,32768,65536,131072]:
                over = [t["turn_index"] for t in per_turn if t["reference_prompt_tokens_before_query_answer"] > threshold]
                basic[f"reference_query_starts_over_{threshold}"] = len(over)
                basic[f"reference_max_over_{threshold}"] = basic["reference_max_prompt_tokens"] > threshold
            conversations.append(basic)
            reference_file.write(json.dumps({"id": sid, "messages": messages,
                "tool_calls": call_records, "tool_errors": tool_errors}, ensure_ascii=False) + "\n")
            if index % 25 == 0:
                print(f"processed {index + 1}/{len(rows)}", flush=True)
            if index == 0:
                (args.output_dir / "example_reference_prompt.txt").write_text(render(messages, tools))

    summary = {"schema_version": 1, "source_commit": json.loads((args.source_root / "upstream_commit.json").read_text())["sha"],
        "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "tokenizer_path": str(args.tokenizer.resolve()),
        "tokenizer_sha256": hashlib.sha256((args.tokenizer / "tokenizer.json").read_bytes()).hexdigest(),
        "tokenizer_config_sha256": hashlib.sha256((args.tokenizer / "tokenizer_config.json").read_bytes()).hexdigest(),
        "conversation_count": len(conversations), "total_user_queries": len(turns),
        "query_count_histogram": dict(sorted(Counter(c["user_query_count"] for c in conversations).items())),
        "distinct_query_count_histogram": dict(sorted(Counter(c["distinct_user_queries_nfc_whitespace"] for c in conversations).items())),
        "distributions": {field: distribution([c[field] for c in conversations]) for field in [
            "user_query_count", "distinct_user_queries_nfc_whitespace", "total_user_query_tokens", "user_only_chat_tokens",
            "first_query_prompt_tokens", "reference_full_prompt_tokens", "reference_max_prompt_tokens", "reference_tool_call_count"]},
        "individual_query_tokens": distribution([t["query_tokens"] for t in turns]),
        "reference_tool_errors": sum(c["reference_tool_error_count"] for c in conversations),
        "adjacent_pairs": len(pairs),
        "pairs_with_rule_based_relation_evidence": sum(p["has_rule_based_relation_evidence"] for p in pairs),
        "pressure": {str(n): {"sessions_exceeding": sum(c[f"reference_max_over_{n}"] for c in conversations),
            "sessions_with_later_user_query_over_budget": sum(c[f"reference_query_starts_over_{n}"] > 0 for c in conversations),
            "total_user_query_starts_over_budget": sum(c[f"reference_query_starts_over_{n}"] for c in conversations)}
            for n in [4096,8192,16384,32768,65536,131072]},
        "length_definition": "Local Qwen3.8 template, thinking disabled, official function schemas; ordered reference calls executed with long_context=True. One assistant tool-call per reference call; no generated reasoning, answer text, exploration, or retries. Not a model rollout and not a strict bound.",
        "relation_definition": "Automatic surface/reference-argument evidence only, not semantic ground truth; see separate analyst review annotations."}
    for name, records in [("conversations", conversations), ("turns", turns), ("adjacent_query_relations", pairs)]:
        write_csv(args.output_dir / f"{name}.csv", records)
        (args.output_dir / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
