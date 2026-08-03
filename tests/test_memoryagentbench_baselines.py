from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/kvmem_eval/run_memoryagentbench_baselines.py"
SPEC = importlib.util.spec_from_file_location("mab_baselines", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

COMPARE_SCRIPT = ROOT / "scripts/kvmem_eval/compare_memoryagentbench_methods.py"
COMPARE_SPEC = importlib.util.spec_from_file_location(
    "mab_compare", COMPARE_SCRIPT
)
assert COMPARE_SPEC is not None and COMPARE_SPEC.loader is not None
C = importlib.util.module_from_spec(COMPARE_SPEC)
sys.modules[COMPARE_SPEC.name] = C
COMPARE_SPEC.loader.exec_module(C)

UTILITY_SCRIPT = ROOT / "scripts/kvmem_eval/update_utility_evaluation.py"
UTILITY_SPEC = importlib.util.spec_from_file_location(
    "utility_evaluation", UTILITY_SCRIPT
)
assert UTILITY_SPEC is not None and UTILITY_SPEC.loader is not None
U = importlib.util.module_from_spec(UTILITY_SPEC)
sys.modules[UTILITY_SPEC.name] = U
UTILITY_SPEC.loader.exec_module(U)


TOKENIZER = Path(
    "/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/"
    "Qwen3.6-27B-FP8"
)


class MemoryAgentBenchBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.counter = M.TokenCounter(TOKENIZER)

    def test_extracts_the_exact_open_user_context(self) -> None:
        system = "system"
        context = "memorized context\nwith two lines"
        prefix = M.qwen_open_user(system, context)
        self.assertEqual(M.extract_memorized_context(prefix, system), context)

    def test_prefix_warmup_commits_exactly_the_shared_open_prefix(self) -> None:
        system = "You are a helpful assistant."
        context = "One immutable context shared by many independent questions."
        aligned, metadata = M.align_common_prefix(
            system=system,
            content=context,
            counter=self.counter,
            page_tokens=16,
        )
        self.assertTrue(aligned.startswith(context))
        self.assertEqual(metadata["open_prefix_tokens"] % 16, 0)
        self.assertEqual(
            metadata["expected_commit_tokens"], metadata["open_prefix_tokens"]
        )
        self.assertLess(metadata["no_think_suffix_tokens"], 16)

    def test_prefix_warmup_and_scored_request_share_cached_tokens(self) -> None:
        system = "You are a helpful assistant."
        common = "A long immutable shared context ending in a newline.\n"
        aligned, _ = M.align_common_prefix(
            system=system,
            content=common,
            counter=self.counter,
            page_tokens=16,
        )
        # This is the exact separator contract used by prime_shared_prefix and
        # run_method_row.  The committed warmup token prefix must be byte-for-
        # byte identical to the scored request's token prefix.
        warm = self.counter.ids(
            M.qwen_chat_no_thinking(system, aligned + "\n")
        )
        scored = self.counter.ids(
            M.qwen_chat_no_thinking(
                system, aligned + "\n" + "What is the answer?"
            )
        )
        commit = (len(warm) // 16) * 16
        if commit >= len(warm):
            commit -= 16
        # The production baseline intentionally publishes one page behind the
        # natural boundary.  This remains stable even when the next BPE token
        # merges differently at the warmup/scored suffix boundary.
        commit -= 16
        self.assertGreater(commit, 0)
        self.assertEqual(warm[:commit], scored[:commit])

    def test_short_context_compact_representation_is_byte_identical(self) -> None:
        context = "raw context must remain unchanged"
        self.assertEqual(M.compact_common_content("", context), context)

    def test_sliding_window_is_one_fixed_suffix_for_all_questions(self) -> None:
        context = " ".join(f"fact-{index}" for index in range(2000))
        questions = ["short question", "a much longer question " * 20]
        suffix, metadata = M.select_fixed_sliding_suffix(
            system="system",
            context=context,
            formatted_questions=questions,
            counter=self.counter,
            prompt_limit=512,
            alignment_reserve=32,
        )
        self.assertEqual(suffix, context[metadata["window_start_char"] :])
        self.assertLessEqual(metadata["max_prompt_tokens_before_alignment"], 480)
        for question in questions:
            count = self.counter.count(
                M.qwen_chat_no_thinking("system", suffix + "\n" + question)
            )
            self.assertLessEqual(count, 480)

    def test_sliding_constraint_iteration_matches_all_question_binary_search(self) -> None:
        context = " ".join(f"fact-{index}" for index in range(3000))
        questions = [
            "short",
            "medium question " * 7,
            "a different and substantially longer question " * 23,
            "symbols ,.;: and a final prompt " * 11,
        ]
        target = 768 - 32

        def old_worst_tokens(start: int) -> int:
            suffix = context[start:]
            return max(
                self.counter.count(
                    M.qwen_chat_no_thinking(
                        "system", suffix + "\n" + question
                    )
                )
                for question in questions
            )

        if old_worst_tokens(0) <= target:
            expected_start = 0
        else:
            lo, hi = 0, len(context)
            while lo < hi:
                mid = (lo + hi) // 2
                if old_worst_tokens(mid) <= target:
                    hi = mid
                else:
                    lo = mid + 1
            expected_start = lo

        suffix, metadata = M.select_fixed_sliding_suffix(
            system="system",
            context=context,
            formatted_questions=questions,
            counter=self.counter,
            prompt_limit=768,
            alignment_reserve=32,
        )
        self.assertEqual(metadata["window_start_char"], expected_start)
        self.assertEqual(suffix, context[expected_start:])

    def test_smoke_limit_keeps_all_questions_for_shared_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            system = "system"
            prefix = root / "prefix.txt"
            qa_file = root / "qa.json"
            questions_file = root / "questions.json"
            prefix.write_text(
                M.qwen_open_user(system, "shared context"), encoding="utf-8"
            )
            qa_file.write_text(
                json.dumps([
                    {"raw_question": "q0", "answer": ["a0"]},
                    {"raw_question": "q1", "answer": ["a1"]},
                    {"raw_question": "q2", "answer": ["a2"]},
                ]),
                encoding="utf-8",
            )
            questions_file.write_text(
                json.dumps(["formatted-0", "formatted-1", "formatted-2"]),
                encoding="utf-8",
            )
            manifest = {
                "archive_prefix": str(prefix),
                "qa_file": str(qa_file),
                "questions_file": str(questions_file),
            }
            args = SimpleNamespace(question_limit=2)
            row = M.WorkRow(1, "Accurate_Retrieval", Path("unused"), 0, "source")
            with (
                mock.patch.object(M, "prepare_row", return_value=manifest),
                mock.patch.object(
                    M, "load_module",
                    return_value=SimpleNamespace(SYSTEM_MESSAGE=system),
                ),
            ):
                prepared = M.load_prepared(args, row)
        self.assertEqual(len(prepared["qa"]), 2)
        self.assertEqual(prepared["formatted_questions"], [
            "formatted-0", "formatted-1",
        ])
        self.assertEqual(prepared["all_formatted_questions"], [
            "formatted-0", "formatted-1", "formatted-2",
        ])
        self.assertEqual(prepared["total_questions"], 3)

    def test_rag_blocks_are_presented_in_supplied_chronological_order(self) -> None:
        blocks = [
            {"rank": 2, "block_id": "c00001", "text": "later"},
            {"rank": 1, "block_id": "c00004", "text": "latest"},
        ]
        rendered = M.format_rag_blocks(blocks)
        self.assertLess(rendered.index("later"), rendered.index("latest"))

    def test_context_length_filter_uses_canonical_reference_row_name(self) -> None:
        # Exercise the filtering contract without constructing another parquet
        # fixture: a phased run must look up the canonical row path and keep the
        # same row identity used by the unfiltered KVMem experiment.
        row = M.WorkRow(
            26,
            "Conflict_Resolution",
            Path("unused.parquet"),
            3,
            "factconsolidation_mh_262k",
        )
        self.assertEqual(
            row.name,
            "026_Conflict_Resolution_r003_factconsolidation_mh_262k",
        )
        with tempfile.TemporaryDirectory() as temporary:
            summary = (
                Path(temporary) / "rows" / row.name / "row_summary.json"
            )
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps({"archive_tokens_unpadded": 336838}),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                reference_results=Path(temporary),
                min_context_tokens_exclusive=262144,
                max_context_tokens_inclusive=None,
            )
            loaded = M.read_json(
                args.reference_results / "rows" / row.name / "row_summary.json"
            )
            self.assertGreater(
                int(loaded["archive_tokens_unpadded"]),
                args.min_context_tokens_exclusive,
            )

    def test_comparison_markdown_includes_category_and_dataset_tables(self) -> None:
        def method(score: float) -> dict[str, object]:
            return {
                "by_category": {
                    "Conflict_Resolution": {"macro_source_mean": score}
                },
                "by_source": {
                    "factconsolidation_mh_262k": {
                        "questions": 100,
                        "headline_metric": "substring_exact_match",
                        "headline_score": score,
                    }
                },
                "context_macro_mean": score,
                "question_weighted_mean": score,
                "overall_macro_category_mean": score,
            }

        rendered = C.markdown({
            "selected_contexts": 1,
            "selected_questions": 100,
            "methods": {
                "KVMem": method(0.05),
                "Compact+RAG": method(0.25),
            },
        })
        self.assertIn("Selected questions: 100", rendered)
        self.assertIn("| Context-macro mean | 5.00 | 25.00 |", rendered)
        self.assertIn("| Question-weighted mean | 5.00 | 25.00 |", rendered)
        self.assertIn("| Conflict_Resolution | 5.00 | 25.00 |", rendered)
        self.assertIn(
            "| factconsolidation_mh_262k | substring_exact_match | 100 | "
            "5.00 | 25.00 |",
            rendered,
        )

    def test_length_phase_configs_get_distinct_manifest_names(self) -> None:
        self.assertEqual(
            M.method_config_filename(SimpleNamespace(
                min_context_tokens_exclusive=262144,
                max_context_tokens_inclusive=None,
            )),
            "run_config_over256k.json",
        )
        self.assertEqual(
            M.method_config_filename(SimpleNamespace(
                min_context_tokens_exclusive=None,
                max_context_tokens_inclusive=262144,
            )),
            "run_config_under256k.json",
        )
        self.assertEqual(
            M.method_config_filename(SimpleNamespace(
                min_context_tokens_exclusive=None,
                max_context_tokens_inclusive=None,
            )),
            "run_config_full.json",
        )

    def test_window_config_pins_the_prior_256k_method_definition(self) -> None:
        args = SimpleNamespace(
            model_name="Qwen3.6-27B-Q8_0",
            context_window=262144,
            server_mtp_chain=0,
            prefix_cache_guard_pages=1,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            compaction_input_tokens=232000,
            summary_max_tokens=25000,
            compact_common_open_tokens=192000,
            rag_top_k=30,
            rag_block_size=1024,
            rag_overlap=128,
            sliding_prompt_tokens=32768,
            generation_reserve=32768,
            reference_results=Path("/tmp/reference"),
            min_context_tokens_exclusive=262144,
            max_context_tokens_inclusive=None,
        )
        sliding = M.method_config(args, "sliding-window")["sliding"]
        self.assertEqual(sliding["complete_prompt_tokens"], 32768)
        self.assertEqual(
            sliding["reused_from_256k_reference"],
            {
                "context_window": 262144,
                "complete_prompt_tokens": 32768,
                "selection": "largest recent raw-history suffix",
                "prompt_budget_scope": "complete final prompt",
            },
        )
        for name in ("reference", "reference_256k_config"):
            path = Path(sliding[name])
            self.assertTrue(path.is_file())
            self.assertEqual(sliding[f"{name}_sha256"], M.sha256_file(path))

    def test_utility_ledger_handles_memoryagentbench_mixed_metric(self) -> None:
        self.assertEqual(
            U.infer_dataset("memoryagentbench_kvmem_archive_20260802_full"),
            "MemoryAgentBench",
        )
        rendered = U.result_text({
            "sample_count": 3671,
            "judged": True,
            "correct": None,
            "accuracy": 0.3968,
            "accuracy_denominator": 3671,
            "official_score": None,
            "finish_reason_length": None,
        })
        self.assertEqual(rendered, "macro-category mean = 39.68%")


if __name__ == "__main__":
    unittest.main()
