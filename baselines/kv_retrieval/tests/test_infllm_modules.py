"""Correctness-only GPU fixtures; their runtimes are not benchmark results."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import unittest

import torch
import torch.nn.functional as F

from kv_baselines.infllm_adapter import InfLLMConfig, PartialRotaryEmbedding, attention_forward
from kv_baselines.upstream import infllm_module, verify_source

METRICS = {}
ASYNC_GLOBAL_STREAM = False


class CPUChecks(unittest.TestCase):
    def test_sources(self):
        for name in ["InfLLM", "FlexiCache"]:
            verify_source(name)

    def test_partial_rope_preserves_nonrotary_channels(self):
        torch.manual_seed(713)
        rope = PartialRotaryEmbedding(64, 1e7, device="cpu")
        q = torch.randn(1, 24, 7, 256)
        k = torch.randn(1, 4, 7, 256)
        qr, kr = rope(q, k)
        torch.testing.assert_close(qr[..., 64:], q[..., 64:], rtol=0, atol=0)
        torch.testing.assert_close(kr[..., 64:], k[..., 64:], rtol=0, atol=0)
        freq = torch.arange(7)[:, None] / (1e7 ** (torch.arange(0, 64, 2) / 64))
        angle = torch.cat((freq, freq), dim=-1)
        expected = q[..., :64] * angle.cos() + torch.cat((-q[..., 32:64], q[..., :32]), -1) * angle.sin()
        torch.testing.assert_close(qr[..., :64], expected)

    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):
            InfLLMConfig(topk=33, max_cached_block=32)


class GPUChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(719)
        self.cfg = InfLLMConfig(n_init=16, n_local=64, block_size=32,
            max_cached_block=4, topk=2, exc_block_size=32, repr_topk=4,
            async_global_stream=ASYNC_GLOBAL_STREAM, pin_memory=ASYNC_GLOBAL_STREAM)

    def tensors(self, length, dtype=torch.float16):
        return [torch.randn(1, heads, length, 256, device="cuda", dtype=dtype) * .25
                for heads in [24, 4, 4]]

    def manager(self, fused=True, **kw):
        return replace(self.cfg, fattn=fused, **kw).create(PartialRotaryEmbedding(64, 1e7))

    def test_fp32_logits_keep_close_candidates_distinct(self):
        from types import SimpleNamespace
        from kv_baselines.precision import use_fp32_logits
        head = torch.nn.Linear(2, 2, bias=False, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1., 1.], [1., 1.0078125]], device="cuda"))
        inputs = torch.tensor([[[1., .25]]], dtype=torch.bfloat16, device="cuda")
        rounded = head(inputs)
        use_fp32_logits(SimpleNamespace(lm_head=head))
        actual = head(inputs)
        expected = inputs.float() @ head.weight.float().t()
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(rounded[0, 0, 0].item(), rounded[0, 0, 1].item())
        self.assertGreater(actual[0, 0, 1].item(), actual[0, 0, 0].item())

    def test_short_context_equals_dense(self):
        q, k, v = self.tensors(61)
        manager = self.manager()
        actual = manager.append(q, k, v, q, k, v)
        rope = PartialRotaryEmbedding(64, 1e7)
        qr, kr = rope(q, k)
        ref = F.scaled_dot_product_attention(qr, kr, v, is_causal=True, enable_gqa=True)
        METRICS["short_dense_max_abs_error"] = (actual - ref).abs().max().item()
        torch.testing.assert_close(actual, ref, rtol=.04, atol=.005)
        self.assertEqual(manager.length, 61)
        self.assertEqual(manager.num_global_block, 0)

    def test_torch_triton_retrieval_and_incremental(self):
        # Force offload, more candidates than top-k, and incremental decode.
        q, k, v = self.tensors(337)
        managers = [self.manager(fused=False), self.manager(fused=True)]
        max_error = 0
        for start, end in [(0, 256), (256, 288), (288, 320), (320, 321), (321, 337)]:
            qq, kk, vv = [t[..., start:end, :].contiguous() for t in (q, k, v)]
            outputs = [m.append(qq, kk, vv, qq, kk, vv) for m in managers]
            max_error = max(max_error, (outputs[0] - outputs[1]).abs().max().item())
            torch.testing.assert_close(outputs[0], outputs[1], rtol=.06, atol=.006)
            self.assertEqual(managers[0].num_global_block, managers[1].num_global_block)
            self.assertEqual(managers[0].global_buffer_block_id_list, managers[1].global_buffer_block_id_list)
            for m in managers:
                self.assertLessEqual(len(m.cached_blocks[0]), m.max_cached_block)
                self.assertEqual(m.length, end)
        METRICS["retrieval_torch_triton_max_abs_error"] = max_error
        METRICS["global_blocks_after_337_tokens"] = managers[1].num_global_block
        self.assertGreater(managers[1].num_global_block, self.cfg.topk)

    def test_gpu_cache_hit_avoids_host_copy(self):
        cm = infllm_module("context_manager")
        cache = cm.CudaCache(1, 2 * 4 * 32 * 256, torch.float16)
        kv = tuple(t[0] for t in self.tensors(32)[1:])
        unit = cm.MemoryUnit(kv, cache, pin_memory=True)
        torch.cuda.synchronize()
        # This is the actual ContextManager call path, which supplies a target.
        target = tuple(torch.empty_like(t) for t in kv)
        first, _ = unit.load(target)
        second, _ = unit.load(target)
        unit.offload()
        third, _ = unit.load(target)
        self.assertEqual((first, second, third), (True, False, True))
        torch.cuda.synchronize()
        torch.testing.assert_close(unit.get()[0], kv[0], rtol=0, atol=0)

    def test_qwen_projection_gate_and_norm(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextRotaryEmbedding
        cfg = Qwen3_5TextConfig(hidden_size=128, num_attention_heads=4,
            num_key_value_heads=2, head_dim=64, num_hidden_layers=1, layer_types=["full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": .25,
                             "mrope_section": [3, 3, 2], "mrope_interleaved": True})
        cfg._attn_implementation = "sdpa"
        module = Qwen3_5Attention(cfg, 0).to(device="cuda", dtype=torch.bfloat16).eval()
        hidden = torch.randn(1, 31, 128, device="cuda", dtype=torch.bfloat16)
        pos = torch.arange(31, device="cuda")[None, None].expand(3, 1, -1)
        hf_rope = Qwen3_5TextRotaryEmbedding(cfg, device="cuda")
        expected, _ = module(hidden, hf_rope(hidden, pos), attention_mask=None)
        manager = self.cfg.create(PartialRotaryEmbedding(16, 1e7))
        actual = attention_forward(module, manager, hidden)
        METRICS["qwen_attention_boundary_max_abs_error"] = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=.04, atol=.01)

    def test_hybrid_model_incremental_states(self):
        from transformers import DynamicCache
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        from kv_baselines.infllm_hybrid import install, make_cache, forward_text
        cfg = Qwen3_5TextConfig(hidden_size=128, intermediate_size=256, vocab_size=512,
            num_attention_heads=4, num_key_value_heads=2, head_dim=64, num_hidden_layers=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=16, linear_value_head_dim=16,
            rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": .25,
                             "mrope_section": [3, 3, 2], "mrope_interleaved": True})
        cfg._attn_implementation = "sdpa"
        model = Qwen3_5ForCausalLM(cfg).to(device="cuda", dtype=torch.float16).eval()
        ids = torch.randint(0, 512, (1, 41), device="cuda")
        dense_cache = DynamicCache(config=cfg)
        spans = [(0, 31), (31, 40), (40, 41)]
        expected = [model(input_ids=ids[:, a:b], past_key_values=dense_cache,
                    use_cache=True).logits for a, b in spans]
        self.assertEqual(install(model), [3])
        cache = make_cache(cfg, self.cfg)
        for (a, b), ref in zip(spans, expected):
            output = forward_text(model, ids[:, a:b], cache)
            torch.testing.assert_close(output.logits, ref, rtol=.03, atol=.004)
            self.assertEqual(cache.get_seq_length(), b)
        for i in range(3):
            torch.testing.assert_close(cache.layers[i].conv_states, dense_cache.layers[i].conv_states, rtol=0, atol=0)
            torch.testing.assert_close(cache.layers[i].recurrent_states, dense_cache.layers[i].recurrent_states, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "checkpoint"):
            cache.layers[3].crop(35)
        METRICS["hybrid_incremental_final_logits_max_abs_error"] = (output.logits - expected[-1]).abs().max().item()

    def test_tail_rewrite_restores_hybrid_state_and_history(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        from kv_baselines.infllm_hybrid import install, make_cache, forward_text
        from kv_baselines.session_state import CheckpointRing
        cfg = Qwen3_5TextConfig(hidden_size=128, intermediate_size=256, vocab_size=512,
            num_attention_heads=4, num_key_value_heads=2, head_dim=64, num_hidden_layers=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=16, linear_value_head_dim=16,
            rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": .25,
                             "mrope_section": [3, 3, 2], "mrope_interleaved": True})
        cfg._attn_implementation = "sdpa"
        model = Qwen3_5ForCausalLM(cfg).to(device="cuda", dtype=torch.float16).eval()
        install(model)
        icfg = replace(self.cfg, n_local=32, n_init=8, block_size=8, exc_block_size=16,
                       topk=2, max_cached_block=4)
        ids = torch.randint(0, 512, (1, 144), device="cuda")
        replacement = torch.randint(0, 512, (1, 11), device="cuda")
        cache = make_cache(cfg, icfg)
        forward_text(model, ids[:, :96], cache)
        ring = CheckpointRing(2)
        ring.capture(ids[0, :96].tolist(), cache)
        forward_text(model, ids[:, 96:128], cache)
        ring.capture(ids[0, :128].tolist(), cache)
        forward_text(model, ids[:, 128:], cache)
        new_ids = ids[0, :96].tolist() + replacement[0].tolist()
        restored, pos = ring.restore_for(new_ids)
        self.assertEqual(pos, 96)
        self.assertEqual(len(ring.checkpoints), 1)
        observed = forward_text(model, replacement, restored).logits
        fresh = make_cache(cfg, icfg)
        forward_text(model, ids[:, :96], fresh)
        expected = forward_text(model, replacement, fresh).logits
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        self.assertEqual(restored.layers[3].manager.num_global_block, fresh.layers[3].manager.num_global_block)
        for i in range(3):
            torch.testing.assert_close(restored.layers[i].recurrent_states, fresh.layers[i].recurrent_states, rtol=0, atol=0)
        self.assertEqual(restored.get_seq_length(), 107)
        METRICS["tail_rewrite_restored_prefix_tokens"] = pos
        METRICS["tail_rewrite_logits_max_abs_error"] = (observed - expected).abs().max().item()
        self.assertEqual(ring.restore_for([999] + new_ids[1:]), (None, 0))

    def test_session_short_append_preserves_safe_checkpoint(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        from kv_baselines.infllm_hybrid import install
        from kv_baselines.session import InfLLMSession
        cfg = Qwen3_5TextConfig(hidden_size=128, intermediate_size=256, vocab_size=512,
            num_attention_heads=4, num_key_value_heads=2, head_dim=64, num_hidden_layers=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=16, linear_value_head_dim=16,
            rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": .25,
                             "mrope_section": [3, 3, 2], "mrope_interleaved": True})
        cfg._attn_implementation = "sdpa"
        model = Qwen3_5ForCausalLM(cfg).to(device="cuda", dtype=torch.float16).eval()
        install(model)
        session = InfLLMSession(model, self.cfg, prefill_chunk=32)
        ids = torch.randint(0, 512, (227,)).tolist()
        _, first = session.begin(ids[:192])
        self.assertEqual(first["checkpoint_positions"], [160])
        for token in ids[192:195]:
            session.consume_generated_token(token)
        _, second = session.begin(ids[:216])
        self.assertEqual(second["resume_origin"], 195)
        self.assertEqual(second["checkpoint_positions"], [160])
        session.consume_generated_token(ids[216])
        rewritten = ids[:211] + [(t + 1) % 512 for t in ids[211:]]
        actual, third = session.begin(rewritten)
        self.assertEqual(third["resume_origin"], 160)
        self.assertEqual(third["prefill_tokens"], 67)
        self.assertFalse(third["cold_prefill"])
        fresh = InfLLMSession(model, self.cfg, prefill_chunk=32)
        expected, _ = fresh.begin(rewritten)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        METRICS["short_append_then_rewrite_prefill_tokens"] = third["prefill_tokens"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--async-global-stream", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    ASYNC_GLOBAL_STREAM = args.async_global_stream
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CPUChecks)
    if args.gpu:
        # Bound this process, never the other workload. These fixtures are tiny.
        torch.cuda.set_per_process_memory_fraction(0.06)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(GPUChecks))
    start = time.time()
    with torch.inference_mode():
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(correctness_only=True, shared_gpu=args.gpu, performance_claim=False,
        async_global_stream=args.async_global_stream,
        timestamp=time.time(), elapsed_seconds=time.time()-start, tests=result.testsRun,
        failures=[(str(t), msg) for t, msg in result.failures],
        errors=[(str(t), msg) for t, msg in result.errors], metrics=METRICS,
        torch=torch.__version__, sources={n: verify_source(n) for n in ["InfLLM", "FlexiCache"]})
    if args.gpu:
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["gpu"] = torch.cuda.get_device_name()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
