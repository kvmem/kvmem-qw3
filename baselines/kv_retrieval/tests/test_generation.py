import unittest
from unittest.mock import patch
import torch

from kv_baselines.frontend import NativeBridge, validate_text_request, meaningful_content
from kv_baselines.generation import sample, ThinkingBudget
from pathlib import Path


class GenerationChecks(unittest.TestCase):
    def test_nucleus_uses_full_vocabulary_mass_after_top_k(self):
        # Top-k mass .7 is below .8: both candidates survive. An incorrectly
        # renormalized intermediate distribution would drop the second token.
        scores = torch.tensor([.6, .1, .1, .1, .1]).log()
        with patch("torch.multinomial", return_value=torch.tensor([1])) as draw:
            token = sample(scores, torch.Generator().manual_seed(1), temperature=1,
                           top_k=2, top_p=.8)
        values = draw.call_args.args[0]
        self.assertEqual(int((values > 0).sum()), 2)
        self.assertIn(token, (1, 2, 3, 4))

    def test_reasoning_budget_feeds_real_guidance_then_close(self):
        root = Path(__file__).resolve().parents[1]
        bridge = NativeBridge(root.parents[1] / "models/Qwen3.8-27B-BF16")
        try:
            eos = bridge.call("tokenize", text="<|im_end|>")["input_ids"][0]
            policy = ThinkingBudget(bridge, True, 2, True, eos)
            self.assertEqual(policy.apply(10), 10)
            self.assertEqual(policy.apply(11), 11)
            forced = [policy.apply(12) for _ in range(len(policy.guidance) + 1)]
            self.assertEqual(forced, policy.guidance + [policy.close_id])
            self.assertFalse(policy.open)
            self.assertEqual(policy.apply(13), 13)
            policy = ThinkingBudget(bridge, True, 100, True, eos)
            self.assertEqual(policy.apply(eos), policy.close_id)
        finally:
            bridge.close()

    def test_reject_nested_image_in_tool_observation(self):
        request = {"messages": [{"role": "user", "content": [{"type": "tool_result",
            "content": [{"type": "image", "source": {"type": "base64", "data": "..."}}]}]}]}
        with self.assertRaisesRegex(ValueError, "unvalidated content block"):
            validate_text_request(request)
        self.assertFalse(meaningful_content({"type": "message_start"}))
        self.assertFalse(meaningful_content({"type": "content_block_start", "content_block": {"type": "text", "text": ""}}))
        self.assertTrue(meaningful_content({"type": "content_block_delta", "delta": {"thinking": "inspect"}}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
