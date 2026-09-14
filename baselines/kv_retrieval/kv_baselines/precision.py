"""Common output-logit precision without duplicating or dequantizing weights."""
from types import MethodType
import torch


def _fp32_logits(self, hidden_states):
    shape = hidden_states.shape[:-1]
    # CUDA tensor-core BF16/FP16 inputs, FP32 accumulation and output. The
    # default HF Linear returns BF16 logits, unlike native qw3's FP32 logits;
    # rounding close candidates into ties can change greedy/top-k decisions.
    logits = torch.mm(hidden_states.reshape(-1, hidden_states.shape[-1]),
                      self.weight.t(), out_dtype=torch.float32)
    if self.bias is not None:
        logits = logits + self.bias.float()
    return logits.reshape(*shape, self.weight.shape[0])


def use_fp32_logits(model):
    head = model.lm_head
    if head.weight.dtype not in (torch.float16, torch.bfloat16) or not head.weight.is_cuda:
        raise ValueError("FP32-logit control requires CUDA BF16/FP16 linear weights")
    head.forward = MethodType(_fp32_logits, head)
