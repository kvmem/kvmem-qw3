"""Qwen3.5/3.8 attention boundary for the original InfLLM ContextManager.

This file does not implement selection, cache replacement, or two-stage attention.
Those remain in the pinned author implementation. Text-only, batch size one.
"""
from dataclasses import asdict, dataclass
import torch

from .upstream import infllm_module, enable_infllm_head_dim_256


@dataclass(frozen=True)
class InfLLMConfig:
    # Author Qwen defaults, with the author's fused attention option enabled.
    n_init: int = 128
    n_local: int = 4096
    block_size: int = 128
    max_cached_block: int = 32
    topk: int = 16
    exc_block_size: int = 512
    repr_topk: int = 4
    fattn: bool = True
    cache_strategy: str = "lru"
    async_global_stream: bool = True
    pin_memory: bool = True
    chunk_topk_calc: int | None = None
    attention_dtype: str = "float16"

    def __post_init__(self):
        if self.attention_dtype not in ("float16", "bfloat16"):
            raise ValueError("attention_dtype must be float16 or bfloat16")
        if not 0 < self.repr_topk <= self.block_size:
            raise ValueError("representative count must fit inside a block")
        if not 0 < self.topk <= self.max_cached_block:
            raise ValueError("GPU cache must hold at least one full retrieval")
        if not 0 < self.exc_block_size <= self.n_local or self.n_init < 0:
            raise ValueError("invalid local/chunk/initial budget")
        if self.chunk_topk_calc is not None and (
            self.chunk_topk_calc < self.exc_block_size or self.chunk_topk_calc % self.exc_block_size
        ):
            raise ValueError("chunk_topk_calc must be a multiple of exc_block_size")

    def create(self, rope):
        if self.fattn:
            enable_infllm_head_dim_256()
        kwargs = asdict(self)
        dtype = kwargs.pop("attention_dtype")
        manager = infllm_module("context_manager").ContextManager(rope, **kwargs)
        manager.baseline_attention_dtype = getattr(torch, dtype)
        if self.fattn and "triton_impl" not in manager.Attn.__module__:
            raise RuntimeError("fused InfLLM attention unavailable; no silent slow fallback")
        return manager


class PartialRotaryEmbedding(infllm_module("rope").RotaryEmbeddingESM):
    """Apply InfLLM's original position rule to the model's rotary channels.

    For pure text, all three MRoPE axes have equal positions. Only the first
    rotary_dim channels rotate; the remaining Qwen head channels pass through.
    The author's original cosine cache, position choices and FP32 multiply remain.
    """
    def __init__(self, rotary_dim, base, device="cuda"):
        torch.nn.Module.__init__(self)
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError("rotary_dim must be positive and even")
        self.rotary_dim = rotary_dim
        self.base = base
        self.distance_scale = 1
        self.register_buffer("inv_freq", 1.0 / (base ** (
            torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim
        )), persistent=False)
        self._seq_len_cached = -1
        self._cos_cached = None
        self._sin_cached = None

    def apply_rotary_pos_emb(self, x, length, right, cos, sin):
        rotated = super().apply_rotary_pos_emb(x[..., :self.rotary_dim], length, right, cos, sin)
        return torch.cat((rotated, x[..., self.rotary_dim:]), dim=-1)

    def apply_rotary_pos_emb_one_angle(self, x, index):
        rotated = super().apply_rotary_pos_emb_one_angle(x[..., :self.rotary_dim], index)
        return torch.cat((rotated, x[..., self.rotary_dim:]), dim=-1)


def attention_forward(module, manager, hidden_states):
    """Original Qwen projections/norms/gate surrounding original InfLLM attention."""
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("this adapter supports one unpadded text sequence")
    shape = hidden_states.shape[:-1]
    head_shape = (*shape, -1, module.head_dim)
    query, gate = torch.chunk(module.q_proj(hidden_states).view(*shape, -1, 2 * module.head_dim), 2, dim=-1)
    q = module.q_norm(query.reshape(head_shape)).transpose(1, 2).contiguous()
    k = module.k_norm(module.k_proj(hidden_states).view(head_shape)).transpose(1, 2).contiguous()
    v = module.v_proj(hidden_states).view(head_shape).transpose(1, 2).contiguous()
    # Report this storage/attention precision explicitly, independent of weights.
    q, k, v = (t.to(manager.baseline_attention_dtype) for t in (q, k, v))
    output = manager.append(q, k, v, q, k, v)
    output = output.transpose(1, 2).reshape(*shape, -1).contiguous().to(hidden_states.dtype)
    output = output * torch.sigmoid(gate.reshape(*shape, -1))
    return module.o_proj(output)
