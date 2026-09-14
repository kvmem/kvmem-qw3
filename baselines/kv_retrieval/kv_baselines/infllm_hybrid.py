"""Attention-only integration with the modern HF hybrid cache interface.

Recurrent layers and their state updates are the original model code. This
prototype exposes only unpadded, append-only text forward; session rollback must
restore an actual snapshot, never crop a recurrent state or reuse a divergent KV.
"""
from types import MethodType
from transformers.cache_utils import DynamicCache, DynamicLayer

from .infllm_adapter import PartialRotaryEmbedding, attention_forward


class InfLLMLayer(DynamicLayer):
    def __init__(self, config, infllm_config):
        super().__init__()
        self.model_config = config
        self.infllm_config = infllm_config
        self.manager = None

    def ensure_manager(self, device):
        if self.manager is None:
            params = self.model_config.rope_parameters
            if params["rope_type"] != "default":
                raise ValueError("only validated default partial RoPE is supported")
            dim = int(self.model_config.head_dim * params["partial_rotary_factor"])
            rope = PartialRotaryEmbedding(dim, params["rope_theta"], device=device)
            self.manager = self.infllm_config.create(rope)
        return self.manager

    def get_seq_length(self):
        return self.manager.length if self.manager is not None else 0

    def get_mask_sizes(self, query_length):
        # HF's generated mask is unused by InfLLM, which constructs both causal
        # masks itself. Avoid allocating an O(query * full_history) dummy mask.
        return query_length, self.get_seq_length()

    def update(self, *args, **kwargs):
        raise RuntimeError("InfLLM cache must be updated through original ContextManager.append")

    def crop(self, *args, **kwargs):
        raise RuntimeError("InfLLM/hybrid rollback requires a verified state checkpoint")

    def reset(self):
        self.manager = None


def make_cache(config, infllm_config):
    cache = DynamicCache(config=config)
    for idx, layer_type in enumerate(config.layer_types):
        if layer_type == "full_attention":
            cache.layers[idx] = InfLLMLayer(config, infllm_config)
        elif layer_type != "linear_attention":
            raise ValueError(f"unsupported layer type: {layer_type}")
    return cache


def _forward(self, hidden_states, position_embeddings, attention_mask,
             past_key_values=None, **kwargs):
    if past_key_values is None or not isinstance(past_key_values.layers[self.layer_idx], InfLLMLayer):
        raise ValueError("pass a cache from infllm_hybrid.make_cache explicitly")
    if self.training:
        raise ValueError("InfLLM baseline is inference-only")
    layer = past_key_values.layers[self.layer_idx]
    return attention_forward(self, layer.ensure_manager(hidden_states.device), hidden_states), None


def install(model):
    """Patch only standard attention forward, preserve every parameter and recurrence."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
    patched = []
    for module in model.modules():
        if isinstance(module, Qwen3_5Attention):
            if hasattr(module, "_baseline_dense_forward"):
                raise RuntimeError("InfLLM already installed")
            module._baseline_dense_forward = module.forward
            module.forward = MethodType(_forward, module)
            patched.append(module.layer_idx)
    if not patched:
        raise ValueError("no compatible Qwen attention layers found")
    return patched


def uninstall(model):
    for module in model.modules():
        if hasattr(module, "_baseline_dense_forward"):
            module.forward = module._baseline_dense_forward
            del module._baseline_dense_forward


def forward_text(model, input_ids, cache, **kwargs):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("one nonempty, unpadded text sequence is required")
    if "attention_mask" in kwargs or "position_ids" in kwargs:
        raise ValueError("custom masks/positions are not validated by this text adapter")
    return model(input_ids=input_ids, past_key_values=cache, use_cache=True, **kwargs)
