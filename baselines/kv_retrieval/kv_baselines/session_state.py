"""Explicit hybrid snapshots for safe tail rewrites in the InfLLM adapter.

Historical CPU KV blocks are immutable and shared. Mutable recent tensors and
recurrent state are copied; GPU residency is an optimization and is rebuilt on
restore. No checkpoint after the token LCP is eligible. This is adapter state
management, not a change to InfLLM's retrieval or attention algorithm.
"""
from copy import copy
from dataclasses import dataclass

import torch
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from .infllm_hybrid import InfLLMLayer
from .upstream import infllm_module


def _freeze(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_freeze(x) for x in value)
    if isinstance(value, list):
        return [_freeze(x) for x in value]
    if isinstance(value, dict):
        return {k: _freeze(v) for k, v in value.items()}
    return value


@dataclass
class ManagerSnapshot:
    attributes: dict
    cpu_blocks: list
    representatives: list

    @classmethod
    def capture(cls, manager):
        if not manager.initialized or manager.perhead or manager.faiss:
            raise ValueError("snapshot requires initialized batched-head Tensor-index InfLLM")
        # CPU data must be complete before a snapshot exposes immutable blocks.
        torch.cuda.synchronize()
        exclude = {"cuda_cache", "global_blocks", "cached_blocks", "block_k", "global_buffer",
                   "global_buffer_block_id_list", "global_buffer_init_st", "global_buffer_init_ed"}
        state = {k: _freeze(v) for k, v in vars(manager).items() if k not in exclude}
        cpu = [[unit.cpu_data for unit in batch] for batch in manager.global_blocks]
        # Representatives are append-only within a branch. Retain only the used
        # prefix. On a branch change the ring invalidates all later snapshots.
        vectors = [copy(v) for v in manager.block_k]
        return cls(state, cpu, vectors)

    def restore(self, layer):
        module = infllm_module("context_manager")
        layer.manager = None
        manager = layer.ensure_manager(self.attributes["local_k"].device)
        a = self.attributes
        q = torch.empty(a["batch_size"], a["num_heads"], 0, a["dim_head"],
                        device=a["local_k"].device, dtype=a["dtype"])
        k = torch.empty(a["batch_size"], a["num_heads_kv"], 0, a["dim_head"],
                        device=q.device, dtype=q.dtype)
        manager.init(q, k, k, q, k, k)
        for key, value in a.items():
            setattr(manager, key, _freeze(value))
        manager.global_blocks = [[module.MemoryUnit(kv, manager.cuda_cache,
            load_to_cache=False, pin_memory=False) for kv in batch] for batch in self.cpu_blocks]
        manager.block_k = [copy(v) for v in self.representatives]
        # Fresh cache/buffer metadata prevents accidentally reusing stale GPU
        # rows from a different branch. Historical CPU blocks remain recoverable.
        return manager


@dataclass
class Checkpoint:
    tokens: tuple
    cache_shell: object
    layers: list

    @classmethod
    def capture(cls, tokens, cache):
        if len(tokens) != cache.get_seq_length():
            raise ValueError("checkpoint tokens do not match authoritative cache length")
        layers = []
        for layer in cache.layers:
            if isinstance(layer, InfLLMLayer):
                layers.append((copy(layer), ManagerSnapshot.capture(layer.manager)))
            elif isinstance(layer, LinearAttentionCacheLayerMixin):
                snapshot = copy(layer)
                snapshot.__dict__ = {k: _freeze(v) for k, v in vars(layer).items()}
                layers.append((snapshot, None))
            else:
                raise TypeError(f"unvalidated cache layer: {type(layer)}")
        shell = copy(cache)
        shell.layers = []
        # Attention shells must not keep mutable managers/GPU cache pools alive.
        for layer, _ in layers:
            if isinstance(layer, InfLLMLayer):
                layer.manager = None
        return cls(tuple(tokens), shell, layers)

    def restore(self):
        cache = copy(self.cache_shell)
        cache.layers = []
        for old_layer, manager_snapshot in self.layers:
            layer = copy(old_layer)
            if manager_snapshot is not None:
                manager_snapshot.restore(layer)
            else:
                layer.__dict__ = {k: _freeze(v) for k, v in vars(old_layer).items()}
            cache.layers.append(layer)
        return cache


class CheckpointRing:
    def __init__(self, capacity=3):
        if capacity < 1:
            raise ValueError("checkpoint capacity must be positive")
        self.capacity = capacity
        self.checkpoints = []

    def capture(self, tokens, cache):
        checkpoint = Checkpoint.capture(tokens, cache)
        self.checkpoints = [c for c in self.checkpoints if len(c.tokens) < len(tokens)]
        self.checkpoints.append(checkpoint)
        self.checkpoints = self.checkpoints[-self.capacity:]

    def restore_for(self, tokens):
        tokens = tuple(tokens)
        eligible = [c for c in self.checkpoints if len(c.tokens) < len(tokens)
                    and tokens[:len(c.tokens)] == c.tokens]
        if not eligible:
            # The caller must perform an explicit cold start, never force reuse.
            self.checkpoints.clear()
            return None, 0
        checkpoint = max(eligible, key=lambda c: len(c.tokens))
        # No discarded-future representative prefix can survive a branch change.
        self.checkpoints = [c for c in self.checkpoints if len(c.tokens) <= len(checkpoint.tokens)]
        return checkpoint.restore(), len(checkpoint.tokens)
