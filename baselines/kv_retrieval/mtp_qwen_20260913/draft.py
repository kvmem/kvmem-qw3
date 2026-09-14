"""Actual GGUF MTP predictor; shared Q8 weights, FP8 cache, explicit positions.

Uses HF Qwen norms/MLP/rotary arithmetic and frozen standalone Q8 projections.
Prefix priming needs only K/V, exactly as the native kv_only path does.
"""
import copy
import numpy as np
import torch
from gguf import GGUFReader, GGMLQuantizationType
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5RMSNorm, apply_rotary_pos_emb)
from common import WEIGHTS
from alignment.q8_model import Q8Linear, SUFFIX
from alignment.native_q8 import pack_native_rows
from qwen.backend import encode, attention, FP8


class Predictor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed = model.model.embed_tokens
        self.head = model.lm_head
        self.rope = model.model.rotary_emb
        config = copy.deepcopy(model.config)
        config.layer_types = ['full_attention']
        with torch.device('meta'):
            self.layer = Qwen3_5DecoderLayer(config, 0)
            self.enorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.hnorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.fc = torch.nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        tensors = {t.name: t for t in GGUFReader(WEIGHTS).tensors}
        names = dict(self.named_parameters())
        self.mapping = []
        for name, param in names.items():
            if name.startswith('layer.'):
                source = 'blk.64.' + SUFFIX[name[6:]]
            else:
                source = 'blk.64.nextn.' + {
                    'enorm.weight':'enorm.weight', 'hnorm.weight':'hnorm.weight',
                    'norm.weight':'shared_head_norm.weight', 'fc.weight':'eh_proj.weight'}[name]
            tensor = tensors[source]
            shape = tuple(param.shape)
            assert tuple(reversed(tensor.shape)) == shape, (name, shape, tensor.shape)
            parent, leaf = name.rsplit('.', 1)
            if tensor.tensor_type == GGMLQuantizationType.Q8_0:
                raw = np.array(tensor.data, copy=True)
                packed = torch.from_numpy(pack_native_rows(raw, shape[1])).cuda()
                self.set_submodule(parent, Q8Linear(packed, shape[1], model.lm_head.api))
            else:
                assert tensor.tensor_type == GGMLQuantizationType.F32 and leaf == 'weight'
                value = torch.from_numpy(np.array(tensor.data, copy=True).reshape(shape)-1)
                self.get_submodule(parent)._parameters[leaf] = torch.nn.Parameter(
                    value.to(device='cuda', dtype=torch.bfloat16), requires_grad=False)
            self.mapping.append(dict(target=name, source=source, shape=shape))
        assert not any(t.is_meta for t in self.parameters())
        self.config = config
        self.kv = None
        self.length = 0

    def reset(self, capacity):
        if capacity > self.config.max_position_embeddings:
            raise ValueError('Full-position MTP drafter exceeds the validated model position range')
        self.kv = torch.empty((2, 1, self.config.num_key_value_heads,
                               capacity, self.config.head_dim), dtype=FP8, device='cuda')
        self.length = 0

    def project(self, ids, hidden, start, query=True):
        x = self.fc(torch.cat((self.enorm(self.embed(ids)), self.hnorm(hidden)), dim=-1))
        norm = self.layer.input_layernorm(x)
        a = self.layer.self_attn
        shape = ids.shape
        k = a.k_norm(a.k_proj(norm).view(*shape, -1, a.head_dim)).transpose(1, 2)
        v = a.v_proj(norm).view(*shape, -1, a.head_dim).transpose(1, 2)
        positions = torch.arange(start, start + shape[1], device=ids.device)[None]
        rope = self.rope(x, positions)
        if query:
            q, gate = a.q_proj(norm).view(*shape, -1, a.head_dim * 2).chunk(2, -1)
            q = a.q_norm(q).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *rope)
        else:
            _, k = apply_rotary_pos_emb(k, k, *rope)
            q = gate = None
        end = start + shape[1]
        assert 0 <= start <= self.length and end <= self.kv.shape[-2]
        self.kv[0, :, :, start:end].copy_(encode(k))
        self.kv[1, :, :, start:end].copy_(encode(v))
        self.length = end
        return x, q, gate

    def prime(self, ids, target_hidden, previous_hidden, start):
        shifted = torch.cat((previous_hidden, target_hidden[:, :-1]), dim=1)
        self.project(ids, shifted, start, query=False)

    def step(self, token, hidden):
        x, q, gate = self.project(token, hidden, self.length)
        a = self.layer.self_attn
        y = attention(q, self.kv[0, :, :, :self.length],
                      self.kv[1, :, :, :self.length], a.scaling)
        y = y.transpose(1, 2).reshape(1, 1, -1).to(x.dtype)
        x = x + a.o_proj(y * gate.reshape(1, 1, -1).sigmoid())
        x = x + self.layer.mlp(self.layer.post_attention_layernorm(x))
        return self.head(self.norm(x)).argmax(-1), x

    def propose(self, seed, hidden, count):
        tokens = []
        for _ in range(count):
            seed, hidden = self.step(seed, hidden)
            tokens.append(seed)
        return torch.cat(tokens, dim=1) if tokens else seed[:, :0]
