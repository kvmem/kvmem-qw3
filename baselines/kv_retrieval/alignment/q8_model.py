"""Load the existing Q8_0 GGUF into the text-only HF/InfLLM adapter.

Weight blocks are rearranged losslessly, never requantized or fully dequantized.
The frozen qw3 CUDA Q8 kernels perform projections. HF/FLA retains BF16
intermediate activations, unlike native qw3's FP32 residual stream; this is
recorded as an implementation difference rather than claimed bitwise parity.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader, GGMLQuantizationType
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding

from .native_q8 import NativeQ8, pack_native_rows


SUFFIX = {
    'input_layernorm.weight': 'attn_norm.weight',
    'post_attention_layernorm.weight': 'post_attention_norm.weight',
    'self_attn.q_proj.weight': 'attn_q.weight',
    'self_attn.k_proj.weight': 'attn_k.weight',
    'self_attn.v_proj.weight': 'attn_v.weight',
    'self_attn.o_proj.weight': 'attn_output.weight',
    'self_attn.q_norm.weight': 'attn_q_norm.weight',
    'self_attn.k_norm.weight': 'attn_k_norm.weight',
    'linear_attn.in_proj_qkv.weight': 'attn_qkv.weight',
    'linear_attn.in_proj_z.weight': 'attn_gate.weight',
    'linear_attn.in_proj_a.weight': 'ssm_alpha.weight',
    'linear_attn.in_proj_b.weight': 'ssm_beta.weight',
    'linear_attn.conv1d.weight': 'ssm_conv1d.weight',
    'linear_attn.A_log': 'ssm_a',
    'linear_attn.dt_bias': 'ssm_dt.bias',
    'linear_attn.norm.weight': 'ssm_norm.weight',
    'linear_attn.out_proj.weight': 'ssm_out.weight',
    'mlp.gate_proj.weight': 'ffn_gate.weight',
    'mlp.up_proj.weight': 'ffn_up.weight',
    'mlp.down_proj.weight': 'ffn_down.weight',
}


def gguf_name(name):
    special = {'model.embed_tokens.weight': 'token_embd.weight',
               'model.norm.weight': 'output_norm.weight', 'lm_head.weight': 'output.weight'}
    if name in special:
        return special[name]
    parts = name.split('.')
    assert parts[:2] == ['model', 'layers']
    return 'blk.' + parts[2] + '.' + SUFFIX['.'.join(parts[3:])]


def inverse_heads(k_heads, v_heads, dim):
    # native index -> HF index is the mapping in frozen qwen_weights.cpp.
    native = np.arange(v_heads * dim)
    head, component = native // dim, native % dim
    native_to_hf = ((head % k_heads) * (v_heads // k_heads) + head // k_heads) * dim + component
    return np.argsort(native_to_hf)


def undo_layout(raw, name, shape, config, quantized):
    raw = np.array(raw, copy=True)
    kh, vh = config.linear_num_key_heads, config.linear_num_value_heads
    kd, vd = config.linear_key_head_dim, config.linear_value_head_dim
    idx = inverse_heads(kh, vh, vd)
    if name.endswith('linear_attn.in_proj_qkv.weight') or name.endswith('linear_attn.conv1d.weight'):
        offset = 2 * kh * kd
        raw[offset:offset + len(idx)] = raw[offset + idx]
    elif name.endswith('linear_attn.in_proj_z.weight'):
        raw = raw[idx]
    elif any(name.endswith('linear_attn.' + s) for s in ['in_proj_a.weight', 'in_proj_b.weight', 'A_log', 'dt_bias']):
        raw = raw[inverse_heads(kh, vh, 1)]
    elif name.endswith('linear_attn.out_proj.weight'):
        if quantized:
            assert vd % 32 == 0
            blocks = raw.reshape(shape[0], shape[1] // 32, 34)
            raw = blocks[:, idx[::32] // 32].reshape(raw.shape)
        else:
            raw = raw[:, idx]
    if not quantized:
        raw = raw.reshape(shape).astype(np.float32)
        if name.endswith('norm.weight') and not name.endswith('linear_attn.norm.weight'):
            raw -= 1
        if name.endswith('linear_attn.A_log'):
            if not np.all(raw < 0):
                raise ValueError('invalid converted negative-exp recurrent weights')
            raw = np.log(-raw)
    return raw


class Q8Linear(torch.nn.Module):
    def __init__(self, packed, cols, api, output_dtype=torch.bfloat16):
        super().__init__()
        self.register_buffer('packed_weight', packed)
        self.in_features, self.out_features = cols, packed.shape[0]
        self.api, self.output_dtype = api, output_dtype
        self.bias = None

    def forward(self, x):
        return self.api.matmul(self.packed_weight, x, self.in_features).to(self.output_dtype)


class Q8Embedding(torch.nn.Module):
    def __init__(self, packed, cols, api):
        super().__init__()
        self.register_buffer('packed_weight', packed)
        self.embedding_dim, self.num_embeddings = cols, packed.shape[0]
        self.api = api

    def forward(self, ids):
        return self.api.embedding(self.packed_weight, ids, self.embedding_dim).to(torch.bfloat16)


def load_q8_model(directory, gguf_path):
    directory, gguf_path = Path(directory), Path(gguf_path)
    config = Qwen3_5TextConfig(**json.loads((directory / 'config.json').read_text())['text_config'])
    config._attn_implementation = 'sdpa'
    with torch.device('meta'):
        model = Qwen3_5ForCausalLM(config)
    expected = dict(model.named_parameters())
    reader = GGUFReader(gguf_path)
    tensors = {t.name: t for t in reader.tensors}
    api = NativeQ8()
    loaded = []
    for name, parameter in expected.items():
        source = gguf_name(name)
        if source not in tensors and 'post_attention_norm' in source:
            source = source.replace('post_attention_norm', 'ffn_norm')
        tensor = tensors[source]
        shape = tuple(parameter.shape)
        quant = tensor.tensor_type == GGMLQuantizationType.Q8_0
        if tuple(reversed(tensor.shape)) != shape and not name.endswith('conv1d.weight'):
            raise ValueError(f'shape mismatch: {name} {shape} {tensor.shape}')
        if not quant and tensor.tensor_type != GGMLQuantizationType.F32:
            raise ValueError(f'unexpected weight format: {source}')
        raw = undo_layout(tensor.data, name, shape, config, quant)
        parent_name, leaf = name.rsplit('.', 1)
        if quant:
            packed = torch.from_numpy(pack_native_rows(raw, shape[1])).cuda()
            if name == 'model.embed_tokens.weight':
                replacement = Q8Embedding(packed, shape[1], api)
            else:
                replacement = Q8Linear(packed, shape[1], api,
                    torch.float32 if name == 'lm_head.weight' else torch.bfloat16)
            model.set_submodule(parent_name, replacement)
        else:
            value = torch.from_numpy(raw).to(device='cuda', dtype=torch.bfloat16)
            # Keep discretization parameters in FP32, as in the original formula.
            if name.endswith(('A_log', 'dt_bias')):
                value = torch.from_numpy(raw).cuda()
            model.get_submodule(parent_name)._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
        loaded.append({'target': name, 'source': source, 'shape': shape, 'type': int(tensor.tensor_type)})
        if len(loaded) % 100 == 0:
            print(json.dumps({'q8_loaded': len(loaded), 'allocated_gib': torch.cuda.memory_allocated()/2**30}), flush=True)
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device='cuda')
    if any(t.is_meta for t in [*model.parameters(), *model.buffers()]):
        raise ValueError('uninitialized model tensor')
    stat = gguf_path.stat()
    manifest = {'weight_dtype': 'Q8_0', 'weight_path': str(gguf_path.resolve()),
        'weight_bytes': stat.st_size, 'weight_mtime_ns': stat.st_mtime_ns,
        'weight_layout': 'lossless inverse GGUF head permutation + native split-plane Q8_0',
        'activation_dtype': 'BF16 HF/FLA; differs from native FP32 residual',
        'logits_dtype': 'float32', 'vision_loaded': False, 'mtp_loaded': False,
        'model_path': str(directory.resolve()), 'text_parameters': len(loaded),
        'mapping_sha256': hashlib.sha256(json.dumps(loaded, sort_keys=True).encode()).hexdigest(),
        'mapping': loaded}
    return model.eval(), manifest
