"""Batched Q8 compute, strictly sequential original retrieval/recurrent updates.

Only the dense linear/MLP operations are batched. In particular, feeding the
whole draft into InfLLM.append would average its queries and change the method.
"""
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
from transaction import Transaction
import infinigen_pipeline


def recurrent(module, x, cache, tx, idx):
    graph = getattr(module, '_verify_graph', None)
    if graph is not None and x.shape[1] == 5:
        assert graph['cache'] is cache
        graph['input'].copy_(x)
        graph['graph'].replay()
        tx.prefixes[idx].extend(graph['snapshots'])
        return graph['output']
    names = ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a')
    projected = {name:getattr(module, name)(x) for name in names}
    original = {name:getattr(module, name).forward for name in (*names, 'out_proj')}
    output = []
    try:
        module.out_proj.forward = lambda z: z
        for pos in range(x.shape[1]):
            for name in names:
                getattr(module, name).forward = lambda z, name=name, pos=pos: projected[name][:, pos:pos+1]
            fn = getattr(module, '_decode_original', module.forward)
            output.append(fn(x[:, pos:pos+1], cache_params=cache))
            tx.record(idx)
    finally:
        for name, fn in original.items():
            getattr(module, name).forward = fn
    return module.out_proj(torch.cat(output, dim=1))


def full_attention(module, x, rope, state, tx, idx, backend, previous_norm):
    shape = x.shape[:-1]
    q, gate = module.q_proj(x).view(*shape, -1, 2*module.head_dim).chunk(2, -1)
    q = module.q_norm(q).transpose(1, 2).contiguous()
    k = module.k_norm(module.k_proj(x).view(*shape, -1, module.head_dim)).transpose(1, 2).contiguous()
    v = module.v_proj(x).view(*shape, -1, module.head_dim).transpose(1, 2).contiguous()
    if backend:
        q, k = apply_rotary_pos_emb(q, k, *rope)
        futures, quant_k, quant_v = infinigen_pipeline.enqueue(
            backend,module,state,previous_norm,rope,k,v)
    else:
        q, k, v = (t.to(state.manager.baseline_attention_dtype) for t in (q, k, v))
    outputs = []
    for pos in range(shape[1]):
        qq, kk, vv = (t[:, :, pos:pos+1].contiguous() for t in (q, k, v))
        if backend:
            y = infinigen_pipeline.consume(state,futures[pos],qq,
                quant_k[:,:,pos:pos+1],quant_v[:,:,pos:pos+1],module.scaling)
        else:
            y = state.manager.append(qq, kk, vv, qq, kk, vv)
        outputs.append(y)
        tx.record(idx)
    y = torch.cat(outputs, dim=2).transpose(1, 2).reshape(*shape, -1).contiguous().to(x.dtype)
    return module.o_proj(y * gate.reshape(*shape, -1).sigmoid())


def verify(model, ids, cache, backend=None):
    assert ids.shape[0] == 1 and 1 <= ids.shape[1] <= 5
    assert backend is None or backend.active_cache is None
    start = cache.get_seq_length()
    x = model.model.embed_tokens(ids)
    pos = torch.arange(start, start+ids.shape[1], device=ids.device)[None]
    rope = model.model.rotary_emb(x, pos)
    tx = Transaction(cache)
    previous_norm = None
    for idx, layer in enumerate(model.model.layers):
        norm = layer.input_layernorm(x)
        if layer.layer_type == 'linear_attention':
            y = recurrent(layer.linear_attn, norm, cache, tx, idx)
        else:
            y = full_attention(layer.self_attn, norm, rope, cache.layers[idx],
                               tx, idx, backend, previous_norm)
        x = x + y
        norm_mlp = layer.post_attention_layernorm(x)
        graph = getattr(layer.mlp, '_verify_graph', None)
        if graph is not None and ids.shape[1] == 5:
            graph['input'].copy_(norm_mlp)
            graph['graph'].replay()
            x = x + graph['output']
        else:
            x = x + layer.mlp(norm_mlp)
        previous_norm = norm
    hidden = x
    logits = model.lm_head(model.model.norm(hidden))
    return logits, hidden, tx


def accepted_prefix(candidates, predictions):
    """Return accepted draft count; a correction/bonus always comes from target."""
    accepted = 0
    for draft, predicted in zip(candidates, predictions):
        if draft != predicted:
            break
        accepted += 1
    return accepted
