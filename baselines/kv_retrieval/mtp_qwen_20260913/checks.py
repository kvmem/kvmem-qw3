"""Validation-only fingerprints of mutable retrieval state (outside timing)."""
import hashlib
import torch


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def retrieval_state(cache):
    records = {}
    for idx, layer in enumerate(cache.layers):
        if hasattr(layer,'manager'):
            m = layer.manager
            records[idx] = dict(length=m.length,blocks=m.num_global_block,load_count=m.load_count,
                cached=[dict(x) for x in m.cached_blocks],buffer_maps=[list(x) for x in m.global_buffer_block_id_list],
                representatives=[digest(x.get_data()) for x in m.block_k],
                scores=digest(m.global_remainder_local_score),
                local=[digest(x[:,:,-8:]) for x in (m.local_k,m.local_v)],
                remainder=[digest(x) for x in m.global_remainder],
                last_block=[digest(t) for unit in m.global_blocks for t in unit[-1].cpu_data] if m.num_global_block else [])
        elif hasattr(layer,'partial_keys'):
            records[idx] = dict(length=layer.length,
                history_tail=digest(layer.cpu_kv[:,max(0,layer.length-8):layer.length]),
                partial_tail=digest(layer.partial_keys[:,max(0,layer.length-8):layer.length]))
    return records
