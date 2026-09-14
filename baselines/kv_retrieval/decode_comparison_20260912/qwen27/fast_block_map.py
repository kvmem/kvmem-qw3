"""Preserve upstream buffer placement while eliminating nested list searches."""
import hashlib
import inspect
from pathlib import Path
import textwrap

HERE=Path(__file__).resolve().parent


def install():
    from kv_baselines.upstream import infllm_module
    cm=infllm_module('context_manager')
    original=cm.ContextManager.get_global_hidden_and_mask
    source=textwrap.dedent(inspect.getsource(original))
    anchor='        global_block_map[u] = deepcopy(self.global_buffer_block_id_list[u])\n'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+
        '        selected_set = set(block_topk[u])\n'
        '        present_set = set(global_block_map[u])\n'
        '        free_slots = iter(j for j, old in enumerate(global_block_map[u])\n'
        '                          if old == -1 or old not in selected_set)\n')
    assert source.count('if b_idx in global_block_map[u]:')==1
    source=source.replace('if b_idx in global_block_map[u]:','if b_idx in present_set:')
    old='''            for j in range(self.topk):
                if global_block_map[u][j] == -1 or global_block_map[u][j] not in block_topk[u]:
                    st = j * self.block_size
                    ed = st + self.block_size
                    global_block_map[u][j] = b_idx
                    break
'''
    new='''            j = next(free_slots)
            st = j * self.block_size
            ed = st + self.block_size
            present_set.discard(global_block_map[u][j])
            global_block_map[u][j] = b_idx
            present_set.add(b_idx)
'''
    assert source.count(old)==1
    source=source.replace(old,new)
    target=HERE/'generated_block_map.py'
    target.write_text(source)
    namespace=dict(original.__globals__)
    exec(compile(source,str(target),'exec'),namespace)
    accelerated=namespace[original.__name__]
    cm.ContextManager.get_global_hidden_and_mask=accelerated
    return original,accelerated,{'original_function_sha256':hashlib.sha256(textwrap.dedent(inspect.getsource(original)).encode()).hexdigest(),
                                 'generated_function_sha256':hashlib.sha256(source.encode()).hexdigest(),
                                 'generated_source':str(target),'change':'set membership and ascending free-slot iterator; identical block load and buffer placement order'}
