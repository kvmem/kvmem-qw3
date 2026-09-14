"""Compare optimization outputs and selection state against existing references."""
import json
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'infinigen/InfiniGen'))
from qwen.backend import gather_gqa
from qwen.optimized import unique_gqa_rows
from kv_baselines.infllm_adapter import InfLLMConfig, PartialRotaryEmbedding
from kv_baselines.upstream import infllm_module
from alignment.fp8_kv import encode, install_fp8_storage
from fastpaths import enable_conv, install_fp8_views


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.manual_seed(112)
    report = {'passed': False, 'checks': []}
    for count in (1, 117, 2048):
        history = torch.randint(0, 256, (8192, 4, 256), dtype=torch.uint8, pin_memory=True)
        ids = torch.randint(0, 8192, (count, 1, 24))
        # Include both repeated GQA rows and disjoint per-head selection.
        if count == 117:
            ids[:] = ids[:, :, :1]
        expected = gather_gqa(ids, history, 24).cuda()
        unique, inverse = unique_gqa_rows(ids, 4, 24)
        compact = history.view(-1, 256).index_select(0, unique).cuda()
        actual = compact.index_select(0, inverse.cuda()).view(count, 24, 256)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        report['checks'].append({'gqa_selected_per_head': count, 'unique_rows': len(unique), 'byte_exact': True})

    cm = infllm_module('context_manager')
    original = cm.CudaCache, cm.MemoryUnit, cm.ContextManager.append
    cfg = InfLLMConfig(n_init=16, n_local=64, topk=2, max_cached_block=4,
        block_size=32, exc_block_size=32, async_global_stream=True)
    q = torch.randn(1, 24, 545, 256, device='cuda', dtype=torch.float16) * .25
    k, v = [torch.randn(1, 4, 545, 256, device='cuda', dtype=torch.float16) * .25 for _ in range(2)]
    spans = [(0, 256), (256, 288)] + [(i, i+1) for i in range(288, 545)]
    outputs, selections = [], []
    for install in (install_fp8_storage, install_fp8_views):
        cm.CudaCache, cm.MemoryUnit, cm.ContextManager.append = original
        cm._baseline_fp8_storage = False
        install()
        manager = cfg.create(PartialRotaryEmbedding(64, 1e7))
        oo, ss = [], []
        for a, b in spans:
            qq, kk, vv = [t[:, :, a:b].contiguous() for t in (q, k, v)]
            oo.append(manager.append(qq, kk, vv, qq, kk, vv).clone())
            ss.append([list(x) for x in manager.global_buffer_block_id_list])
        torch.cuda.synchronize()
        outputs.append(oo)
        selections.append(ss)
        del manager
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert selections[0] == selections[1]
    report['checks'].append({'infllm_appends': len(spans), 'outputs_byte_exact': True,
                            'selection_and_buffer_order_exact': True})

    enable_conv()
    from transformers.models.qwen3_5 import modeling_qwen3_5 as model
    for dtype in (torch.float16, torch.bfloat16):
        x = torch.randn(1, 8192, 1, device='cuda', dtype=dtype) * .2
        w = torch.randn(8192, 4, device='cuda', dtype=dtype) * .2
        state = torch.randn(1, 8192, 4, device='cuda', dtype=dtype) * .2
        a, b = state.clone(), state.clone()
        expected = model.torch_causal_conv1d_update(x, a, w, None, 'silu')
        actual = model.causal_conv1d_update(x, b, w, None, 'silu')
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.002)
        report['checks'].append({'conv_dtype': str(dtype), 'state_exact': True,
            'max_abs': float((actual-expected).abs().max())})
    report['passed'] = True
    (HERE / 'fastpath_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
