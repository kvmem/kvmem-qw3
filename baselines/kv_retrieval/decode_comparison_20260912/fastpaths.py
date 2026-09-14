"""Opt-in engineering controls, independent of production qw3 and old trials."""
from pathlib import Path
import sys
from types import MethodType
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def enable_conv():
    sys.path.insert(0, str(HERE / 'vendor/conv_build'))
    sys.path.insert(0, str(HERE / 'vendor/causal-conv1d'))
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    modeling.causal_conv1d_fn = causal_conv1d_fn
    modeling.causal_conv1d_update = causal_conv1d_update
    modeling.is_fast_path_available = all((causal_conv1d_fn, causal_conv1d_update,
        modeling.chunk_gated_delta_rule, modeling.fused_recurrent_gated_delta_rule))
    assert modeling.is_fast_path_available


def install_fp8_views():
    """Retain exact FP16 views of already-rounded FP8 recent KV between appends.

    Historical CPU units and GPU block cache remain FP8. Recent working buffers
    contain FP8-representable values in FP16, eliminating lossless repeated
    conversion. Extra resident working-buffer memory is explicitly reported.
    """
    from kv_baselines.upstream import infllm_module
    from alignment.fp8_kv import encode, DTYPE
    cm = infllm_module('context_manager')
    if getattr(cm, '_baseline_fp8_storage', False):
        raise RuntimeError('Choose one FP8 storage implementation per process')
    original_cache, original_unit, original_append = cm.CudaCache, cm.MemoryUnit, cm.ContextManager.append

    class FP8CudaCache(original_cache):
        def __init__(self, num_units, unit_size, dtype):
            super().__init__(num_units, unit_size, DTYPE)

    class FP8MemoryUnit(original_unit):
        def __init__(self, kv, cache, load_to_cache=False, pin_memory=False):
            super().__init__(tuple(encode(t) for t in kv), cache, load_to_cache, pin_memory)

    def append(self, lq, lk, lv, gq, gk, gv):
        if lk.data_ptr() != gk.data_ptr() or lv.data_ptr() != gv.data_ptr():
            raise ValueError('Require identical raw local/global KV')
        lk, lv = (encode(t).to(lq.dtype) for t in (lk, lv))
        return original_append(self, lq, lk, lv, gq, lk, lv)

    cm.CudaCache, cm.MemoryUnit = FP8CudaCache, FP8MemoryUnit
    cm.ContextManager.append = append
    cm._baseline_fp8_storage = True


def install_graph_dispatch(model):
    """Capture compute-only MLP/recurrent modules; never capture retrieval."""
    for decoder in model.model.layers:
        for module in (decoder.mlp, getattr(decoder, 'linear_attn', None)):
            if module is None:
                continue
            module._decode_original = module.forward
            module._decode_graph = None
            def dispatch(self, hidden_states, *args, **kwargs):
                graph = self._decode_graph
                if hidden_states.shape[:2] == (1, 1) and graph is not None:
                    if args or kwargs.get('attention_mask') is not None:
                        raise ValueError('Unvalidated graph arguments')
                    if graph['cache'] is not None and kwargs.get('cache_params') is not graph['cache']:
                        raise ValueError('Graph belongs to another recurrent cache')
                    graph['input'].copy_(hidden_states)
                    graph['graph'].replay()
                    return graph['output']
                return self._decode_original(hidden_states, *args, **kwargs)
            module.forward = MethodType(dispatch, module)


@torch.inference_mode()
def prepare_graphs(model, cache):
    checks = []
    for index, decoder in enumerate(model.model.layers):
        for label, module in (('mlp', decoder.mlp), ('recurrence', getattr(decoder, 'linear_attn', None))):
            if module is None:
                continue
            # MLP has no session state and reuses its graph across prefixes.
            if label == 'mlp' and module._decode_graph is not None:
                continue
            module._decode_graph = None
            state = cache.layers[index] if label == 'recurrence' else None
            snapshots = (state.conv_states.clone(), state.recurrent_states.clone()) if state is not None else None
            x = torch.zeros((1, 1, model.config.hidden_size), device='cuda', dtype=torch.bfloat16)
            kwargs = {'cache_params': cache} if state is not None else {}
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    module._decode_original(x, **kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = module._decode_original(x, **kwargs)
            def restore():
                if state is not None:
                    state.conv_states.copy_(snapshots[0])
                    state.recurrent_states.copy_(snapshots[1])
            restore()
            x.normal_(std=.2)
            reference = module._decode_original(x, **kwargs).clone()
            expected = (state.conv_states.clone(), state.recurrent_states.clone()) if state is not None else None
            restore()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
            if state is not None:
                torch.testing.assert_close(state.conv_states, expected[0], rtol=0, atol=0)
                torch.testing.assert_close(state.recurrent_states, expected[1], rtol=0, atol=0)
            restore()
            module._decode_graph = {'graph': graph, 'input': x, 'output': output,
                                    'cache': cache if state is not None else None}
            checks.append({'layer': index, 'component': label, 'output_and_state_exact': True})
    return checks


def release_cache_graphs(model):
    for decoder in model.model.layers:
        module = getattr(decoder, 'linear_attn', None)
        if module is not None and hasattr(module, '_decode_graph'):
            module._decode_graph = None
