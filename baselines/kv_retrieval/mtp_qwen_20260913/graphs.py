"""Capture compute plus bounded recurrent snapshots, never retrieval or IO."""
import torch
from transaction import Snapshot, Transaction
from verify import recurrent


@torch.inference_mode()
def prepare(model, cache):
    checks = []
    for idx, decoder in enumerate(model.model.layers):
        for label, module in (('mlp', decoder.mlp), ('recurrent', getattr(decoder,'linear_attn',None))):
            if module is None:
                continue
            if label == 'mlp' and getattr(module, '_verify_graph', None) is not None:
                continue
            module._verify_graph = None
            original_state = Snapshot(cache.layers[idx]) if label == 'recurrent' else None
            x = torch.randn((1,5,model.config.hidden_size),device='cuda',dtype=torch.bfloat16)*.2
            def run():
                tx = Transaction(cache)
                if label == 'recurrent':
                    y = recurrent(module,x,cache,tx,idx)
                else:
                    y = getattr(module,'_decode_original',module.forward)(x)
                return y,tx.prefixes[idx]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            if original_state:
                original_state.restore()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output, snapshots = run()
            if original_state:
                original_state.restore()
            reference, refs = run()
            reference = reference.clone()
            if original_state:
                original_state.restore()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output,reference,rtol=0,atol=0)
            for got, expected in zip(snapshots, refs):
                torch.testing.assert_close(got.conv,expected.conv,rtol=0,atol=0)
                torch.testing.assert_close(got.recurrent,expected.recurrent,rtol=0,atol=0)
            if original_state:
                original_state.restore()
            module._verify_graph = dict(graph=graph,input=x,output=output,snapshots=snapshots,
                                       cache=cache if label=='recurrent' else None)
            checks.append(dict(layer=idx,component=label,batch=5,output_and_all_prefix_states_exact=True))
    return checks


def release(model):
    for layer in model.model.layers:
        module = getattr(layer,'linear_attn',None)
        if module is not None:
            module._verify_graph = None
