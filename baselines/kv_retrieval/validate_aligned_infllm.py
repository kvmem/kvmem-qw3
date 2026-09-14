"""Independent quantized-weight audit, FP8 retrieval oracle, full model smoke."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from gguf import GGUFReader, GGMLQuantizationType
from safetensors import safe_open

from alignment.q8_model import load_q8_model, undo_layout, gguf_name
from alignment.fp8_kv import encode, install_fp8_storage, DTYPE
from kv_baselines.infllm_adapter import InfLLMConfig, PartialRotaryEmbedding
from kv_baselines.infllm_hybrid import install, forward_text, make_cache
from alignment.session import AlignedSession as InfLLMSession
from kv_baselines.upstream import infllm_module
from run_model_smoke import save

ROOT = Path(__file__).resolve().parent


def weight_audit(directory, gguf):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    config = Qwen3_5TextConfig(**json.loads((directory/'config.json').read_text())['text_config'])
    index = json.loads((directory/'model.safetensors.index.json').read_text())['weight_map']
    reader = GGUFReader(gguf)
    tensors = {t.name:t for t in reader.tensors}
    checks = []
    for suffix in ['in_proj_qkv.weight','in_proj_z.weight','in_proj_a.weight','in_proj_b.weight',
                   'A_log','dt_bias','conv1d.weight','norm.weight','out_proj.weight']:
        name = 'model.layers.0.linear_attn.' + suffix
        src = name.replace('model.', 'model.language_model.', 1)
        with safe_open(directory/index[src], framework='pt', device='cpu') as f:
            expected = f.get_tensor(src).float().numpy()
        tensor = tensors[gguf_name(name)]
        quant = tensor.tensor_type == GGMLQuantizationType.Q8_0
        actual = undo_layout(tensor.data, name, expected.shape, config, quant)
        if quant:
            b = actual.reshape(expected.shape[0], -1, 34)
            d = b[...,:2].copy().view(np.float16).astype(np.float32)
            actual = (d * b[...,2:].copy().view(np.int8)).reshape(expected.shape)
        error = float(np.sqrt(np.mean((actual-expected)**2)))
        relative = error/max(float(np.sqrt(np.mean(expected**2))), 1e-8)
        if relative > (.02 if quant else 2e-5):
            raise AssertionError(f'GGUF -> HF mapping mismatch {name}: {relative}')
        checks.append({'name':name,'rmse':error,'relative_rmse':relative})
    return checks


def fp8_oracle():
    torch.manual_seed(1821)
    cfg = InfLLMConfig(n_init=16, n_local=64, topk=2, max_cached_block=4,
        block_size=32, exc_block_size=32, async_global_stream=True)
    q = torch.randn(1,24,337,256, device='cuda', dtype=torch.float16)*.25
    k,v = [torch.randn(1,4,337,256,device='cuda',dtype=torch.float16)*.25 for _ in range(2)]
    # The unmodified FP16 author sees exactly representable quantized values.
    k,v = [encode(t).half() for t in (k,v)]
    spans = [(0,256),(256,288),(288,320),(320,321),(321,337)]
    ref = cfg.create(PartialRotaryEmbedding(64,1e7))
    expected, selected = [], []
    for a,b in spans:
        qq,kk,vv = [t[:,:,a:b,:].contiguous() for t in (q,k,v)]
        expected.append(ref.append(qq,kk,vv,qq,kk,vv).clone())
        selected.append([list(x) for x in ref.global_buffer_block_id_list])
    install_fp8_storage()
    tested = cfg.create(PartialRotaryEmbedding(64,1e7))
    max_error = 0
    for (a,b),want,ids in zip(spans,expected,selected):
        qq,kk,vv = [t[:,:,a:b,:].contiguous() for t in (q,k,v)]
        got = tested.append(qq,kk,vv,qq,kk,vv)
        torch.testing.assert_close(got,want,rtol=0,atol=0)
        assert tested.global_buffer_block_id_list == ids
        max_error = max(max_error,float((got-want).abs().max()))
    assert tested.local_k.dtype == DTYPE and tested.cuda_cache.data.dtype == DTYPE
    assert all(t.dtype == DTYPE for u in tested.global_blocks[0] for t in u.cpu_data)
    # Independent SATFINITE edge values (PyTorch's unclamped cast yields NaN).
    torch.testing.assert_close(encode(torch.tensor([-999.,-448.,0.,448.,999.],device='cuda')).float(),
        torch.tensor([-448.,-448.,0.,448.,448.],device='cuda'),rtol=0,atol=0)
    return {'max_output_error':max_error,'retrieved_ids_identical':True,
            'global_blocks':tested.num_global_block,'cpu_local_gpu_cache_dtype':str(DTYPE)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT.parents[1]/'models/Qwen3.8-27B-BF16')
    p.add_argument('--gguf',type=Path,default=ROOT.parents[1]/'models/Qwen3.8-27B-Q8_0.gguf')
    p.add_argument('--component-only',action='store_true')
    args=p.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    report={'status':'running','started':time.time(),'performance_eligible':False,'shared_gpu':True}
    save(args.out,report)
    torch.set_num_threads(4)
    try:
        report['weight_mapping']=weight_audit(args.model,args.gguf)
        with torch.inference_mode():
            report['fp8_oracle']=fp8_oracle()
            save(args.out,report)
            if not args.component_only:
                gc.collect(); torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(58*2**30/torch.cuda.get_device_properties(0).total_memory)
                model,manifest=load_q8_model(args.model,args.gguf)
                report['model']=manifest
                from kv_baselines.frontend import NativeBridge
                bridge=NativeBridge(args.model)
                ids=bridge.call('tokenize',text='<|im_start|>user\nWrite a Python function that adds two integers.<|im_end|>\n<|im_start|>assistant\n<think>\n')['input_ids']
                from transformers import DynamicCache
                dense=model(input_ids=torch.tensor([ids],device='cuda'),past_key_values=DynamicCache(config=model.config),use_cache=True,logits_to_keep=1).logits
                install(model)
                session=InfLLMSession(model,InfLLMConfig(),prefill_chunk=512)
                got,reuse=session.begin(ids)
                report['short_dense_argmax']=int(dense[0,-1].argmax())
                report['short_infllm_argmax']=int(got[0].argmax())
                report['short_logit_rmse']=float((dense[0,-1]-got[0]).square().mean().sqrt())
                if report['short_dense_argmax']!=report['short_infllm_argmax']:
                    raise AssertionError('short-context argmax mismatch')
                generated=[]
                for _ in range(24):
                    token=int(got.argmax())
                    generated.append(token)
                    got=session.consume_generated_token(token)
                report['short_generation_ids']=generated
                from transformers import AutoTokenizer
                report['short_generation_text']=AutoTokenizer.from_pretrained(args.model,local_files_only=True).decode(generated)
                del session,dense,got
                gc.collect(); torch.cuda.empty_cache()
                # Allocate every layer's full large-window buffers, using real
                # model output, then exercise a safe checkpoint and tail rewrite.
                big=InfLLMConfig(**json.loads((ROOT/'configs/infllm_192k_fp8.json').read_text()))
                session=InfLLMSession(model,big,prefill_chunk=32)
                text=(ROOT.parents[1]/'src/model_source.cpp').read_text()
                tokens=bridge.call('tokenize',text=text)['input_ids'][:227]
                session.begin(tokens[:192])
                for token in tokens[192:195]: session.consume_generated_token(token)
                session.begin(tokens[:216])
                session.consume_generated_token(tokens[216])
                rewritten=tokens[:211]+list(reversed(tokens[211:]))
                actual,reuse=session.begin(rewritten)
                assert reuse['resume_origin']==160 and not reuse['cold_prefill']
                saved=actual.detach().cpu()
                del session,actual
                gc.collect(); torch.cuda.empty_cache()
                fresh=InfLLMSession(model,big,prefill_chunk=32)
                expected,_=fresh.begin(rewritten)
                torch.testing.assert_close(saved,expected.cpu(),rtol=0,atol=0)
                report['large_window']={'active_capacity':big.n_init+big.n_local+big.topk*big.block_size,
                    'tail_rewrite':reuse,'tail_logits_bitwise_equal':True,
                    'reserved_gib':torch.cuda.memory_reserved()/2**30}
        report['status']='passed'
    except BaseException as exc:
        import traceback
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished=time.time(),peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_initialized() else 0)
        save(args.out,report)
        print(json.dumps({k:v for k,v in report.items() if k!='model'}),flush=True)


if __name__=='__main__': main()
