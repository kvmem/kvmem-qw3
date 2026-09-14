"""Synthetic component correctness only; never task/throughput evidence."""
import importlib.util
import json
from pathlib import Path
import time
import unittest
import torch
from alignment.fp8_kv import install_fp8_storage, DTYPE
from kv_baselines.infllm_adapter import InfLLMConfig, PartialRotaryEmbedding
from run_model_smoke import save

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'reports/fp8_long_window_components_20260912.json'
    if out.exists(): raise FileExistsError(out)
    report={'status':'running','synthetic_component_only':True,'performance_eligible':False,'started':time.time()}
    save(out,report)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.06)
    install_fp8_storage()
    try:
        with torch.inference_mode():
            spec=importlib.util.spec_from_file_location('original_fixtures',ROOT/'tests/test_infllm_modules.py')
            test=importlib.util.module_from_spec(spec); spec.loader.exec_module(test)
            test.ASYNC_GLOBAL_STREAM=True
            suite=unittest.TestSuite(test.GPUChecks(name) for name in [
                'test_tail_rewrite_restores_hybrid_state_and_history',
                'test_session_short_append_preserves_safe_checkpoint'])
            result=unittest.TextTestRunner(verbosity=2).run(suite)
            if not result.wasSuccessful(): raise AssertionError('FP8 hybrid recovery fixture failed')
            report['hybrid_recovery']=test.METRICS
            cfg=InfLLMConfig(**json.loads((ROOT/'configs/infllm_192k_fp8.json').read_text()))
            manager=cfg.create(PartialRotaryEmbedding(64,1e7))
            torch.manual_seed(731)
            total=202752
            for start in range(0,total,512):
                q,k,v=[torch.randn(1,h,512,256,device='cuda',dtype=torch.float16)*.25 for h in [24,4,4]]
                output=manager.append(q,k,v,q,k,v)
                if start%32768==0:
                    assert torch.isfinite(output).all()
                    report['processed_tokens']=manager.length
                    save(out,report)
            assert manager.num_global_block>cfg.topk
            assert len(manager.global_buffer_block_id_list[0])==cfg.topk
            assert len(set(manager.global_buffer_block_id_list[0]))==cfg.topk
            assert min(manager.global_buffer_block_id_list[0])>=0
            assert manager.local_k.dtype==DTYPE
            assert len(manager.cached_blocks[0])<=cfg.max_cached_block
            report.update(status='passed',processed_tokens=manager.length,
                global_blocks=manager.num_global_block,selected_blocks=cfg.topk,
                cached_blocks=len(manager.cached_blocks[0]),active_capacity=cfg.n_init+cfg.n_local+cfg.topk*cfg.block_size)
    except BaseException as exc:
        import traceback
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished=time.time(),peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        save(out,report)
        print(json.dumps(report))


if __name__=='__main__': main()
