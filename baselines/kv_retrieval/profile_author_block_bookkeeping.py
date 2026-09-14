"""CPU-only diagnostic of the exact author metadata function, with inert copies.

This is NOT an InfLLM task, attention benchmark, or estimate of full-model
latency. KV copies/events are inert so only Python metadata organization is
measured. Inputs explicitly state assumed turnover; no observed hit rate is
inferred from them. No CUDA context, model, live process attachment or mutation.
"""
import cProfile
import argparse
import json
from pathlib import Path
import pstats
from types import SimpleNamespace
import time

from kv_baselines.upstream import infllm_module, verify_source

ROOT=Path(__file__).resolve().parent


class InertTensor:
    def __init__(self,length):self.length=length;self.copies=0
    def __getitem__(self,index):return self
    def copy_(self,other,non_blocking=False):self.copies+=1
    def size(self,index):return self.length


class InertUnit:
    def __init__(self):self.loads=0
    def load(self,target):self.loads+=1


def fixture(topk,populated,replacements):
    previous=list(range(populated))
    selected=previous[replacements:]+list(range(populated,populated+replacements))
    if replacements==0:selected=previous
    n=max(populated+replacements,topk)
    units=[InertUnit() for _ in range(n)]
    manager=SimpleNamespace(num_units=1,n_local=4096,n_init=128,
        topk=topk,block_size=128,_global_remainder_ed=4096,_global_remainder_st=0,
        init_k=InertTensor(128),init_v=InertTensor(128),
        global_buffer=InertTensor(128*topk+1024),
        global_remainder=(InertTensor(4097),InertTensor(4097)),
        cached_blocks=[dict.fromkeys(set(previous)|set(selected),1)],
        global_blocks=[units],global_buffer_block_id_list=[previous+[-1]*(topk-populated)],
        global_buffer_init_st=populated*128,global_buffer_init_ed=populated*128+128)
    return manager,[selected]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'reports/author_block_bookkeeping_cpu_20260912_v2.json')
    out=parser.parse_args().out
    if out.exists():raise FileExistsError(out)
    function=infllm_module('context_manager').ContextManager.get_global_hidden_and_mask
    report={'status':'running','started':time.time(),'diagnostic_only':True,
        'no_cuda_context':True,'task_performance_eligible':False,
        'source':verify_source('InfLLM'),
        'function':'ContextManager.get_global_hidden_and_mask',
        'scope':'Exact author Python function; KV copy/event work omitted via inert objects; assumed block turnover, not measured agent turnover.',
        'timing_instrumentation':'Wall timings without profiler; separate cProfile capture for attribution only',
        'cases':[]}
    cases=[('small_unchanged',16,16,0),('small_eight_changes',16,16,8),
           ('large_below_capacity_unchanged',1503,266,0),
           ('large_full_unchanged',1503,1503,0),
           ('large_full_sixteen_changes',1503,1503,16),
           ('large_full_150_changes',1503,1503,150)]
    for name,topk,populated,replacements in cases:
        durations=[];profile=cProfile.Profile();loads=[]
        for repetition in range(7):
            manager,selected=fixture(topk,populated,replacements)
            start=time.perf_counter()
            output=function(manager,1,selected)
            durations.append(time.perf_counter()-start)
            assert set(output[3][0])==set(selected[0])
            assert output[4]==populated
            loads.append(sum(unit.loads for unit in manager.global_blocks[0]))
        assert loads==[replacements]*7
        manager,selected=fixture(topk,populated,replacements)
        profile.runcall(function,manager,1,selected)
        stats=pstats.Stats(profile)
        hot=[]
        for (filename,line,fn),(primitive,total,own,cumulative,callers) in stats.stats.items():
            hot.append({'file':filename,'line':line,'function':fn,'calls':total,
                        'own_s':own,'cumulative_s':cumulative})
        report['cases'].append({'name':name,'topk_capacity':topk,'populated_blocks':populated,
            'assumed_replaced_blocks':replacements,'metadata_wall_s':durations,
            'inert_load_calls':loads,'hot_functions':sorted(hot,key=lambda r:r['own_s'],reverse=True)[:8]})
    report.update(status='passed',finished=time.time())
    # Importing the author module may import torch, but it must not allocate a
    # CUDA context for this diagnostic.
    import torch
    assert not torch.cuda.is_initialized()
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'no_cuda_context':True,
        'cases':[{'name':r['name'],'wall_ms':[round(t*1000,3) for t in r['metadata_wall_s']]} for r in report['cases']]}))


if __name__=='__main__':main()
