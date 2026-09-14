"""Fresh five-length runs, online-query timing, and exact-output diagnostics."""
from pathlib import Path
import argparse
import collections
import copy
import hashlib
import json
import os
import subprocess
import time
import traceback
from types import SimpleNamespace
from probe import Harness, HERE, previous, torch, unique_gqa_rows
from verify import verify, accepted_prefix
from kv_baselines.upstream import infllm_module

LENGTHS = [8192,16384,32768,65536,131072]


class Experiment(Harness):
    def __init__(self, method):
        super().__init__(SimpleNamespace(method=method,window='small',tokens=256,graphs=True))
        self.extra = collections.Counter()
        self.inf_tapes = {}
        self.inf_cursors = collections.Counter()
        self.trace = False
        self.trace_label = 'measured_decode'
        self.trace_active = False
        if not self.backend:
            cls = infllm_module('context_manager').MemoryUnit
            original = cls.load
            def load(unit, target=None):
                tick = time.perf_counter()
                active = self.in_measure
                if active and self.profile:
                    self.extra['load_calls'] += 1
                    self.extra['gpu_hits'] += unit.gpu_data is not None
                    if unit.gpu_data is None:
                        self.extra['miss_source_payload_bytes'] += sum(x.numel()*x.element_size() for x in unit.cpu_data)
                saved = None
                if active and 'gpu_source' in self.mode and hasattr(unit,'_diag_gpu'):
                    saved, unit.cpu_data = unit.cpu_data, unit._diag_gpu
                try:
                    return original(unit,target)
                finally:
                    if saved is not None:unit.cpu_data=saved
                    if active and self.profile:self.extra['load_host_s'] += time.perf_counter()-tick
            cls.load = load

    def prepare(self, length):
        super().prepare(length)
        # Full GPU historical copies are allocated only for capacity diagnostics.
        if self.backend:
            for i in self.backend.modules:self.prepared[1].layers[i]._resident=None
            torch.cuda.empty_cache()
        else:
            for idx, layer in enumerate(self.prepared[1].layers):
                if not hasattr(layer,'manager'):continue
                m=layer.manager;original=m.calc_block_topk
                def topk(query, original=original, m=m, idx=idx):
                    tick=time.perf_counter();n=self.inf_cursors[idx];self.inf_cursors[idx]+=1
                    if self.in_measure and self.mode.startswith('oracle_'):
                        record=self.inf_tapes[(self.prepared[0],self.current_mtp)][idx][n]
                        assert record['length']==m.length
                        result=copy.deepcopy(record['ids'])
                    else:result=original(query)
                    if self.in_measure and self.capture:
                        self.inf_tapes[(self.prepared[0],self.current_mtp)][idx].append(dict(length=m.length,ids=copy.deepcopy(result)))
                    if self.in_measure and self.profile:
                        self.extra['topk_calls']+=1;self.extra['topk_host_s']+=time.perf_counter()-tick
                    return result
                m.calc_block_topk=topk

    def resident(self):
        cache=self.prepared[1]
        if self.backend:
            for idx in self.backend.modules:
                s=cache.layers[idx]
                if s._resident is None:
                    s._resident=torch.empty(s.cpu_kv.shape,dtype=torch.uint8,device='cuda')
                    s._resident[:,:s.length].copy_(s.cpu_kv[:,:s.length],non_blocking=True)
            for key,tapes in self.controller.tapes.items():
                for idx,items in tapes.items():
                    s=cache.layers[idx]
                    for record in items:
                        if 'unique_gpu' in record:continue
                        ids=record['ids'].cpu()
                        unique,inverse=unique_gqa_rows(ids,s.kh,s.qh)
                        record.update(unique_gpu=unique.cuda(),inverse_gpu=inverse.cuda())
        else:
            for layer in cache.layers:
                if not hasattr(layer,'manager'):continue
                m=layer.manager
                for units in m.global_blocks:
                    if not units:continue
                    storage=tuple(torch.stack([u.cpu_data[w] for u in units]).cuda() for w in range(2))
                    for i,u in enumerate(units):u._diag_gpu=tuple(t[i] for t in storage)
        torch.cuda.synchronize()

    def prefix(self,length,prime=True):
        result=super().prefix(length,prime)
        self.extra.clear();self.inf_cursors.clear()
        if self.capture and not self.backend:
            self.inf_tapes[(length,self.current_mtp)]=collections.defaultdict(list)
        if self.trace:
            torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(self.trace_label)
            self.trace_active=True
        return result

    def cleanup(self,cache):
        if self.trace_active:
            torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop();self.trace_active=False
        super().cleanup(cache)
        self.last_profile['infllm_extra']=dict(self.extra)

    def measure_mode(self,*args,**kwargs):
        row=super().measure_mode(*args,**kwargs)
        row['extra_resident_history']='GPU history shadow is allocated/read only in gpu_source diagnostics; absent from formal measurements.'
        return row

    def new_query(self,length,mtp,profile=False):
        self.current_mtp=mtp;self.mode='optimized_both' if self.backend else 'full'
        self.capture=False;self.profile=profile
        if self.backend:self.controller.prepare_query_only()
        cache,logits,hidden,metadata=self.prefix(length)
        query=torch.tensor([json.loads((HERE/'new_query.json').read_text())['input_ids']],device='cuda')
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        if self.backend is None:
            # Supplied user tokens are a prefill chunk, unlike generated MTP
            # candidates. Use the author's normal chunk-level query aggregation.
            # Serial candidate verification remains unchanged below.
            base=self.predictor.length
            logits,target_hidden=self.forward(query,cache)
            if mtp:self.predictor.prime(query,target_hidden,hidden,base)
            hidden=target_hidden[:,-1:].clone()
            query_append='upstream InfLLM append of the whole supplied 40-token query; normal chunk query aggregation'
        else:
            # InfiniGen's finalized adapter accepts single-token selection.
            # Batch dense compute while preserving each causal retrieval.
            for pos in range(0,query.shape[1],5):
                ids=query[:,pos:pos+5];base=self.predictor.length
                logits,target_hidden,tx=verify(self.model,ids,cache,self.backend)
                tx.commit(ids.shape[1])
                if mtp:self.predictor.prime(ids,target_hidden,hidden,base)
                hidden=target_hidden[:,-1:].clone()
                del tx,target_hidden
            query_append='causal batches <=5 through verified adapter; all supplied positions committed'
        seed=logits[:,-1].argmax(-1,keepdim=True);generated=[int(seed)]
        torch.cuda.synchronize();first=time.perf_counter();batches=[]
        while len(generated)<128:
            tick=time.perf_counter()
            if not mtp:
                logits,hidden=self.forward(seed,cache)
                seed=logits[:,-1].argmax(-1,keepdim=True);generated.append(int(seed))
                batches.append(dict(total_s=time.perf_counter()-tick,emitted=1));continue
            count=min(mtp,128-len(generated)-1);base=self.predictor.length
            candidates=self.predictor.propose(seed,hidden,count)
            torch.cuda.synchronize();draft_end=time.perf_counter()
            inputs=torch.cat((seed,candidates),dim=1)
            logits,target_hidden,tx=verify(self.model,inputs,cache,self.backend)
            predictions=logits.argmax(-1)[0].tolist();drafts=candidates[0].tolist()
            accepted=accepted_prefix(drafts,predictions)
            torch.cuda.synchronize();verify_end=time.perf_counter();commit=accepted+1
            tx.commit(commit);self.predictor.length=base
            self.predictor.prime(inputs[:,:commit],target_hidden[:,:commit],hidden,base)
            hidden=target_hidden[:,commit-1:commit].clone()
            generated.extend(drafts[:accepted]+[predictions[accepted]])
            seed=torch.tensor([[generated[-1]]],device='cuda');torch.cuda.synchronize()
            batches.append(dict(drafted=count,accepted=accepted,emitted=commit,
                draft_s=draft_end-tick,verify_s=verify_end-draft_end,
                commit_prime_s=time.perf_counter()-verify_end,total_s=time.perf_counter()-tick))
            del tx,target_hidden
        torch.cuda.synchronize();end=time.perf_counter()
        row=dict(history_tokens=length,prompt_tokens=length+query.shape[1],new_query_tokens=query.shape[1],
            mtp=mtp,outputs=128,output_ids=generated,query_ttft_s=first-started,
            decode_s=end-first,decode_tokens_per_s=127/(end-first),request_s=end-started,
            effective_tokens_per_s=128/(end-started),batches=batches,instrumented=profile,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            query_append=query_append,
            prefix_input_ids_sha256=metadata['input_ids_sha256'])
        self.cleanup(cache);row['profile']=self.last_profile
        return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['infllm','infinigen'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--trace',action='store_true')
    p.add_argument('--trace-query',action='store_true')
    p.add_argument('--query-only',action='store_true')
    p.add_argument('--trace-mtp',type=int,default=0);p.add_argument('--lengths',nargs='+',type=int,default=LENGTHS)
    a=p.parse_args();assert not a.out.exists()
    assert json.loads((HERE/'pause.json').read_text())['status']=='paused'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    os.sched_setaffinity(0,set(range(16))|set(range(32,48)))
    n=8 if a.method=='infinigen' else 2
    torch.set_num_threads(n);torch.set_num_interop_threads(1);torch.manual_seed(0);previous.enable_conv()
    report=dict(status='loading',started=time.time(),method=a.method,threads=n,measurements=[],profiles=[],
                references=[],captures=[],diagnostics=[],new_queries=[],source_hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in HERE.glob('*.py')})
    previous.save(a.out,report);h=None
    try:
        h=Experiment(a.method);report['model']=h.manifest
        if h.backend:
            h.backend.executor.submit(torch.set_num_threads,n).result()
            assert h.backend.executor.submit(torch.get_num_threads).result()==n
        main_mode='optimized_both' if h.backend else 'full'
        def record(group,row,length,mtp,rep=0):
            row.update(rep=rep,threads=n)
            report[group].append(row);previous.save(a.out,report)
            print(json.dumps(dict(group=group,length=length,mtp=mtp,rep=rep,tps=row.get('decode_tokens_per_s'),effective_tps=row.get('effective_tokens_per_s'),mode=row.get('mode'))),flush=True)
        with torch.inference_mode():
            for length in a.lengths:
                report['status']=f'prepare_{length}';previous.save(a.out,report);h.prepare(length)
                if a.trace:
                    if a.trace_query:
                        h.new_query(length,a.trace_mtp)
                        h.trace=True;h.trace_label='measured_query'
                        row=h.new_query(length,a.trace_mtp)
                    else:
                        h.measure_mode(length,a.trace_mtp,32,main_mode)
                        h.trace=True;row=h.measure_mode(length,a.trace_mtp,32,main_mode)
                    record('measurements',row,length,a.trace_mtp);break
                if a.query_only:
                    # Warm the identical append/decoder path before timed runs.
                    reference=h.new_query(length,0);expected=reference['output_ids']
                    record('references',reference,length,0)
                    h.new_query(length,4)
                    for rep in range(3):
                        for mtp in ((0,4) if rep%2==0 else (4,0)):
                            row=h.new_query(length,mtp)
                            row['exact_output_vs_serial_query']=row['output_ids']==expected
                            assert row['exact_output_vs_serial_query'],(length,mtp,'chunk query output mismatch')
                            record('new_queries',row,length,mtp,rep)
                    continue
                reference=h.measure_mode(length,0,128,'full')
                record('references',reference,length,0);expected=reference['output_ids']
                for rep in range(3):
                    for mtp in ((0,4) if rep%2==0 else (4,0)):
                        row=h.measure_mode(length,mtp,128,main_mode)
                        row['exact_output_vs_reference']=row['output_ids']==expected
                        assert row['exact_output_vs_reference'],(length,mtp,'steady output mismatch')
                        record('measurements',row,length,mtp,rep)
                for mtp in (0,4):
                    row=h.measure_mode(length,mtp,128,main_mode,profile=True)
                    assert row['output_ids']==expected
                    record('profiles',row,length,mtp)
                query_expected=None
                for rep in range(3):
                    for mtp in ((0,4) if rep%2==0 else (4,0)):
                        row=h.new_query(length,mtp)
                        if query_expected is None:query_expected=row['output_ids']
                        row['exact_output_vs_serial_query']=row['output_ids']==query_expected
                        assert row['exact_output_vs_serial_query'],(length,mtp,'query output mismatch')
                        record('new_queries',row,length,mtp,rep)
                if length in (32768,131072):
                    # Oracle tapes and full GPU copies are diagnostics only.
                    for mtp in (0,4):
                        row=h.measure_mode(length,mtp,128,main_mode,capture=True)
                        assert row['output_ids']==expected;record('captures',row,length,mtp)
                    h.resident()
                    modes=['optimized_both_gpu_source','oracle_ids_gpu_source_fast'] if h.backend else ['gpu_source','oracle_ids','oracle_ids_gpu_source']
                    for rep in range(3):
                        for mtp in (0,4):
                            for mode in modes[rep:]+modes[:rep]:
                                row=h.measure_mode(length,mtp,128,mode)
                                row['exact_output_vs_reference']=row['output_ids']==expected
                                assert row['exact_output_vs_reference'],(length,mtp,mode,'diagnostic mismatch')
                                record('diagnostics',row,length,mtp,rep)
        report['status']='complete'
    except BaseException as e:
        report.update(status='error',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if h:
            h.dispose()
            if h.backend:h.backend.close()
        report['ended']=time.time();previous.save(a.out,report)

if __name__=='__main__':main()
