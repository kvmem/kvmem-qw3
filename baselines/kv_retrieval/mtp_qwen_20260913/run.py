"""Exclusive GPU validation and throughput harness for external Q8 MTP."""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from common import HERE, BASE, ROOT, CAMPAIGN, MODEL, WEIGHTS
import torch
from alignment.q8_model import load_q8_model
from alignment.native_q8 import NativeQ8
from kv_baselines.infllm_adapter import InfLLMConfig
from kv_baselines.infllm_hybrid import install, make_cache
from qwen.backend import Config
from qwen.optimized import OptimizedBackend
from fastpaths import enable_conv, install_fp8_views, install_graph_dispatch, prepare_graphs, release_cache_graphs
from draft import Predictor
from verify import verify, accepted_prefix
import graphs
from checks import retrieval_state


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n')
    tmp.replace(path)


class Harness:
    def __init__(self, args):
        self.args = args
        self.model, self.manifest = load_q8_model(MODEL, WEIGHTS)
        library = HERE/'q8_consistent/libconsistent_q8.so'
        assert library.exists(), 'Build the isolated consistent-reduction kernels first'
        api = NativeQ8(library)
        for module in self.model.modules():
            if hasattr(module, 'api'):
                module.api = api
        self.manifest['q8_compute_library'] = str(library)
        self.manifest['q8_compute_library_sha256'] = hashlib.sha256(library.read_bytes()).hexdigest()
        self.manifest['q8_change'] = 'same 4-warp reduction and explicit rounded sum for scalar decode and batched verify; Q8 weights unchanged'
        self.predictor = Predictor(self.model).eval()
        # torch.mean selects a different reduction for a wider row count. At
        # BF16 rounding ties this can change retrieval downstream. Keep each
        # token's RMSNorm reduction identical to the single-token reference.
        for module in {*self.model.modules(), *self.predictor.modules()}:
            if type(module).__name__ == 'Qwen3_5RMSNorm':
                original = module.forward
                def norm(x, original=original):
                    if x.ndim >= 3 and x.shape[0] == 1 and 2 <= x.shape[1] <= 5:
                        return torch.cat([original(x[:,i:i+1]) for i in range(x.shape[1])],dim=1)
                    return original(x)
                module.forward = norm
        self.manifest['norm_change'] = 'single-token RMSNorm reductions for 2..5-token verification/prime'
        self.backend = None
        self.config = None
        self.handle = self.model.model.norm.register_forward_pre_hook(self.hidden_hook)
        self.hidden = None
        if args.method == 'infllm':
            install_fp8_views()
            install(self.model)
            topk = 16 if args.window == 'small' else 991
            self.config = InfLLMConfig(topk=topk, max_cached_block=2*topk, async_global_stream=False)
            if topk > 16:
                import sys
                sys.path.insert(0, str(CAMPAIGN/'qwen27'))
                from fast_block_map import install as fast_map_install
                fast_map_install()
        else:
            self.backend = OptimizedBackend(self.model)
            calibration = HERE/'q8_rotations.pt'
            if calibration.exists():
                self.backend.rotations = {i:t.cuda() for i,t in torch.load(calibration,weights_only=True).items()}
            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
                pg19 = BASE/'infinigen/InfiniGen/accuracy/setup/pg19_firstbook.txt'
                ids = tokenizer.encode(pg19.read_text(),add_special_tokens=False)[:2048]
                cache = self.backend.cache(Config(capacity=2048,dense=True,collect_calibration=True))
                self.backend.forward(torch.tensor([ids],device='cuda'),cache)
                self.backend.calibrate(cache)
                torch.save({i:t.cpu() for i,t in self.backend.rotations.items()},calibration)
                self.hidden = None
                del cache, tokenizer
                gc.collect(); torch.cuda.empty_cache()
        if args.graphs:
            install_graph_dispatch(self.model)
        source = json.loads((CAMPAIGN/'inputs.json').read_text())
        self.history = source['evaluation']['input_ids']
        self.suffix = sum((source['suffix'][k] for k in ('before','query','after')), [])

    def hidden_hook(self, module, inputs):
        self.hidden = inputs[0]

    def forward(self, ids, cache):
        if self.backend:
            logits = self.backend.forward(ids, cache)
        else:
            logits = self.model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1).logits
        return logits, self.hidden

    def prefix(self, length, prime=True):
        ids = self.history[:length-len(self.suffix)] + self.suffix
        cache = make_cache(self.model.config,self.config) if self.backend is None else self.backend.cache(
            Config(capacity=length+self.args.tokens+32,max_num_kv=2048 if self.args.window=='small' else 131072))
        self.predictor.reset(length+self.args.tokens+32)
        previous = torch.zeros((1,1,self.model.config.hidden_size),device='cuda',dtype=torch.bfloat16)
        tick = time.perf_counter()
        for start in range(0,length,2048):
            chunk = torch.tensor([ids[start:start+2048]],device='cuda')
            logits, hidden = self.forward(chunk,cache)
            if prime:
                self.predictor.prime(chunk,hidden,previous,start)
            previous = hidden[:, -1:].clone()
        if self.backend:
            self.backend.finalize(cache)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter()-tick
        tick = time.perf_counter()
        checks = prepare_graphs(self.model,cache) if self.args.graphs else []
        if self.args.graphs:
            checks += graphs.prepare(self.model,cache)
        return cache, logits, previous, dict(prefill_and_prime_s=prefill_s,
            input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),graph_checks=checks,
            graph_setup_s=time.perf_counter()-tick)

    def cleanup(self, cache):
        release_cache_graphs(self.model)
        graphs.release(self.model)
        self.hidden = None
        self.predictor.kv = None
        del cache
        gc.collect(); torch.cuda.empty_cache()

    def measure(self, length, mtp, outputs):
        cache, logits, hidden, row = self.prefix(length, prime=bool(mtp))
        seed = logits[:, -1].argmax(-1,keepdim=True)
        generated = [int(seed)]
        batches = []
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        while len(generated) < outputs:
            begin = time.perf_counter()
            if not mtp:
                logits, hidden = self.forward(seed,cache)
                seed = logits[:, -1].argmax(-1,keepdim=True)
                generated.append(int(seed))
                batches.append(dict(total_s=time.perf_counter()-begin,emitted=1))
                continue
            count = min(mtp, outputs-len(generated)-1)
            base = self.predictor.length
            candidates = self.predictor.propose(seed,hidden,count)
            torch.cuda.synchronize(); draft_done = time.perf_counter()
            inputs = torch.cat((seed,candidates),dim=1)
            logits, target_hidden, tx = verify(self.model,inputs,cache,self.backend)
            predictions = logits.argmax(-1)[0].tolist()
            drafts = candidates[0].tolist()
            accepted = accepted_prefix(drafts,predictions)
            torch.cuda.synchronize(); verify_done = time.perf_counter()
            commit = accepted+1
            tx.commit(commit)
            self.predictor.length = base
            self.predictor.prime(inputs[:, :commit],target_hidden[:, :commit],hidden,base)
            hidden = target_hidden[:, commit-1:commit].clone()
            generated.extend(drafts[:accepted]+[predictions[accepted]])
            seed = torch.tensor([[generated[-1]]],device='cuda')
            torch.cuda.synchronize()
            batches.append(dict(drafted=count,accepted=accepted,emitted=commit,
                draft_s=draft_done-begin,verify_s=verify_done-draft_done,
                commit_prime_s=time.perf_counter()-verify_done,total_s=time.perf_counter()-begin))
            del tx, target_hidden
        torch.cuda.synchronize(); elapsed = time.perf_counter()-tick
        row.update(prompt_tokens=length,mtp=mtp,outputs=outputs,output_ids=generated,
            decode_s=elapsed,decode_tokens_per_s=(outputs-1)/elapsed,batches=batches,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            draft_cache='full true-position FP8 GPU history; main target retains method-specific retrieval')
        self.cleanup(cache)
        return row

    def validate(self):
        # Cross the 128-token block boundary after a two-token append.
        length = self.args.validation_length
        teacher = torch.tensor([self.history[9000:9005]],device='cuda')
        refs = []
        cache, _, _, _ = self.prefix(length,prime=False)
        for i in range(5):
            logits, hidden = self.forward(teacher[:, i:i+1],cache)
            refs.append((logits[:, -1].clone(),hidden.clone()))
        reference_retrieval = retrieval_state(cache)
        reference_states = []
        for layer in cache.layers:
            if hasattr(layer,'conv_states'):
                reference_states.append((layer.conv_states.clone(),layer.recurrent_states.clone()))
            else:
                reference_states.append(None)
        self.cleanup(cache); del cache
        result = []
        for keep in (1, 3, 5):
            cache, _, _, _ = self.prefix(length,prime=False)
            logits, hidden, tx = verify(self.model,teacher,cache,self.backend)
            reference_logits = torch.stack([v[0] for v in refs],dim=1)
            error = float((logits-reference_logits).abs().max())
            token_match = torch.equal(logits.argmax(-1),reference_logits.argmax(-1))
            tx.commit(keep)
            # Continue rejected positions through the serial path. A corrupt
            # recurrent/selection state is visible here, not just in lengths.
            continuation_errors = []
            for i in range(keep,5):
                out, _ = self.forward(teacher[:, i:i+1],cache)
                continuation_errors.append(float((out[:, -1]-refs[i][0]).abs().max()))
                assert torch.equal(out[:, -1].argmax(-1),refs[i][0].argmax(-1))
            state_error = 0.
            for layer, ref in zip(cache.layers,reference_states):
                if ref:
                    state_error = max(state_error,float((layer.conv_states-ref[0]).abs().max()),
                                      float((layer.recurrent_states-ref[1]).abs().max()))
            row = dict(keep=keep,logits_max_abs_error=error,token_match=token_match,
                       continuation_errors=continuation_errors,recurrent_max_abs_error=state_error,
                       cache_length=cache.get_seq_length())
            row['retrieval_state_exact'] = retrieval_state(cache)==reference_retrieval
            result.append(row)
            print(json.dumps(dict(validation=row)),flush=True)
            # Start with strict equality; investigate any failure before loosening.
            assert token_match and error == 0 and state_error == 0 and not any(continuation_errors), row
            assert cache.get_seq_length() == length+5
            assert row['retrieval_state_exact'], row
            del tx, logits, hidden
            self.cleanup(cache); del cache
        return result


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method',choices=['infllm','infinigen'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--lengths',nargs='+',type=int,default=[8192,32768,122880,245760])
    p.add_argument('--tokens',type=int,default=128)
    p.add_argument('--reps',type=int,default=3)
    p.add_argument('--mtp',nargs='+',type=int,choices=[0,4],default=[0,4])
    p.add_argument('--window',choices=['small','128k'],default='small')
    p.add_argument('--graphs',action='store_true')
    p.add_argument('--validate',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--validation-length',type=int,default=8190)
    a = p.parse_args()
    assert not a.out.exists()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(0)
    enable_conv()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    report = dict(status='loading',args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        started=time.time(),results=[],cpu_affinity=sorted(os.sched_getaffinity(0)),
        timing='(outputs-1)/full decode wall; includes draft, all retrieval, transfer, snapshots, rejected verification and commit/prime',
        source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob('*.py')})
    save(a.out,report)
    h = None
    try:
        h = Harness(a)
        report.update(model=h.manifest,mtp_mapping=h.predictor.mapping,status='validating' if a.validate else 'measuring')
        save(a.out,report)
        if a.validate:
            report['validation'] = h.validate(); save(a.out,report)
        if not a.validate_only:
            for length in a.lengths:
                for rep in range(a.reps):
                    for mtp in a.mtp if rep%2==0 else list(reversed(a.mtp)):
                        report['status']=f'measuring_{length}_rep{rep}_mtp{mtp}';save(a.out,report)
                        row = h.measure(length,mtp,a.tokens);row['rep']=rep
                        report['results'].append(row);save(a.out,report)
                        paired = [r for r in report['results'] if r['rep']==rep and r['prompt_tokens']==length and r['mtp']!=mtp]
                        if paired:
                            row['same_output_as_other_mtp_setting'] = row['output_ids']==paired[0]['output_ids']
                            save(a.out,report)
                            assert row['same_output_as_other_mtp_setting'], 'MTP altered the greedy target sequence'
                        print(json.dumps({k:row[k] for k in ('prompt_tokens','mtp','rep','decode_tokens_per_s')}),flush=True)
        report['status']='complete'
    except BaseException as exc:
        report.update(status='error',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        if h and h.backend:
            h.backend.close()
        report['ended']=time.time();save(a.out,report)


if __name__=='__main__':
    main()
