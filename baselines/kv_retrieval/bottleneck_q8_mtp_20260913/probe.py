"""Same-output causal diagnostics of the external Qwen InfiniGen adapter."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'mtp_qwen_20260913'
sys.path.insert(0, str(PRIOR))
import run as previous
from transaction import Snapshot
import infinigen_pipeline
from qwen.backend import Layer, encode, speculate_attention_from_query
from qwen.optimized import unique_gqa_rows
from alignment.q8_model import Q8Linear
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
import torch
import time
import collections
import types
import gc


class FutureProbe:
    def __init__(self, future, controller):
        self.future, self.controller = future, controller

    def result(self):
        begin = time.perf_counter()
        result = self.future.result()
        self.controller.consumer_waits.append(time.perf_counter() - begin)
        ready = result[0]
        stream = torch.cuda.current_stream()
        start, end = self.controller.events()
        start.record(stream)
        stream.wait_event(ready)
        end.record(stream)
        self.controller.device_ranges.append(('consumer_gpu_wait', start, end))
        return result


class Controller:
    def __init__(self, backend):
        self.backend = backend
        self.mode = 'full'
        self.capture = self.profile = self.active = False
        self.tapes = {}
        self.query_only = {}
        self.counters = collections.defaultdict(int)
        self.metrics, self.device_ranges, self.consumer_waits = [], [], []
        self.original_forecast = backend._forecast
        backend._forecast = self.forecast
        original_schedule = backend.schedule

        def schedule(idx, state, hidden, positions):
            original_schedule(idx, state, hidden, positions)
            if self.active and self.profile:
                state.future = FutureProbe(state.future, self)
        backend.schedule = schedule
        original_consume = Layer.consume

        def consume(state, q, k, v, scaling):
            base = state.length
            output = original_consume(state, q, k, v, scaling)
            if self.active and 'gpu_source' in self.mode:
                packed = torch.stack((encode(k), encode(v))).view(torch.uint8)[:, 0].transpose(1, 2)
                state._resident[:, base:base+1].copy_(packed)
            return output
        Layer.consume = consume
        original_enqueue = infinigen_pipeline.enqueue

        def enqueue(backend, module, state, previous_norm, rope, k, v):
            if self.active and 'gpu_source' in self.mode:
                packed = torch.stack((encode(k), encode(v))).view(torch.uint8)[:, 0].transpose(1, 2)
                state._resident[:, state.length:state.length+k.shape[-2]].copy_(packed)
            futures, qk, qv = original_enqueue(backend, module, state, previous_norm, rope, k, v)
            if self.active and self.profile:
                futures = [FutureProbe(f, self) for f in futures]
            return futures, qk, qv
        infinigen_pipeline.enqueue = enqueue

    @staticmethod
    def events():
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def begin(self, key, mode, capture=False, profile=False):
        self.key, self.mode, self.capture, self.profile = key, mode, capture, profile
        self.active = True
        self.counters.clear()
        self.metrics, self.device_ranges, self.consumer_waits = [], [], []
        if capture:
            self.tapes[key] = collections.defaultdict(list)
        if profile:
            self.anchor = torch.cuda.Event(enable_timing=True)
            self.anchor.record()

    def range_start(self):
        if not self.profile:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record(self.backend.stream)
        return event

    def range_end(self, name, start):
        if start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record(self.backend.stream)
            self.device_ranges.append((name, start, end))

    @torch.inference_mode()
    def forecast(self, module, state, hidden, positions, main_ready):
        if not self.active or (self.mode == 'full' and not self.capture and not self.profile):
            return self.original_forecast(module, state, hidden, positions, main_ready)
        idx = module.layer_idx
        number = self.counters[idx]
        self.counters[idx] += 1
        tape = None
        if self.mode.startswith('oracle_'):
            tape = self.tapes[self.key][idx][number]
            assert tape['length'] == state.length, (idx, number, tape['length'], state.length)
        start = time.perf_counter()
        with torch.cuda.stream(self.backend.stream):
            self.backend.stream.wait_event(main_ready)
            hidden.record_stream(self.backend.stream)
            for tensor in positions:
                tensor.record_stream(self.backend.stream)
            event = self.range_start()
            if self.mode.startswith('oracle_ids'):
                partial_q = None
            elif self.mode == 'oracle_query':
                partial_q = tape['partial_q']
            else:
                shape = hidden.shape[:-1]
                if self.mode in ('optimized_q_only', 'optimized_both'):
                    q = self.query_only[idx](hidden).view(*shape, -1, module.head_dim)
                else:
                    q, _ = module.q_proj(hidden).view(*shape, -1, module.head_dim * 2).chunk(2, -1)
                q = module.q_norm(q).transpose(1, 2)
                q, _ = apply_rotary_pos_emb(q, q, *positions)
                partial_q = torch.bmm(q[0].half(), state.basis) * module.scaling
            self.range_end('query_projection_stream', event)
            query_done = time.perf_counter()
            event = self.range_start()
            gpu_bookkeeping = self.mode in ('optimized_bookkeeping', 'optimized_both')
            if self.mode.startswith('oracle_ids'):
                ids = tape['ids']
                ids_ready = sort_done = time.perf_counter()
            else:
                ids = speculate_attention_from_query(partial_q,
                    state.partial_keys[:, :state.length].permute(1, 0, 2),
                    state.config.alpha, state.config.max_num_kv)
                if gpu_bookkeeping:
                    # Integer sorting/deduplication is exact. Preserve each
                    # head's chronological attention order and all selections.
                    ids = ids.sort(dim=0).values
                    mapped = torch.arange(state.qh, device=ids.device) // (state.qh // state.kh)
                    flat = (ids[:, 0] * state.kh + mapped).reshape(-1)
                    unique_gpu, inverse = torch.unique(flat, sorted=True, return_inverse=True)
                    inverse = inverse.to(torch.int32)
                    unique = unique_gpu.cpu()
                else:
                    ids = ids.cpu()
                ids_ready = time.perf_counter()
                self.range_end('selector_and_index_sync_stream', event)
                event = None
                if not gpu_bookkeeping:
                    ids = ids.sort(dim=0).values
                sort_done = time.perf_counter()
            if event is not None:
                self.range_end('selector_and_index_sync_stream', event)
            if self.capture:
                self.tapes[self.key][idx].append({'length': state.length,
                    'ids': ids.clone(), 'partial_q': partial_q.clone()})
            count = ids.shape[0]
            if not gpu_bookkeeping:
                unique, inverse = unique_gqa_rows(ids, state.kh, state.qh)
            unique_done = time.perf_counter()
            cpu_gather_s = h2d_submit_s = 0.
            rows = []
            if 'gpu_source' in self.mode:
                event = self.range_start()
                indices_gpu = unique.to(hidden.device, non_blocking=True)
                self.range_end('h2d_stream', event)
                event = self.range_start()
                for which in range(2):
                    rows.append(torch.index_select(state._resident[which].view(-1, state.dim), 0, indices_gpu))
                self.range_end('gpu_source_gather_stream', event)
            else:
                for which in range(2):
                    tick = time.perf_counter()
                    stage = state.staging[which].view(-1, state.dim)[:len(unique)]
                    torch.index_select(state.cpu_kv[which].view(-1, state.dim), 0, unique, out=stage)
                    cpu_gather_s += time.perf_counter() - tick
                    event = self.range_start()
                    tick = time.perf_counter()
                    rows.append(stage.to(hidden.device, non_blocking=True))
                    h2d_submit_s += time.perf_counter() - tick
                    self.range_end('h2d_stream', event)
            event = self.range_start()
            lookup = inverse.to(hidden.device, non_blocking=True)
            self.range_end('h2d_stream', event)
            event = self.range_start()
            selected = torch.stack([torch.index_select(row, 0, lookup).view(count, state.qh, state.dim)
                                    for row in rows])
            self.range_end('gpu_reconstruction_stream', event)
            ready = torch.cuda.Event()
            ready.record(self.backend.stream)
        stats = {'history_tokens': state.length, 'selected_tokens_per_head': count,
            'unique_kv_rows': len(unique), 'h2d_bytes':
                (unique.numel()*unique.element_size() if 'gpu_source' in self.mode else 2*len(unique)*state.dim)
                + (0 if gpu_bookkeeping else inverse.numel()*inverse.element_size()),
            'forecast_until_indices_cpu_s': sort_done-start,
            'gather_and_enqueue_host_s': time.perf_counter()-sort_done,
            'worker_host_s': time.perf_counter()-start,
            'query_enqueue_host_s': query_done-start,
            'selector_and_indices_sync_host_s': ids_ready-query_done,
            'sort_host_s': sort_done-ids_ready,
            'dedup_host_s': unique_done-sort_done,
            'cpu_gather_host_s': cpu_gather_s, 'h2d_submit_host_s': h2d_submit_s,
            'worker_torch_threads': torch.get_num_threads()}
        # Kept separately from state.metrics so rejected MTP positions remain.
        self.metrics.append(dict(stats, layer=idx, call=number))
        return ready, selected, stats

    def finish(self):
        torch.cuda.synchronize()
        ranges = []
        for label, start, end in self.device_ranges:
            ranges.append({'phase': label, 'start_ms': self.anchor.elapsed_time(start),
                           'end_ms': self.anchor.elapsed_time(end), 'duration_ms': start.elapsed_time(end)})
        self.active = False
        return {'forecasts': self.metrics, 'gpu_stream_ranges': ranges,
            'consumer_host_wait_s': self.consumer_waits,
            'includes_rejected_speculation': True,
            'warning': 'Host and GPU-stream spans overlap; they are not exclusive kernel times or additive wall fractions.'}

    def prepare_query_only(self):
        if self.query_only:
            return
        for idx, module in self.backend.modules.items():
            original = module.q_proj
            packed = original.packed_weight
            selected = packed.view(-1, module.head_dim*2, packed.shape[-1])[:, :module.head_dim].reshape(-1, packed.shape[-1]).contiguous()
            projection = Q8Linear(selected, original.in_features, original.api, original.output_dtype)
            x = torch.randn((1,1,original.in_features), device='cuda', dtype=torch.bfloat16)*.1
            expected = original(x).view(1,1,-1,module.head_dim*2).chunk(2,-1)[0]
            actual = projection(x).view_as(expected)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.query_only[idx] = projection


class Harness(previous.Harness):
    def __init__(self, args):
        super().__init__(args)
        self.controller = Controller(self.backend) if self.backend else None
        self.prepared = None
        self.mode = 'full'
        self.capture = self.profile = False
        self.infllm_stats = collections.Counter()
        self.in_measure = False

    def prepare(self, length):
        if self.prepared:
            self.dispose()
        cache, logits, hidden, metadata = super().prefix(length, prime=True)
        if self.controller:
            for idx in self.backend.modules:
                state = cache.layers[idx]
                state._resident = torch.empty(state.cpu_kv.shape, dtype=torch.uint8, device='cuda')
                state._resident[:, :state.length].copy_(state.cpu_kv[:, :state.length], non_blocking=True)
        else:
            for layer in cache.layers:
                if not hasattr(layer, 'manager'):
                    continue
                manager = layer.manager
                original = manager.calc_block_topk
                def topk(query, original=original, manager=manager):
                    result = original(query)
                    if self.in_measure and self.profile:
                        for unit, blocks in enumerate(result):
                            for block in blocks:
                                self.infllm_stats['selected_blocks'] += 1
                                if block in manager.global_buffer_block_id_list[unit]:
                                    self.infllm_stats['in_attention_buffer'] += 1
                                if manager.global_blocks[unit][block].gpu_data is not None:
                                    self.infllm_stats['gpu_resident'] += 1
                                else:
                                    self.infllm_stats['cpu_only'] += 1
                    return result
                manager.calc_block_topk = topk
        torch.cuda.synchronize()
        self.prepared = (length, cache, logits.clone(), hidden.clone(), metadata)
        self.snapshots = [Snapshot(layer) for layer in cache.layers]
        self.base_predictor_length = self.predictor.length

    def prefix(self, length, prime=True):
        assert self.prepared and self.prepared[0] == length
        _, cache, logits, hidden, metadata = self.prepared
        self.in_measure = False
        if self.controller:
            self.controller.active = False
        for snapshot in self.snapshots:
            snapshot.restore()
        self.predictor.length = self.base_predictor_length
        torch.cuda.synchronize()
        if self.controller:
            self.controller.begin((length, self.current_mtp), self.mode, self.capture, self.profile)
        self.infllm_stats.clear()
        self.in_measure = True
        return cache, logits, hidden, {k:v for k,v in metadata.items() if k != 'graph_checks'}

    def cleanup(self, cache):
        self.in_measure = False
        self.last_profile = self.controller.finish() if self.controller else {'infllm_cache': dict(self.infllm_stats)}
        if self.controller and not self.last_profile['forecasts']:
            self.last_profile['committed_forecasts_only'] = [dict(r, layer=idx)
                for idx in self.backend.modules for r in cache.layers[idx].metrics]

    def measure_mode(self, length, mtp, outputs, mode='full', capture=False, profile=False):
        self.current_mtp, self.mode, self.capture, self.profile = mtp, mode, capture, profile
        if self.controller and mode in ('optimized_q_only', 'optimized_both'):
            self.controller.prepare_query_only()
        row = super().measure(length, mtp, outputs)
        if self.controller and not profile:
            raw = self.last_profile.get('forecasts') or self.last_profile.get('committed_forecasts_only', [])
            self.last_profile = {'forecast_count': len(raw),
                'scope': 'committed_only' if 'committed_forecasts_only' in self.last_profile else 'all_including_rejected',
                'total_h2d_bytes': sum(x.get('h2d_bytes',0) for x in raw),
                'mean_selected_tokens_per_head': sum(x['selected_tokens_per_head'] for x in raw)/len(raw) if raw else None,
                'host_sums': {key: sum(x.get(key,0) for x in raw) for key in
                    ('worker_host_s','query_enqueue_host_s','selector_and_indices_sync_host_s',
                     'sort_host_s','dedup_host_s','cpu_gather_host_s','h2d_submit_host_s')}}
        row.update(mode=mode, capture=capture, instrumented=profile, profile=self.last_profile,
            cache_reset='Exact initial recurrent/cache/working-set restoration; prefill and restore excluded',
            extra_resident_history='Full identical GPU FP8 shadow allocated; only read/updated by gpu_source diagnostics')
        return row

    def dispose(self):
        if not self.prepared:
            return
        cache = self.prepared[1]
        self.prepared = None
        self.snapshots = []
        if self.controller:
            self.controller.active = False
            self.controller.tapes.clear()
            for layer in cache.layers:
                if hasattr(layer, '_resident'):
                    layer._resident = None
        super().cleanup(cache)
        del cache
        gc.collect()
        torch.cuda.empty_cache()
