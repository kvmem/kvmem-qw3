"""Bounded speculative prefix state; never copy immutable full KV history."""
import torch


class Snapshot:
    def __init__(self, state):
        self.state = state
        if hasattr(state, 'manager'):
            m = state.manager
            assert m.initialized and not m.faiss and not m.async_global_stream
            self.kind = 'infllm'
            # append creates fresh tail/score tensors before updating them.
            self.attrs = {k:getattr(m, k) for k in (
                'length', 'local_k', 'local_v', 'global_remainder',
                'global_remainder_local_score', 'init_k', 'init_v', 'init_exc',
                '_global_remainder_st', '_global_remainder_ed', 'num_global_block',
                'load_count', 'global_buffer_init_st', 'global_buffer_init_ed')}
            self.cached = [dict(x) for x in m.cached_blocks]
            self.maps = [list(x) for x in m.global_buffer_block_id_list]
            self.block_lengths = [x.length for x in m.block_k]
        elif hasattr(state, 'partial_keys'):
            assert state.finalized and state.future is None
            self.kind = 'infinigen'
            self.length = state.length
            self.metrics = len(state.metrics)
        else:
            self.kind = 'recurrent'
            self.conv = state.conv_states.clone()
            self.recurrent = state.recurrent_states.clone()

    def restore(self):
        s = self.state
        if self.kind == 'recurrent':
            s.conv_states.copy_(self.conv)
            s.recurrent_states.copy_(self.recurrent)
        elif self.kind == 'infinigen':
            assert s.future is None
            # CPU history and GPU partial index only appended after the prefix.
            # The next append overwrites those bytes; selectors obey length.
            s.length = self.length
            del s.metrics[self.metrics:]
        else:
            m = s.manager
            for u, old in enumerate(self.cached):
                for idx in set(m.cached_blocks[u]) - set(old):
                    unit = m.global_blocks[u][idx]
                    if unit.gpu_data is not None:
                        unit.offload()
                m.cached_blocks[u] = dict(old)
                for idx in old:
                    unit = m.global_blocks[u][idx]
                    if unit.gpu_data is None:
                        # Upstream load(None) leaves target_event undefined on
                        # a cache miss. Use a temporary block in the existing
                        # working buffer; its physical map is restored below.
                        unit.load(tuple(m.global_buffer[w,u,:,:m.block_size] for w in range(2)))
                    else:
                        unit.load()
                del m.global_blocks[u][self.attrs['num_global_block']:]
                m.block_k[u].length = self.block_lengths[u]
            for k, v in self.attrs.items():
                setattr(m, k, v)
            # Recreate the previous physical attention ordering, not just its
            # set of blocks (FP16 reduction order matters). Copies only the
            # selected working set, never full CPU history. Units remain cached.
            m.global_buffer_block_id_list = [list(x) for x in self.maps]
            for u, blocks in enumerate(self.maps):
                for slot, idx in enumerate(blocks):
                    if idx >= 0:
                        start = slot * m.block_size
                        unit = m.global_blocks[u][idx]
                        for which in range(2):
                            source = unit.gpu_data[which] if unit.gpu_data is not None else unit.cpu_data[which]
                            m.global_buffer[which, u, :, start:start + m.block_size].copy_(source, non_blocking=True)
            st, ed = m.global_buffer_init_st, m.global_buffer_init_ed
            m.global_buffer[0, :, :, st:ed].copy_(m.init_k)
            m.global_buffer[1, :, :, st:ed].copy_(m.init_v)
            m._use_chunk_topk = False


class Transaction:
    def __init__(self, cache):
        self.cache = cache
        self.prefixes = [[] for _ in cache.layers]

    def record(self, layer):
        self.prefixes[layer].append(Snapshot(self.cache.layers[layer]))

    def commit(self, count):
        assert count >= 1
        assert len({len(x) for x in self.prefixes}) == 1
        if count == len(self.prefixes[0]):
            return
        for row in self.prefixes:
            row[count - 1].restore()
