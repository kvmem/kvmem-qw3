def get_global_hidden_and_mask(
    self, len_q, block_topk
):
    assert len(block_topk) == self.num_units
    global_block_map = [[] for _ in range(self.num_units)]
    global_remainder_len = max(self._global_remainder_ed - self._global_remainder_st + len_q - self.n_local, 0)
    init_len = self.init_k.size(-2)
    sliding_window = None

    global_h_k = self.global_buffer[0]
    global_h_v = self.global_buffer[1]

    block_num = len(block_topk[0])
    for u in range(self.num_units):
        assert len(block_topk[u]) == block_num

        block_topk[u].sort()
        global_block_map[u] = deepcopy(self.global_buffer_block_id_list[u])
        selected_set = set(block_topk[u])
        present_set = set(global_block_map[u])
        free_slots = iter(j for j, old in enumerate(global_block_map[u])
                          if old == -1 or old not in selected_set)
        for b_idx in block_topk[u]:
            if b_idx in present_set:
                continue

            st = -1
            ed = -1
            j = next(free_slots)
            st = j * self.block_size
            ed = st + self.block_size
            present_set.discard(global_block_map[u][j])
            global_block_map[u][j] = b_idx
            present_set.add(b_idx)


            assert b_idx in self.cached_blocks[u]
            self.global_blocks[u][b_idx].load((global_h_k[u, :, st:ed, :], global_h_v[u, :, st:ed, :]))


    init_st = block_num * self.block_size
    init_ed = init_st + init_len
    if self.global_buffer_init_st != init_st or self.global_buffer_init_ed != init_ed:
        global_h_k[:, :, init_st: init_ed, :].copy_(self.init_k, non_blocking=True)
        global_h_v[:, :, init_st: init_ed, :].copy_(self.init_v, non_blocking=True)

    ed = init_ed

    rmd_st = init_ed
    rmd_ed = rmd_st + global_remainder_len
    ed = rmd_ed
    global_h_k[:, :, rmd_st: rmd_ed, :].copy_(self.global_remainder[0][:, :, self._global_remainder_st:self._global_remainder_st+global_remainder_len, :], non_blocking=True)
    global_h_v[:, :, rmd_st: rmd_ed, :].copy_(self.global_remainder[1][:, :, self._global_remainder_st:self._global_remainder_st+global_remainder_len, :], non_blocking=True)


    sliding_window = (self.global_remainder[0].size(-2) + rmd_st, self.n_local)

    self.global_buffer_block_id_list = deepcopy(global_block_map)
    self.global_buffer_init_st = init_st
    self.global_buffer_init_ed = init_ed

    for u in range(self.num_units):
        assert max(global_block_map[u][block_num:] + [-1]) == -1
        assert min(global_block_map[u][:block_num] + [0]) > -1
        global_block_map[u] = list(global_block_map[u][:block_num])


    global_h_k = global_h_k[:, :, :ed, :]
    global_h_v = global_h_v[:, :, :ed, :]
    return global_h_k, global_h_v, sliding_window, global_block_map, block_num
