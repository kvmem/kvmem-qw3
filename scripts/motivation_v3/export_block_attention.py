"""Export complete query-by-block scores with conservatively verified block IDs."""
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import shutil
import zipfile

import numpy as np
from metrics import frozen_curves, js, kl, normalize

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'benchmark/results/motivation_v3_20260910/corrected80'
OUT = SOURCE / 'block_attention_export_20260913'
LAYERS = list(range(3, 64, 4))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lcp(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def read_gz(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    analysis = json.loads((SOURCE/'analysis_kl/analysis.json').read_text())
    reference = {r['id']: r for r in analysis['trajectories']}
    assert len(manifest['trajectories']) == len(reference) == 80
    counts = Counter()
    index, boundary_index = [], []
    max_error, max_row_sum_error = 0., 0.
    hashes = {}
    for trajectory in manifest['trajectories']:
        tid = trajectory['id']
        folder = OUT/'trajectories'/tid
        folder.mkdir(parents=True, exist_ok=True)
        previous = None
        expected = reference[tid]
        for step, frame in enumerate(trajectory['frames']):
            source_meta = SOURCE/'measurements'/tid/f'step_{step:03d}.json'
            s = json.loads(source_meta.read_text())
            assert s['status'] == 'collected' and s['step'] == step and s['trajectory_id'] == tid
            assert s['source_step'] == frame['source_step'] and s['layers'] == LAYERS and s['block_tokens'] == 128
            inp = read_gz(Path(s['input_tokens_path']))
            with np.load(s['path'], allow_pickle=False) as z:
                mass, positions = z['mass'], z['positions']
            assert mass.dtype == np.float32 and np.isfinite(mass).all() and (mass >= 0).all()
            prompt, width, bt = s['prompt_tokens'], mass.shape[1], s['block_tokens']
            assert len(inp['tokens']) == prompt and inp['append_base'] == 0
            np.testing.assert_array_equal(positions, np.arange(prompt-s['prefill_samples'], prompt+s['decode_samples']))
            assert s['prefill_samples'] == min(64, prompt)
            row_error = float(np.max(np.abs(mass.sum(1)-1)))
            max_row_sum_error = max(max_row_sum_error, row_error)
            assert row_error < 1e-6
            for q, pos in enumerate(positions):
                assert not mass[q, pos//bt+1:].any()
            prefill = positions < prompt
            capped = s['completion_tokens'] >= manifest['max_decode_tokens'] or s['finish_reason'] == 'length'
            assert s['decode_samples'] == s['completion_tokens']-int(s['completion_tokens'] == manifest['max_decode_tokens'])
            adjacent = previous is not None and previous['meta']['source_step']+1 == s['source_step']
            oldimages = previous['frame'].get('image_sha256', []) if previous else []
            newimages = frame.get('image_sha256', [])
            image_ok = adjacent and (not oldimages or oldimages == newimages[:len(oldimages)])
            verified = bool(image_ok)
            assert verified == s['prefix_identity_verified']
            common = 0
            if verified:
                old = previous['input']
                common = lcp(old['tokens'], inp['tokens'])
                if old['mrope'] and inp['mrope']:
                    common = min(common, lcp(old['mrope'], inp['mrope']))
                assert common == s['common_prefix_tokens'] == s['history_end_tokens']
            block_positions = np.arange(width, dtype=np.int64)
            # IDs are scoped by trajectory. New/nonverified content gets a new birth-step ID.
            block_ids = (np.int64(step) << np.int64(32)) + block_positions
            if verified:
                reusable = min(common//bt, previous['meta']['prompt_tokens']//bt, len(previous['ids']), width)
                block_ids[:reusable] = previous['ids'][:reusable]
                counts['inherited_block_occurrences'] += reusable
            prompt_end = min(width, max(1, prompt//bt-8))
            prompt_history = (block_positions >= 1) & (block_positions < prompt_end)
            candidate = prompt_history & (block_positions < common//bt) if verified else np.zeros(width, dtype=bool)
            common_mask = candidate.copy()
            if previous:
                common_mask &= block_positions < previous['prompt_end']
            else:
                common_mask[:] = False
            full_prompt = (block_positions+1)*bt <= prompt
            sink = block_positions == 0
            recent = full_prompt & (block_positions >= max(1, prompt//bt-8))
            kind = np.full(width, 4, dtype=np.int8)
            kind[full_prompt] = 3
            kind[candidate] = 2
            kind[recent] = 1
            kind[sink] = 0
            decode_mass, prefill_mass = mass[~prefill], mass[prefill]
            scores = np.zeros(width, dtype=np.float64)
            candidate_columns = np.flatnonzero(candidate)
            if len(candidate_columns):
                scores[candidate] = normalize(prefill_mass[:, candidate]).mean(0)
            selection_defined = bool(scores.sum() > 0)
            arrays = dict(block_ids=block_ids, block_position_indices=block_positions,
                          block_token_start=block_positions*bt, block_token_end=(block_positions+1)*bt,
                          block_kind=kind, is_sink=sink, is_recent_prompt=recent,
                          is_full_prompt_block=full_prompt, prompt_history_mask=prompt_history,
                          candidate_history_mask=candidate, common_history_with_previous_mask=common_mask,
                          query_positions=positions, query_is_prefill=prefill,
                          prefill_query_positions=positions[prefill], decode_query_positions=positions[~prefill],
                          decode_block_mass=decode_mass, prefill_tail_block_mass=prefill_mass,
                          prefill_selection_scores=scores,
                          standard_attention_layer_indices=np.asarray(LAYERS, dtype=np.int64))
            for budget, fraction in [('r05', .05), ('r10', .1), ('r20', .2)]:
                k = max(1, int(np.ceil(len(candidate_columns)*fraction))) if selection_defined else 0
                selected = candidate_columns[np.argsort(-scores[candidate], kind='stable')[:k]]
                arrays['offline_prefill_selected_block_ids_'+budget] = block_ids[selected]
            npz_path = folder/f'step_{step:03d}.npz'
            np.savez_compressed(npz_path, **arrays)
            # Verify exact float32 scores, query ordering and identities survive serialization.
            with np.load(npz_path, allow_pickle=False) as z:
                for key in arrays:
                    np.testing.assert_array_equal(z[key], arrays[key])
                np.testing.assert_array_equal(np.concatenate([z['prefill_tail_block_mass'], z['decode_block_mass']]), mass)
            token_path = folder/f'input_tokens_{step:03d}.json.gz'
            shutil.copy2(s['input_tokens_path'], token_path)
            metadata = dict(schema_version=1, trajectory_id=tid, domain=trajectory['domain'],
                step_id=step, source_step_id=s['source_step'], npz_file=npz_path.name,
                input_identity_file=token_path.name, prompt_tokens=prompt,
                completion_tokens=s['completion_tokens'], decode_queries=len(decode_mass), prefill_queries=len(prefill_mass),
                finish_reason=s['finish_reason'], truncated=capped, natural_end=not capped,
                block_tokens=bt, block_id_scope='trajectory_id',
                block_id_rule='(birth_step_id << 32) + position_block_index; inherit only full blocks in verified adjacent input prefix',
                previous_step_id=step-1 if adjacent else None,
                prefix_identity_verified=verified, common_prefix_tokens=common if verified else None,
                common_history_block_count=int(common_mask.sum()), candidate_history_block_count=int(candidate.sum()),
                image_prefix_verified=bool(image_ok), image_sha256=newimages,
                prefix_exclusion_reason=s.get('prefix_exclusion_reason', 'No adjacent predecessor' if not adjacent else None),
                selection_defined=selection_defined,
                selection_method='Offline exact prefill-tail attention; per-query candidate normalization, then query mean; stable descending sort; ceil budget',
                actual_kvmem_selected_block_ids=None, actual_kvmem_enabled=False,
                attention_aggregation='24 query heads summed per standard-attention layer, normalize each layer over all visible blocks, arithmetic mean of 16 layers; float32',
                mass_normalization='Normalized over all visible blocks, NOT conditioned on candidate history; future positions have zero mass',
                model=manifest['model'], kv_dtype=manifest['kv_dtype'], replay=True, closed_loop=False,
                source_npz_sha256=sha(Path(s['path'])), source_metadata_sha256=sha(source_meta),
                source_record=s)
            meta_path = folder/f'step_{step:03d}.json'
            meta_path.write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
            index.append({k: metadata[k] for k in ('trajectory_id','domain','step_id','source_step_id','prompt_tokens',
                'completion_tokens','decode_queries','prefill_queries','finish_reason','truncated','natural_end',
                'prefix_identity_verified','common_prefix_tokens','common_history_block_count','candidate_history_block_count','selection_defined')} |
                dict(npz_file=str(npz_path.relative_to(OUT)), metadata_file=str(meta_path.relative_to(OUT))))
            if previous and verified:
                common_ids = block_ids[common_mask]
                old_lookup = {int(b):i for i,b in enumerate(previous['ids'])}
                old_columns = np.asarray([old_lookup[int(b)] for b in common_ids], dtype=np.int64)
                assert np.array_equal(old_columns, np.flatnonzero(common_mask))
                # Verified common support is contiguous. Keep the original slice layout
                # so float32 window means use the same reduction order as the archive.
                np.testing.assert_array_equal(old_columns, np.arange(1, 1+len(common_ids)))
                old_decode = previous['decode'][:, 1:1+len(common_ids)]
                new_decode = decode_mass[:, 1:1+len(common_ids)]
                eligible = []
                for window in (16,32,64,128):
                    saved = next((b for b in expected['settings'][str(window)]['matched_boundaries']
                                  if b['from_step']==previous['meta']['source_step'] and b['to_step']==s['source_step']), None)
                    is_eligible = len(common_ids)>0 and len(old_decode)>=2*window and len(new_decode)>=2*window
                    assert is_eligible == (saved is not None)
                    if not is_eligible:
                        continue
                    eligible.append(window)
                    before, tail = old_decode[-2*window:-window].mean(0), old_decode[-window:].mean(0)
                    head, after = new_decode[:window].mean(0), new_decode[window:2*window].mean(0)
                    for metric, fn in [('kl', kl), ('js', js)]:
                        actual = {'within_'+metric: (fn(before,tail)+fn(head,after))/2, 'cross_'+metric: fn(tail,head)}
                        for key,value in actual.items():
                            error=abs(value-saved[key]);max_error=max(max_error,error)
                            assert error < 1e-7, (tid, step, window, key, error)
                    assert len(common_ids)==saved['common_blocks']
                    counts[f'reproduced_boundaries_w{window}'] += 1
                boundary_index.append(dict(trajectory_id=tid, domain=trajectory['domain'],
                    from_step_id=previous['meta']['step'], to_step_id=step,
                    common_prefix_tokens=common, common_history_blocks=len(common_ids),
                    eligible_windows=eligible, common_block_ids=common_ids.tolist()))
            if selection_defined and len(decode_mass) and (decode_mass[:, candidate].sum(1)>0).any():
                curves = frozen_curves(decode_mass[:, candidate], scores[candidate])
                detail = next(d for d in expected['settings']['32']['details'] if d['step']==s['source_step'])
                for budget, values in curves.items():
                    for key in ('frozen','oracle','gap','mean','oracle_mean'):
                        np.testing.assert_allclose(values[key], detail['prefill_diagnostic'][budget][key], rtol=1e-6, atol=1e-7)
                counts['reproduced_coverage_steps'] += 1
            for path in (npz_path, meta_path, token_path):
                hashes[str(path.relative_to(OUT))] = sha(path)
            counts['steps'] += 1
            counts['prefill_queries'] += len(prefill_mass)
            counts['decode_queries'] += len(decode_mass)
            counts['truncated_steps'] += capped
            counts['verified_prefix_steps'] += verified
            counts['candidate_history_steps'] += bool(candidate.any())
            counts['stored_attention_values'] += mass.size
            previous = dict(meta=s,input=inp,frame=frame,ids=block_ids,decode=decode_mass,prompt_end=prompt_end)
        counts['trajectories'] += 1
        print(f"{counts['trajectories']}/80 trajectories exported; {counts['steps']} steps verified", flush=True)
    assert counts['steps']==957 and counts['trajectories']==80
    assert counts['reproduced_boundaries_w32']==791 and counts['reproduced_coverage_steps']==858
    with (OUT/'index.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(index[0]));writer.writeheader();writer.writerows(index)
    (OUT/'boundaries.json').write_text(json.dumps(boundary_index,indent=2)+'\n')
    shutil.copy2(SOURCE/'manifest.json',OUT/'manifest.json')
    shutil.copy2(Path(__file__),OUT/'export_block_attention.py')
    shutil.copy2(Path(__file__).with_name('metrics.py'),OUT/'metrics.py')
    shutil.copy2(SOURCE/'analysis_kl/report.md',OUT/'reference_kl_report.md')
    (OUT/'audit.json').write_text(json.dumps(dict(status='complete',counts=dict(counts),
        max_reproduced_divergence_absolute_error=max_error, max_full_row_sum_error=max_row_sum_error,
        source_analysis_sha256=sha(SOURCE/'analysis_kl/analysis.json'),
        checks=['All 957 source arrays and token/mRoPE records present and readable',
                'Adjacent token/mRoPE/image prefix identity independently recomputed',
                'Every exported array round-trips exactly; all block columns retained',
                'All four windows reproduce archived matched KL/JS values',
                'All 858 eligible steps reproduce all seven budget coverage diagnostics'],
        actual_kvmem_selection_available=False),indent=2)+'\n')
    (OUT/'README_zh.md').write_text(README)
    (OUT/'load_example.py').write_text(EXAMPLE)
    for p in OUT.iterdir():
        if p.is_file() and p.name!='checksums.json': hashes[p.name]=sha(p)
    (OUT/'checksums.json').write_text(json.dumps(hashes,indent=2)+'\n')
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():z.write(path, arcname=OUT.name+'/'+str(path.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    print(json.dumps(dict(counts=dict(counts),zip=str(archive),zip_bytes=archive.stat().st_size,
                         max_divergence_error=max_error),indent=2))


README = '''# 每 query 完整 block attention 导出

沿用现有 80 条主轨迹、957 次调用；没有重新执行模型。每个 step 一个 NPZ + JSON，并保留实际输入 token/mRoPE 的 JSON.gz 身份证据。仅包含聚合 block 分数，不包含原始 Q/K、逐 head 分数、完整 token-to-token attention 或模型权重。

## 入口

index.csv：957 个 step 的路径、轨迹/源 step 标识、结束/截断、前缀及候选历史资格。
trajectories/<trajectory_id>/step_NNN.npz：该 step 的全部 block 分数和身份/角色数组。
同名 JSON：元数据、归一化方式、来源 SHA256、图像 SHA256 序列及原采集元数据。
boundaries.json：相邻已核验边界的共同 block IDs，以及可用于 16/32/64/128 窗口的资格。
audit.json：全量复核与原 KL/JS、coverage 的重现结果。checksums.json：包内各文件 SHA256。
load_example.py：只依赖 numpy 的读取、固定集覆盖率与共同历史对齐示例，可在解压目录运行。

## NPZ 字段

- block_ids [B]：int64，身份必须用 (trajectory_id, block_id) 二元组识别。不是全局 token 内容 hash。
- block_position_indices [B]：当前 execution view 中的位置块号，区间为 [128*b,128*(b+1))。
- block_token_start/end [B]：半开 token 位置区间。end 是块容量边界；最后一块可尚未填满。多模态位置包含图像展开位置，不等于文本字符偏移。
- decode_block_mass [D,B] / prefill_tail_block_mass [P,B]：float32 原精度，保留所有列（包括 sink、recent、当前 prompt、decode 尾部），并非 top-k。P=64，D 是实际 decode forward-query 数。
- query_positions [P+D] / query_is_prefill [P+D]：完整有序记录及阶段标识；前 P 行 prefill，后 D 行 decode。
- prefill_query_positions [P] / decode_query_positions [D]：两矩阵对应的位置。每个 query 时尚不可见的块分数为零。
- is_sink / is_recent_prompt / is_full_prompt_block / prompt_history_mask / candidate_history_mask / common_history_with_previous_mask [B]：角色/资格布尔数组，详见下文。
- block_kind [B]：0=sink，1=最近完整 prompt 块，2=已核验候选历史，3=其他完整 prompt 块，4=部分 prompt 块或 decode 尾部。
- prefill_selection_scores [B]：离线初始选择得分，非候选列为零；方法见下文。
- offline_prefill_selected_block_ids_r05/r10/r20：重建的 5%/10%/20% 初始选集，仅用于离线诊断，绝不是实际 KVMem 运行选集。
- standard_attention_layer_indices：零基 3,7,...,63 的 16 个标准 attention 层。

## Block 身份与候选历史

原记录的 block_ids 是位置号，不能无条件跨 step 当成同一内容。本包以 (birth_step_id << 32) + position_block_index 分配新 ID；仅当相邻源 step 的实际 token、mRoPE（适用时）及图像前缀关系核验通过，才继承共同前缀内完整块的旧 ID。部分块、变更内容、未核验前缀都分配新 ID。图像身份不通过时保守地不继承任何块。
这能证明相邻 step 的已核验公共前缀身份，不提供压缩/改写/搬移之后的语义内容追踪；相同位置号不保证身份相同。无跨步身份证明的 step 仍保留全部分数，但不应进入跨步比较。

prompt_history_mask：完整 prompt 块，去掉首个 sink 块和最后 8 个完整 prompt 块。
candidate_history_mask：在上述范围内，进一步限制到与前一步已核验共同输入前缀；无核验则为空。这与原初始选集/coverage 资格一致。
common_history_with_previous_mask：当前候选历史再与前一步 prompt_history_mask 取公共范围，重现原配对 KL/JS 的共同支持集。上一侧应通过稳定 IDs 对齐。不要直接把两侧 candidate_history_mask 求交替代它：前一步候选资格可能受更早边界限制，原配对统计不传播这种限制。
第一步及身份不通过的调用 candidate_history_mask 为空，这不代表没有 attention 数据。可用 prompt_history_mask 做新的单步诊断，但必须单独说明候选定义已变化。

## Attention 聚合与归一化

原 tracer 每个标准 attention 层对 24 个 query heads 求和。解析器对该层的全部可见块归一化，再对 16 个层等权平均，保存 float32。每行全部 block mass 之和约为 1；没有按候选历史重新归一化。覆盖了模型的标准 attention 层，不包含 hybrid linear-state 层；NPZ 不保留逐 head/逐 layer 自由度。

重建初始选集：截取 prefill_tail_block_mass[:, candidate_history_mask]，每行先除以候选历史质量，再对这 64 行求均值；按得分降序，位置顺序稳定打破平局；k=ceil(fraction*候选块数)。无正质量则不定义选集。
原 KL/JS 窗口定义：先对窗口中的原 mass 求 query 平均，再限制共同历史并归一化（限制与求均值可交换）；并非先对每个 query 条件归一化再平均。注意它与 prefill 选择的聚合顺序不同。
固定集 coverage：每个 decode query 对候选历史归一化，再求初始集合的质量。零候选历史质量的 query 应排除，不能填 0 或 100%。分子/全部可见质量则是另一个指标，可从完整列计算。

## 推荐后续统计

跨步替换率使用 boundaries.json 的共同 IDs，在两侧相同支持集上独立构造非重叠 W-query 平均分布，再比较 top-ceil(10%*B)。步内替换率应使用同样支持集、窗口和预算。
明确替换率定义：1-|A∩B|/k（等大小 top-k）与 Jaccard 距离 1-|A∩B|/|A∪B| 不是同一指标。
固定初始选集的后续覆盖率必须使用完整矩阵查询旧 IDs，不能只保留各窗口新 top-k 分数。
可用 5%/20% 预算及 16/64/128-query 窗口离线分析。短 decode 不符合窗口条件时应排除并报告样本数；主分析 W=32 的配对条件为两侧均至少 64 个 decode queries。
头部 prefill logits 产生第一个输出 token。正常结束时 decode queries=completion_tokens；达到 2048 输出上限时最后一个输出没有再次 forward，因此 decode queries=2047。用保存的 query_positions，不要用输出数构造 query 行号。

## 实际 KVMem 策略的范围

本批采集没有启用 KVMem（也没有执行新输出动作），没有实际选集 IDs、真实 KV 搬运或任务结果。JSON 的 actual_kvmem_selected_block_ids=null。这些数据足以完成所述精确 attention 的离线替换率、稳定性、coverage 和敏感性分析；若研究实际生产选择器的执行行为，则需要该策略的额外日志或实验。
'''

EXAMPLE = '''"""Run from anywhere: python load_example.py. Only numpy is required."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
with (ROOT/'index.csv').open(encoding='utf-8-sig', newline='') as f:
    index = list(csv.DictReader(f))
row = next(r for r in index if r['selection_defined']=='True' and int(r['decode_queries'])>=64)
with np.load(ROOT/row['npz_file'], allow_pickle=False) as z:
    ids = z['block_ids']
    mask = z['candidate_history_mask']
    mass = z['decode_block_mass']
    selected = np.isin(ids, z['offline_prefill_selected_block_ids_r10'])
    denominator = mass[:,mask].sum(1)
    valid = denominator>0
    coverage = mass[valid][:,selected].sum(1)/denominator[valid]
    print(row['trajectory_id'], row['step_id'], 'decode shape:', mass.shape)
    print('Fixed initial 10% set mean conditional coverage:', float(coverage.mean()))

boundary = next(b for b in json.loads((ROOT/'boundaries.json').read_text()) if 32 in b['eligible_windows'])
folder = ROOT/'trajectories'/boundary['trajectory_id']
ids = boundary['common_block_ids']
distributions = []
for step, last in [(boundary['from_step_id'],True), (boundary['to_step_id'],False)]:
    with np.load(folder/f'step_{step:03d}.npz',allow_pickle=False) as z:
        mapping = {int(b): i for i,b in enumerate(z['block_ids'])}
        cols = [mapping[b] for b in ids]
        window = z['decode_block_mass'][-32:] if last else z['decode_block_mass'][:32]
        distributions.append(window[:,cols].mean(0))
k=max(1,int(np.ceil(.1*len(ids))))
sets = [set(np.asarray(ids)[np.argsort(-p,kind='stable')[:k]].tolist()) for p in distributions]
print('Example boundary:', boundary['trajectory_id'], boundary['from_step_id'], boundary['to_step_id'])
print('Top-10% replacement rate (1 - intersection/k):',1-len(sets[0]&sets[1])/k)
'''


if __name__ == '__main__':
    main()
