"""Summarize Q8/MTP measurements without replacing earlier precision variants."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def main():
    plan = json.loads((HERE / 'jobs_v2.json').read_text())
    state = json.loads((HERE / 'runs/status.json').read_text())
    assert state['status'] == 'complete'
    rows = []
    for job in plan['jobs']:
        path = Path(job['result'])
        report = json.loads(path.read_text())
        assert report['status'] == 'complete'
        assert report['weight_dtype'] == 'Q8_0'
        for r in report['results']:
            if r['warmup']:
                continue
            assert r['usage']['completion_tokens'] == 128
            mtp = r['mtp_summaries'][-1]['values'] if r['mtp_summaries'] else {}
            rows.append(dict(
                method=job['name'], history_tokens=r['prompt_tokens'], rep=r['rep'],
                weight_dtype=report['weight_dtype'], kv_dtype=report['kv_storage_dtype'],
                mtp_chain=report['mtp_chain'], decoded_tokens=128, timed_output_tokens=127,
                decode_tokens_per_s=r['decode_tokens_per_s'], decode_interval_s=r['decode_interval_s'],
                prefill_s=r['phases'].get('prefill_ms', 0) / 1000,
                request_wall_s=r['request_wall_s'], server_ttft_s=r['server_timing']['server_ttft_ms'] / 1000,
                native_logged_decode_s=r['phases'].get('decode_ms', 0) / 1000,
                mtp_drafted=mtp.get('drafted'), mtp_accepted=mtp.get('accepted'),
                mtp_acceptance=mtp.get('acceptance'),
                input_ids_sha256=r['input_ids_sha256'], output_text_sha256=r['output_text_sha256'],
                source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(rows) == plan['expected_rows']
    for length in {r['history_tokens'] for r in rows}:
        assert len({r['input_ids_sha256'] for r in rows if r['history_tokens'] == length}) == 1
    groups = {}
    for r in rows:
        groups.setdefault((r['method'], r['history_tokens']), []).append(r)
    summaries = []
    for (method, length), values in sorted(groups.items()):
        assert sorted(r['rep'] for r in values) == [0, 1, 2]
        rates = [r['decode_tokens_per_s'] for r in values]
        drafted = sum(r['mtp_drafted'] or 0 for r in values)
        accepted = sum(r['mtp_accepted'] or 0 for r in values)
        summaries.append(dict(method=method, history_tokens=length, n=len(values),
            median_tokens_per_s=statistics.median(rates), min_tokens_per_s=min(rates),
            max_tokens_per_s=max(rates), mtp_acceptance=accepted / drafted if drafted else None,
            pooled_tokens_per_s=127 * len(values) / sum(r['decode_interval_s'] for r in values),
            median_prefill_s=statistics.median(r['prefill_s'] for r in values),
            repeat_output_text_variants=len({r['output_text_sha256'] for r in values})))
    out = HERE / 'export'
    out.mkdir(exist_ok=True)
    control = [r for r in rows if r['method'] == 'kvmem_q8_mtp0' and r['history_tokens'] == 32768]
    speculative = [r for r in rows if r['method'] == 'kvmem_q8_mtp4' and r['history_tokens'] == 32768]
    outputs_equal = ([r['output_text_sha256'] for r in control] ==
                     [r['output_text_sha256'] for r in speculative])
    for name, data in [('runs', rows), ('summary', summaries)]:
        with (out / f'{name}.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (out / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    (out / 'audit.json').write_text(json.dumps(dict(status='complete', runs=len(rows), groups=len(groups),
        matching_input_ids=True, same_q8_weight_file=True, external_mtp_not_implemented=True,
        control_32k_output_text_equal=outputs_equal,
        output_equivalence_claim=False,
        timing='127 output tokens / (server completion - first model token); includes final bookkeeping',
        weight_identity=json.loads((HERE / 'weight_identity.json').read_text())), indent=2) + '\n')
    lines = ['# Qwen3.8-27B Q8_0 / FP8 KV / MTP 吞吐复测', '',
        '独占同一 GPU，绑定 GPU 所在 NUMA 的 CPU；同一真实 DeepSWE 历史前缀，固定 128 个输出 token，贪心生成、忽略 EOS。每项三次重复。吞吐统计首个模型 token 后的 127 个输出，初始 prefill、guided query 和 query replay 单独保留。', '',
        'KVMem：128K 选择＋128K 生成预算，block=32，CPU 历史/index，NVMe=0，GC 关闭；MTP=4 主组保留 guided-reselect=both 开关及完整 replay。请求提供显式 query span，240K 实际观测到 query replay，没有私有 guided-query 生成日志。32K MTP=0 对照低于选择预算，用于观察配置切换后的吞吐。', '',
        '| 方法 | 输入 | Decode 中位数 [最小, 最大] token/s | MTP 接受率 |',
        '|---|---:|---:|---:|']
    for r in summaries:
        acceptance = f"{r['mtp_acceptance']:.1%}" if r['mtp_acceptance'] is not None else '—'
        lines.append(f"| {r['method']} | {r['history_tokens']//1024}K | {r['median_tokens_per_s']:.2f} [{r['min_tokens_per_s']:.2f}, {r['max_tokens_per_s']:.2f}] | {acceptance} |")
    lines += ['', '这些是部署配置的 decode 测量，不是完整 agent 成功率或端到端 latency。每个请求的 prefill、首 token、完整请求时间均保留在 export/runs.csv。重复相同 prompt 会撤回先前生成的尾部，checkpoint 回退也不得从原始记录中删除。', '',
        f'输出一致性边界：32K 的 MTP=0/4 输出文本逐次相同为 {outputs_equal}。两组输入 IDs 相同，但前缀缓存的先前请求路径和解码执行不同；本次未做状态对齐的数值等价测试。240K KVMem 三次输出也不同。吞吐及接受率为真实观测，不能宣称逐 token 等价加速或推断质量。', '',
        'InfLLM / InfiniGen 的现有 Qwen 适配未实现 MTP；它们不在这张 Q8/MTP 对照表中。此前 BF16/MTP=0 结果保留为不同配置的补充数据。', '',
        '量化权重身份见 weight_identity.json；启动参数、MTP 实际草稿/接受计数及服务器原始日志见 runs/。50 token/s 是待检验的预期，不是筛选或停止测试的条件。']
    (HERE / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': 'complete', 'runs': len(rows), 'summary': summaries}), flush=True)


if __name__ == '__main__':
    main()
