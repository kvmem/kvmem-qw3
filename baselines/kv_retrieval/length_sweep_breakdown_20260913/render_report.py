"""Render the completed measurements without copying rounded numbers by hand."""
from pathlib import Path
import csv
import json
import statistics

HERE = Path(__file__).resolve().parent
METHODS = ['qw3', 'kvmem', 'infllm', 'infinigen']
NAMES = dict(qw3='普通 qw3', kvmem='KVMem', infllm='InfLLM Qwen 适配', infinigen='InfiniGen Qwen 适配')


def read(name):
    return list(csv.DictReader((HERE / name).open()))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    audit = json.loads((HERE / 'final_audit.json').read_text())
    assert audit['status'] == 'complete'
    rows = read('measurements.csv')
    summaries = read('summary.csv')

    def summary(scenario, method, n, mtp):
        matches = [r for r in summaries if r['scenario'] == scenario and r['method'] == method
                   and int(r['history_tokens']) == n and int(r['mtp']) == mtp]
        assert len(matches) == 1
        return matches[0]

    def grid(name):
        return table(['历史长度'] + [NAMES[m] for m in METHODS],
                     [[r['context_label']] + [f'{float(r[m]):.2f}' for m in METHODS] for r in read(name)])

    def median_native(n, field):
        return statistics.median(float(r[field]) for r in rows if r['scenario'] == 'new_query'
                                 and r['method'] == 'kvmem' and int(r['mtp']) == 4
                                 and int(r['history_tokens']) == n)

    sections = [
        '# 四系统吞吐与新增 query 开销：最终实验记录',
        '完成日期：2026-09-14。正式主表共 240 次测量（两种场景 × 四系统 × 五长度 × 两档 MTP × 三次重复）。另保留 240K 补测、等价诊断、旧追加路径诊断和 guided-query 补测；共 374 行，未删除慢重复。',
        '本次修改仅在本目录。使用同一张独占 RTX PRO 6000 Blackwell Server Edition、同一 Qwen3.8-27B Q8_0 权重、FP8 历史 KV、batch=1、greedy、128 个输出 token。主表长度为 8,192 / 16,384 / 32,768 / 65,536 / 131,072，表内取三次中位数，CSV 保留每次结果和 min/max。',
        'KVMem 为 selection=128K、generation=128K、block=32，2M CPU 历史容量，不使用 NVMe/GC，保留完整 query replay。InfLLM 使用作者 ContextManager 的小窗口配置（128 initial + 4096 local + 16×128 retrieval）；InfiniGen 使用每 query head 至多 2048 行的选择上限。两者经隔离的 Qwen 适配运行；不是作者原仓库开箱运行 Qwen 的结果。输入历史长度相同，实际 attention 窗口不同。',
        '输入来自一条已记录的 DeepSWE superjson-error-stack-serialization 历史的不同长度前缀，不是五个独立任务。普通 qw3 在这个固定 token 微基准中没有 Claude Code compaction。这里不评估 agent 成功率。完整定义见 [DATA_GUIDE.md](DATA_GUIDE.md)。',
        '## 1. 持续 decode 吞吐',
        '单位 token/s；首 token 后的 127 个输出除以该 decode 区间，不含历史 prefill。',
        '### MTP 关闭', grid('steady_decode_mtp0.csv'),
        '### MTP=4', grid('steady_decode_mtp4.csv'),
        '![Steady decode](steady_decode_throughput.png)',
        '## 2. 用户新增 query：把实际重选开销计入请求',
        '先准备 N token 的已有历史，再追加同一个问题（正文 27 token，含聊天包装共 40 token），生成 128 个有效输出。有效吞吐 = 128 / 完整新增请求耗时。计入本次 query prefill、实际发生的选择、KV 工作集整理、query replay、生成和请求收尾；不计入请求到来前已有历史的准备。每次重复从相同的准备边界开始，禁止缓存命中新问题。',
        '原生组主表使用服务器请求计时，同时保存 HTTP wall time；外部方法从已准备好的 query tensor 开始进程内计时。它们不是完全一致的客户端端到端边界，也不包含 Claude Code 构造消息和工具运行。',
        '8K–64K 尚未达到 KVMem 的 128K 选择预算，因此追加问题不会强制触发语义重选；128K + 40 token 实际触发选择和完整 replay。另用 240K 验证预算以上的路径。不能将所有五档都标为“发生重选”。',
        '### MTP 关闭：含新增 query 的有效吞吐', grid('new_query_mtp0.csv'),
        '### MTP=4：含新增 query 的有效吞吐', grid('new_query_mtp4.csv'),
        '![New-query effective throughput](new_query_throughput.png)',
        '这张表直接使用用户给出的 query 检索。内部另外生成 guided query 的更完整路径在下一节单列。InfLLM 采用作者支持的整段 40-token 输入追加；旧的逐 token 追加数据保留为诊断，不进入主表。InfiniGen 当前适配仍按位置执行因果选择，已知输入的 dense compute 批量上限为 5，存在进一步优化输入追加的空间。',
        '原持续 decode 场景与新问题会生成不同答案，MTP 接受率也可能变化，不能直接相减当作维护开销。对照同一个新增请求的 decode 区间，得到：',
    ]
    same = []
    for n in [131072, 245760]:
        s = summary('new_query', 'kvmem', n, 4)
        same.append([f'{n//1024}K', f"{float(s['median_decode_tokens_per_s']):.2f}",
                     f"{float(s['median_effective_tokens_per_s']):.2f}",
                     f"{float(s['median_query_ttft_s']):.3f}",
                     f"{median_native(n, 'semantic_ms'):.1f}", f"{median_native(n, 'query_replay_ms'):.1f}"])
    sections += [table(['KVMem MTP=4', '同请求 decode t/s', '含 query 有效 t/s', 'TTFT 秒', '选择 ms', 'replay ms'], same),
                 'TTFT、选择和 replay 各取三次中位数；中位数分项不保证精确相加。原生 MTP=0 未暴露的分项时间在 CSV 留空，不冒充零开销。',
                 '![New-query TTFT](new_query_ttft.png)',
                 '## 3. 内部 guided-query 生成补测',
                 '显式要求模型先生成内部检索 query（thinking=0，query max=256，MTP=4），再选择、完整 replay、生成 128 个有效输出。内部检索 query 的 token 不增加有效输出分子。每档只有一次独立补测，因此仅用于确认完整路径的实际开销，不与三次中位数同等解读。']
    guided = read('guided_query_supplement.csv')
    sections.append(table(['历史', '实际 query 路径', '内部生成 token', '请求总秒', 'TTFT 秒', 'guided query ms', '选择 ms', 'replay ms', '有效 t/s'],
                          [[f"{int(r['history_tokens'])//1024}K", r['guided_query_mode'], r['guided_query_generated_tokens']] +
                           [f'{float(r[k]):.3f}' for k in ['request_s', 'query_ttft_s']] +
                           [f'{float(r[k]):.1f}' for k in ['guided_query_ms', 'semantic_ms', 'query_replay_ms']] +
                           [f"{float(r['effective_tokens_per_s']):.2f}"] for r in guided]))
    sections += ['128K 生成了 20 个内部 query token。240K 的模型立即结束内部 query，日志为 fallback-original / reason=empty-output，因此回退使用原始 27-token 用户问题；130 ms 包含尝试生成和状态处理，但它不是成功生成 guided query 的耗时样本。保留该自然回退结果，不删除或替换。不能将这两次时间推广为任意 agent 问题的上限。',
                 '## 4. Performance breakdown：差距从哪里来',
                 '### 工作集复用减少重复选择与搬运',
                 'KVMem 在刷新时付出选择和恢复开销，随后沿用当前工作集完成输出。外部适配在生成/验证的位置上继续做各 attention layer 的检索或预测。128K 关闭 MTP 的 host profile 中，两者均为每输出 16 次检索/预测，对应该模型的 16 个 full-attention 层。KVMem 的 full-attention 工作集更大，因此优势不能解释为“少关注历史”；它主要避免重复选择和重复 CPU→GPU 访问，而不是减少 attention token 数。',
                 'InfLLM 作者的 GPU 驻留缓存已启用。128K、MTP=0 时，32,512 个被选块中 26,831 个已驻留 GPU，命中率 82.5%；其中 20,879 个已在当前 attention buffer。它仍需评分、索引同步和部分块整理。不能将它的耗时解释为每次把所有检索块从 CPU 重传。',
                 '同条件下，InfLLM 的逻辑缺失块负载约 11.18 MiB/输出；InfiniGen 预测的选择集平均约 1,632 行/query head，逻辑 H2D 约 210.36 MiB/输出。开启 MTP 后，InfiniGen 每个有效输出约 21.42 次预测，逻辑 H2D 增为 281.85 MiB。该逻辑计数只描述当前适配的检索负载，和下面涵盖整条请求的实际 memcpy 计数不是同一口径。',
                 '### 实际 GPU trace：分开传输与执行空隙',
                 '对每个系统和 MTP 档另采集一次 128K 新 query 请求。采样计时只用于定位，不替换未插桩正式吞吐。H2D/D2H 包含模型辅助状态等所有被记录传输，不是纯历史 KV。多个 stream 的活动取时间并集；“无记录 GPU 活动”既不是 CPU 执行时长，也不是 SM 利用率。']
    traces = json.loads((HERE / 'trace_analysis.json').read_text())
    traffic = []
    for mtp in [0, 4]:
        for method in METHODS:
            r = next(x for x in traces if x['method'] == method and x['mtp'] == mtp)
            copies = {x['label']: x for x in r['copy_groups']}
            h2d = copies['Host-to-Device']
            traffic.append([NAMES[method], mtp, f"{r['nvtx_request_s']:.3f}",
                            f"{h2d['bytes']/2**20:.1f}", f"{h2d['union_ms']:.1f}",
                            f"{r['gpu_kernel_union_s']:.3f}", f"{r['no_recorded_gpu_activity_s']:.3f}"])
    sections.append(table(['系统', 'MTP', '插桩请求秒', 'H2D MiB', 'H2D GPU ms', 'kernel 并集秒', '无记录 GPU 活动秒'], traffic))
    sections += ['![GPU activity](new_query_gpu_activity.png)',
                 'GPU memcpy 执行时间不能直接等同于搬运造成的请求停顿：CPU gather、索引回传同步、Python/调度等待也可能出现在前后。host profile 各区间会重叠，因此不把它们相加成 100% 堆叠图。',
                 'kernel 明细也给出了具体的执行差异：native 使用融合 SiLU×gate 的 Q8 kernel 和融合 residual-add 的 Q8 kernel；外部路径有更多独立 tensor copy、拼接和逐元素操作。128K、MTP=0 的该条 trace 中，InfLLM 的 `_attn_fwd` 累计约 1.075 秒 / 4096 次调用，KVMem 的 FlashInfer BatchDecode 约 0.384 秒 / 2032 次调用。两者 attention 组织和输入不同，这不是孤立 kernel 的等工作量加速比，但证明除了 H2D，attention 布局、执行次数和融合也参与了总差距。逐 kernel 的原始累计时间和调用次数在 trace_kernels.csv。',
                 '### 同实现等价诊断：量化可去掉的检索/搬运路径',
                 '在 32K 和 128K 保存相同的选择结果，分别用 GPU 历史来源、已记录 oracle IDs、两者同时启用做诊断。输出检查通过，选择内容不变；但是需要额外 GPU 内存或预知选择结果，不能作为可部署 baseline。GPU-source-only 仍保留索引同步，并不等于彻底优化的全 GPU 实现。']
    gaps = read('diagnostic_gap_attribution.csv')
    sections.append(table(['系统', '长度', '原始 t/s', 'GPU 来源 t/s', 'oracle+GPU t/s', 'KVMem t/s', '消除的观测差距'],
                          [[NAMES[r['method']], f"{int(r['history_tokens'])//1024}K"] +
                           [f'{127/float(r[k]):.2f}' for k in ['normal_s', 'gpu_kv_source_s', 'oracle_ids_gpu_source_s', 'kvmem_native_s']] +
                           [f"{100*float(r['fraction_of_observed_gap_removed']):.1f}%"] for r in gaps]))
    sections += ['上表均关闭 MTP。128K 时，对 InfLLM 和 InfiniGen 分别消除了当前观测耗时差距的 31.9% 和 62.9%。这是在各自实现内部联合绕过选择与数据路径后测到的变化，不是“方法固有开销占比”，也不是可相加的独立归因。剩余差距仍包含不同 kernel、attention 执行、调度、状态处理和适配成本，不能全部归于某一个因素。',
                 '### MTP 与引擎会进一步扩大差距',
                 '128K 稳态 decode 中，KVMem 从 39.32 升到 76.91 t/s（约 1.96×）；InfLLM 从 19.74 变为 19.02；InfiniGen 从 15.44 变为 10.11。这里的外部 MTP 要维护 full-history draft cache，并对推测位置执行检索、预测与恢复；较高接受率并不自动抵消这些开销。',
                 '因此 MTP=4 的总差距同时体现 native qw3 引擎、草稿/验证执行效率与工作集复用。InfLLM/InfiniGen 的原论文并不因此被证明具有相同幅度的固有劣势。这组数据可以支持“当前同模型系统实现下 KVMem 更快”，不能支持“检索策略单独带来全部加速”。',
                 '普通 qw3 与 KVMem 的 native 请求路径、状态维护和 MTP 接受行为也不同；KVMem 在稳态表略低不能直接当作重选成本，因为当时并没有包含一次新重选。128K 新 query 场景中 KVMem 同请求 decode 反而比普通 qw3 略快，而加入选择/replay 后有效吞吐更低，正好说明需要把两个计时区间分开。',
                 '## 5. 验证与使用边界',
                 f"完成输入 token ID 对齐、全部正式单元三次重复完整性检查，以及 {audit['external_equivalence_checks']} 次外部优化/MTP 同状态输出一致性检查。Nsight 八条请求均独立于正式吞吐。",
                 'native MTP 并非所有点都与 MTP=0 输出字节一致：KVMem 32K 和普通 qw3 128K 的各三次稳态输出存在差异，已在 CSV/audit 中明确记录。不能宣称所有系统、所有 MTP 档输出完全一致，更不能由吞吐推断能力相等。',
                 '没有过滤慢重复或扣除 cold prefill；是否有 checkpoint fallback 可逐行检查 measurements.csv 和原始 server.log。原生新增问题的缓存边界检查全部通过。',
                 '正式新 query 主表的原始外部 InfLLM 输入路径修正为作者支持的整段追加；旧路径保留为 query_serial_append_diagnostic，不混进最终主表。MTP=0 CLI 自动 guided 配置曾在启动阶段被拒绝，该零请求尝试已归档，修正后才采样。',
                 '## 数据与复现',
                 '- 主表：steady_decode_mtp0.csv / steady_decode_mtp4.csv / new_query_mtp0.csv / new_query_mtp4.csv。',
                 '- 同请求 decode：new_query_decode_mtp0.csv / new_query_decode_mtp4.csv。',
                 '- 全部分次结果：measurements.csv；统计：summary.csv；guided：guided_query_supplement.csv；分项：kvmem_query_phases.csv。',
                 '- Breakdown：host_traffic_profiles.csv、diagnostic_gap_attribution.csv、trace_activity.csv、trace_copy_groups.csv、trace_kernels.csv、trace_runtime.csv。',
                 '- 校验与来源：final_audit.json、experiment_manifest.json、各 result.json/server.log、PLAN.md、DATA_GUIDE.md。',
                 '- 原始数据始终保留；数据包附包含 CSV、图、说明、脚本和校验元数据，不打包大型 Nsight trace 与暂停前 OSWorld 轨迹。',
                 '- OSWorld 的暂停归档、原有完成结果哈希和恢复成功记录见 pause.json 与 resume_status.json。']
    (HERE / 'REPORT.md').write_text('\n\n'.join(sections) + '\n')
    print(HERE / 'REPORT.md')


if __name__ == '__main__':
    main()
