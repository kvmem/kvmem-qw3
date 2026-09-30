# ≤1M CacheBlend: remaining work migrated to RTX4090

2026-09-28 用户要求将剩余测试从 SUFE-3090 改到4090。本文件补充原 `memory_eval_le1m_handoff_20260928.md`，筛选和推理/评分协议仍以原冻结清单为准。

## 冻结迁移点

本机统一结果目录：`/data/chaidi/kvmem_eval/results/cacheblend_memory_le1m_ab_20260928`。

`cutover-4090.json` 固定1208份 on 组 result/answer SHA：原 PRO6000 970问 + SUFE-3090 238问。4090负责 on剩余410问，以及 ratio0 全部1618问。首次续跑是 `memoryagentbench/017_Long_Range_Understanding_r012_infbench_sum_eng_shots2_q0000`（269918输入tokens）；3090该次请求未保存response/result，作为基础设施中断归档，不能计模型答错。

SUFE-3090 因内存耗尽后SSH banner超时，停止命令未执行成功。**不要宣称它已停止。** 旧同步进程已退出，旧同步脚本增加 cutover 硬阻断。之后即使3090恢复并产出结果，也只能另行归档，不能覆盖接受的4090结果。旧在途文件保存在 `evidence/migration-4090/`，原970和已接收238份答案未改写。

## 4090 实际部署

- SSH：`root@ssh-cn-xibei2.ebcloud.com`，端口30375；已有认证 control socket `/tmp/kvmem-4090-reconnect-6maz_jyk/ssh.sock`。
- hostname：`cs-5a419-71c01-server`。
- GPU：RTX4090 24GB，UUID `GPU-4ae333ac-58da-f319-2442-e143ad673970`，driver580.65.06。
- 内存按 cgroup100GiB 限制计算，不能用 `free` 显示的宿主1TiB作为可用预算；无swap，CPU配额10核。
- 独立目录：`/root/data/cacheblend-memory-le1m-ab-20260928`，位于256GB数据盘。既有其他实验不改动。
- 模型复用 `/root/kvmem-hard256-20260927/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` 的只读使用路径，通过新目录models软链接访问；SHA256 `58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f` 已验证一致。
- 完整3714文件源码快照逐文件校验；CUDA12.2.140/GCC11.4/CMake3.31.6，`89-real`、Release、共享库构建成功。实际binary/library/model/source/hardware哈希见每个case引用的 `blend*/identities/*.json`，不要以旧3090根identity推断新case来源。
- 本机控制器：`/home/chaidi/kvmem-freetoken/benchmarks/cacheblend-memory-le1m-4090-20260928`。

模型参数仍 IQ3_S、MTP3、KV36864、logical context2097152、完整输入上限1048576、原输出/思考/采样参数，ratio仅0.1/0。全部1618预检长度保留，最大1038027，没有新增排除或截断；每个新请求再次检查渲染长度/hash。

监督器先on后off，独立上下文之间重启server，同一MAB上下文问题顺序和warm prefix保留。每5秒记录GPU/内存，host/cgroup内存余量任一低于4GiB时只停止自身子进程，保留在途证据，停队列等待处理。不能因答案好坏重试；response存在但result缺失时必须恢复已有response。

## 运行、同步、评分

启动时监督器PID8315（build脚本exec进入run_ab），初始controller14193/server14206；后两者随上下文重启变化，以状态文件为准。本机新同步PID3089026，既有正式judge PID3068073保持运行，密钥仅进程内存中，不写文件、不传远端。**不要重启旧waiter或旧3090同步。**

4090状态：`ab-status.json`、`blend*/status.json`、`execution_batches.jsonl`、`resource-usage.jsonl`；本机 `sync-4090.log`、`judge-ab-status.json`、`remote-4090-tail/`。完整server日志仍保留4090。

同步仅接受新完成case，先放identity再原子发布result，校验1208份固定结果；不覆盖本机judge缓存。原3090根identity保留，4090新identity进入按hash索引目录。

正式评分仍由本机 `cacheblend-memory-le1m-ab-20260928` 的score/compare执行。原LME102已正式评83/102=81.37%，校准12/12。MAB一题 `Accurate_Retrieval|19|15` 在历史1024 judge预算下无verdict，仍pending；用户尚未授权更改judge预算，不得静默改成4096，也不计错。

所有1618问进行配对正确率比较并明确PRO6000/3090/4090混合硬件；另报两组都由同一4090产生的410问。仅相同GPU identity/host及实际cache状态、cached token数量匹配时才可比较时间，不做跨设备延迟相减。MAB只报三个≤1M覆盖类别，不报原四类总分。

迁移/验证详见本机结果目录 `migration-manifest.json`、`provenance.json`、`cutover-4090.json`、`evidence/migration-4090/validation.json`。完整推理和正式评分仍待队列完成；不得把部署完成写成评测完成。

## 首问验证及元数据修复

4090首问于21:53:56完成，耗时246.859秒，269918输入tokens，CacheBlend事件1次，result SHA256 `923f5bcb9dc7e0eda09b92f6a59b488781f8b9211a5af5028b0cacfdb7a50805`。已回传本机并完成对应正式judge。此时on1209/1618，on剩余409，off仍1618待跑。上下文结束后显存清空、容器内存余量恢复约83GiB。

首次跨进程续跑发现suite计数dict顺序随Python hash seed改变，导致唯一差异input_manifest_sha256触发身份校验。已改为按suite排序序列化，两种hash seed验证一致；新增controller identity，旧identity及首问答案保留，不重跑首问。此后监督器PID14378，正式judge新PID3092023，同步PID见`sync-4090.pid`。judge已改为在推理基础设施暂时停顿时等待修复，凭据仍仅内存，未写文件。

## 2026-09-29 22:07：按用户要求暂停 on，改跑已完成样本的 ratio0

本节取代上文“on续跑到1618、off全跑1618”的活动队列安排。用户要求立即停止ratio0.1，在目前已完成的相同样本上运行ratio0。监督器及其自有controller/server已停止；冻结1489个完成result：LME102、MAB1116、ALB≤256K250、512K20、1M1。剩余129个on样本暂停，不计答错，不再自动续跑。停止时在途512K题没有response/result，元数据保留在evidence/paired-completed-20260929/，未加入配对集合。

新增独立不可变active-paired-cohort.json，SHA256 076009c9775c36fd4ff8d7ff8b8dcbda39c9f07bb917438fd465ba6081521884，逐题绑定1489个on结果及答案哈希。原selection-manifest.json的1618题、原始答案、原迁移cutover均保留。新scope只按停止时完成状态筛选，不按答案得分筛选。两组同为4090的配对子集现为281题；其余仍为混合硬件准确率比较。

远端新监督器PID54992只运行blend0，目标1489题。模型、binary、源码、MTP3、KV36864、逻辑2M、输入/输出上限、提示及采样参数不变，命令逐项验证仅ratio由0.1改为0。MAB所有1116题均保留，原问题顺序和上下文warm-prefix协议不变。

本机同步与正式judge已改为识别新scope；judge凭据仅从旧进程内存移交新进程环境，没有存盘或传到4090。实时PID见结果目录judge-ab.pid及sync-4090.pid。compare.py按1489个配对样本统计；未完成的129个on样本不再作为当前任务待办。原MAB一题judge预算耗尽仍记录为pending，按用户此前指示只在暂报中排除，不伪造判定、不改变judge预算。后续不能把有限样本报告写成原1618全量结论。

部署证据、旧控制器备份、停止元数据及验证记录见evidence/paired-completed-20260929/。截至启动时ratio0尚待产出首个结果；完成状态必须以实时status与结果文件为准。

ratio0首题LME e47becba于2026-09-29 22:13:24完成，108368输入tokens，78.246秒。startup blend_ratio=0，result blend_events为空；与on题目的prompt/token SHA一致。验证证据：evidence/paired-completed-20260929/first-ratio0-validation.json。

## 2026-09-29 22:38：修复 ratio0 首批评分目录问题

ratio0首个LME先于MAB完成，score.py调用冻结MAB评分器时输出目录尚不存在，导致judge在三次失败后退出。已在调用前创建输出目录；watch模式连续失败时保留进程与内存凭据，等待修复后重试。评分协议及冻结评分器未改动。ratio0本地评分和compare验证成功，正式judge重启PID4118483，推理与同步未中断。证据见evidence/ratio0-scoring-repair-20260929/。

## 2026-09-29 22:41：用户要求 ratio0 优先完成 LME

新增queue-priority.json，优先longmemeval_s。停止旧监督器54992及其自有server，当前未完成的ALB1M题af004d2443fcd470b168453ff3553d666b3001456ffa55a21726a6850725af0b没有response/result；请求元数据及中断trace归档evidence/lme-priority-20260929/，之后可在LME完成后继续运行该题。四份已完成ratio0结果逐字节校验保留。

新监督器55852首先运行剩余101道LME（已完成1/102），当前下一题118b2229。总配对清单仍1489，其他套件保持原相对顺序，MAB上下文内顺序不变，推理参数不变。同步PID4120943；正式judge继续4118483。LME跑完后自动继续余下MAB/ALB。此优先级变更只改变调度，不按得分筛题、不重跑完成答案。

## 2026-09-30 09:52：用户明确要求暂停4090测试

4090监督器55852及自有server9147均已停止，ab-status标记paused，user-pause.json记录用户暂停及恢复约束。ratio0已完成1214/1489：LME102、MAB1109、ALB≤256K1、512K1、1M1；ratio0.1保留1489。当前在途MAB摘要023_Long_Range_Understanding_r035_infbench_sum_eng_shots2_q0000已中断；请求、response（若存在）及日志证据保留于evidence/user-pause-20260930T095230+0800。全部2703份完成结果已同步至本机并按SHA256核验，旧结果未改写。本机同步已停止；正式judge可继续对已保存答案评分，不消耗4090。未经用户明确恢复指令不得重启推理，恢复前先检查在途response并优先恢复已有答案。
