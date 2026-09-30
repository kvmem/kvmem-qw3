> 已被最新方案取代：本机已停止，输入限1,048,576 tokens，排除ReDial200问。请读 [≤1M最新交接](/home/chaidi/qw3/docs/memory_eval_le1m_handoff_20260928.md)，不要按本文旧1818问范围直接恢复。

# Memory benchmark 实验交接：SUFE-3090，10% 重算与关闭重算对照

交接快照：2026-09-28 15:34:06（北京时间）。接手任务先读本文，再读取实时状态。

## 1. 目标与已授权范围

用户要求将剩余 memory evaluation 移至 SUFE-3090，并增加关闭重算对照。保留本次统一 **36,864 KV history budget**、本次模型及其他推理配置，只对齐历史样本、提示词和评分协议。

本轮仅编写交接文档与快照；**没有停止本机实验，没有搬迁文件到3090，没有在3090编译或启动实验，也没有启动关闭重算组**。之前已验证从当前Linux机器能直接SSH到目标服务器。

接手后的具体任务：

1. 安全迁移当前 `kvmem-llama-cacheblend` 实验，续跑10%重算组尚缺的样本。
2. 新建独立的关闭重算组，运行同一份完整1818问清单，包括LongMemEval-S 102题，以便对全部已有/新增10%结果逐题比较。不要只跑两组中容易成功的子集。
3. 用历史评分协议正式评分两组。LongMemEval当前只有Codex语义复核，DeepSeek尚未正式评分，应先补上已有102题的judge。
4. 输出各benchmark分项结果、配对差异、失败与未评分数量及延迟；标明硬件来源。

不是qw3引擎；不是此前大海捞针任务；不再跑S500全量；不增加50/50混合组。

## 2. 服务器与访问现状

用户提供的Mac SSH配置：

```sshconfig
Host SUFE-3090
    HostName 10.1.139.69
    User dell
    IdentityFile /Users/dichai/.ssh/id_rsa.txt
    Port 22
```

当前Linux工作机没有 `/Users/dichai/.ssh/id_rsa.txt`，也没有 `SUFE-3090` 别名配置。但已有认证可用，以下命令已成功（无需复制或输出私钥）：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=yes -p 22 dell@10.1.139.69 'hostname; nvidia-smi'
```

检查到：

- 主机名 `dell-Precision-3680`；用户目录 `/home/dell`。
- RTX 3090，24,576 MiB显存；检查时仅36 MiB占用。
- 驱动570.86.10；主存约62 GiB、可用约56 GiB；磁盘可用约160 GiB。
- `cmake`、`rsync`、`python3`可用；`nvcc`不在默认PATH，CUDA工具链尚未进一步验证。
- 主目录存在其他kvmem历史实验，不能覆盖、删改或停止它们。新建独立目录，例如 `/home/dell/cacheblend-memory-ab-20260928/`（尚未创建）。

当前PRO6000编译目标是SM120a/CUDA13；3090需要SM86及其驱动兼容的工具链。**不要直接把PRO6000二进制当成3090可执行文件。** 使用冻结的同一份源代码、现有overlay和子模块构建，记录编译器、CUDA、架构、共享库与新二进制SHA。显存和长上下文内存可行性尚未验证，不能声称已经适配成功。若必须改变KV预算、KV精度、模型、输入或重算策略，先明确影响，不能悄悄降低配置。

## 3. 最新快照与实时进度

| Suite标识 | 全量 | 已完成10%组 | 快照剩余 |
|---|---:|---:|---:|
| longmemeval_s | 102 | 102 | 0 |
| memoryagentbench | 1316 | 577 | 739 |
| agentlongbench_le256k | 250 | 1 | 249 |
| agentlongbench_512k | 100 | 1 | 99 |
| agentlongbench_1m | 50 | 1 | 49 |
| 合计 | **1818** | **682** | **1136** |

本机仍在推进，因此这些是快照数。迁移时重新按 `(suite,id)` 和完整 `result.json` 盘点，不按旧PID或旧summary简单续接。不要重跑LongMemEval已完成的10%组来挑选更好的答案。

本机路径：

```text
源码仓库：/home/chaidi/kvmem-freetoken/kvmem-llama-cacheblend
控制器目录：/home/chaidi/kvmem-freetoken/benchmarks/cacheblend-memory-pro6000-20260928
结果根目录：/data/chaidi/kvmem_eval/results/cacheblend_memory_pro6000_20260928
交接快照：/data/chaidi/kvmem_eval/results/cacheblend_memory_pro6000_20260928/handoff_3090_ab_20260928
```

最近验证的进程（须重新核实命令行和存活状态）：

- controller PID 2982410：控制器目录下 `campaign.py`。
- server PID 2983010：`build-pro6000/bin/llama-kvmem-server`，本机端口18228。
- judge waiter PID 2942386：`grading_queue.py`，仍等待整个1818问队列完成。

重要文件：

- `status.json`：实时阶段/当前样本；`summary.json`：推理计数。
- `cases/<suite>/<id>/result.json`：最终回答、reasoning、finish_reason、usage、timings、query边界、prompt/token/query hash、重算日志。
- 同目录 `response.json`、`request_meta.json`、`trace.log`、`memory_after.json`：原始响应与验证证据。
- `identity.json`、`input_manifest.json`：当前版本模型、配置、代码与输入身份。
- `events.jsonl`：含旧版本事件，不能仅数日志行充当新组完成数。
- `report.json`：独立评分报告，可能滞后；不能把其旧计数当作当前进度。

交接快照中 `jobs-snapshot.json` 含完整1818个 `(suite,id)`、已完成结果路径与SHA、回答SHA、prompt/token/query SHA及时间。`artifact-hashes.json` 校验快照文件；`controller/` 保存当前调度/评分代码副本。两个工作树diff与git status也已保存。**diff不包含所有未跟踪文件**，迁移应冻结完整源码及子模块/必要未跟踪源码，不能只checkout一个commit再假定一致。

## 4. 两个实验组：唯一计划算法变量

| 项目 | blend10 | blend0 |
|---|---|---|
| KVMem | 开启 | 开启 |
| `--kvmem-blend-ratio` | `0.1` | `0` |
| `--kvmem-blend-alpha` | `1` | `1`（不触发重算） |
| Query Replay | `auto` | `auto` |
| history budget | 36864 | 36864 |
| 其余输入/参数 | 相同 | 相同 |

关闭重算用 **ratio=0**。不要以 `alpha=0` 代替，不关闭KVMem、Query Replay或MTP，不改为native full-context。源码 `src/adapter/llama-memory-kvmem.cpp` 的blend入口在ratio为0时直接返回；CLI接受0..1。

保留配置（精确服务命令以快照 `identity.json.command` 为准）：

| 配置项 | 值 |
|---|---|
| 模型 | Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf |
| 模型SHA256 | `58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f` |
| 本机模型路径 | `/data/chaidi/models/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` |
| main KV / MTP KV | q8_0 / f16 |
| MTP | draft-mtp，spec-draft-n-max=3 |
| history / generation reserve | 36864 / 32768，两个独立概念 |
| logical context | 2097152，覆盖历史最长约1.665M-token上下文 |
| block / sink / recent | 32 / 1312 / 10496 |
| batch / ubatch | 1024 / 1024 |
| CPU参数 / GPU层 / threads | kvmem-cpu-gb=0，ngl=99，t=8，tb=8 |
| Flash attention | on |
| query replay / policy / max | auto / user / 32768；请求显式query span |
| MTP state | replay |
| blend | ratio=.1或0，alpha1，无neighbor扩展，state reset-restore |
| sampling | temperature=.6，top_p=.95，top_k=20，min_p=0，repeat_penalty=1，presence_penalty=0，seed=20260722 |
| LME/ALB | thinking=true，reasoning_budget_tokens=4096，max_tokens=32768 |
| MAB | thinking=false，reasoning_budget_tokens=0，按family输出上限 |
| 环境 | KVMEM_BENCHMARK_API=1，KVMEM_TRACE=1，CUDA_VISIBLE_DEVICES=0 |

MAB family输出上限保留runner原值：`ruler=50, eventqa=40, longmemeval=50, fact=10, infbench=1200, detective=2000, icl=20, recsys=300`。不要用统一输出上限替代。

路径/端口可适配远端，其余命令逐字段比较。当前launcher清除HTTP/HTTPS/ALL_PROXY大小写变量及QW3_*，保持该行为。QW3_*不是本次llama.cpp重算开关。

## 5. 历史样本与提示词必须保持

### LongMemEval-S

- 固定102题，六类各17题，无abstention。
- 本机源 `/home/chaidi/qw3/selected_12_samples.jsonl`，虽文件名写12，实际102。
- 冻结输入 `inputs/longmemeval_matched102.jsonl`。
- SHA256：`7bdc816736e4505d16d0fa196415ee32cb63314e9f64b7341a50ba76f4ec888c`。
- 不使用 `dataset.build_subset()` 的排序抽样；不从S500只按ID替换内容。历史102中89题的会话记录与旧S500文件不同。
- 用冻结 `dataset.load_all` 解析；不要用 `str.splitlines()` 粗拆含Unicode行分隔符的JSONL。
- `historical_protocol.py` 从 `frozen/historical_lme_reader.py` 提取历史SYSTEM、sessions、final_message。三条messages为 system / dated history user / question user。
- 102题消息正文已逐题对齐历史；当前query span仍按本次runner的最终问题message计算，保留两组一致。
- 原S500已完成39题在 `revisions/matched102_20260928/s500_longmemeval_cases`，不能计入这102题。旧36题/100%等语义复核同样属于归档组。

### MemoryAgentBench

- 冻结 `inputs/memoryagentbench/<row>/` 中五类文件：prepare_manifest、row_summary、qa、questions、archive_prefix.qwen-chat-open-user.txt。
- 固定历史 >256K 的30个上下文、1316问，四大类完整保留。
- 历史参考目录 `/home/chaidi/qw3/benchmark/memoryagentbench_9b_matched27b_20260916/`，其中 `reference_lock.json`、`metric_contract.json` 说明来源与汇总方式。
- 原system + 单条user(history+formatted_question)；raw_question用于query span。不重复套历史chat标记，不把gold放进请求。
- 同一上下文多问使用当前prefix checkpoint/rollback路径；保留顺序与cache_reset策略。换机器/重启后第一条缺失问冷启动，记录这一点。

### AgentLongBench

- `inputs/agentlongbench/{le256k,512k,1m}/{samples.jsonl,manifest.jsonl}`。
- 原long250 / normal100 / DeepseekMillion50，依stable_sample_id配对，不按文件序号近似配对。
- 通过冻结 `canonical_alb.full_context_prompt` 生成原单user prompt，核验存在的prompt_sha256；`query_byte_span`定位问题。
- 冻结canonical evaluator直接评分最终answer，不能把reasoning当答案。

ALB与MAB的156个冻结输入/清单文件已和历史来源逐文件SHA验证一致。重新部署时再核验；不重新下载随机生成替代数据。

## 6. 现有runner不能直接用于远端双组：必须做的小改动

先冻结当前脚本和身份，再在新目录做迁移版本，避免改坏仍在跑的本机进程。

1. 参数化 `REPO/MODEL/BINARY/OUT/BASE/port`，输入根、Python解释器和每组ratio。输出独立 `blend10/`、`blend0/`；两组同一3090二进制、相同源/权重hash。
2. `prepare()` 当前硬编码本机历史来源。远端优先校验已复制的冻结inputs及manifest，而非再去不存在的 `/home/chaidi/...` 生成输入。subset SHA、六类各17、全量1818断言保留。
3. `jobs()` 生成所有1818个唯一键，再以该组有完整合法结果的键做差集；不要把blend10结果当blend0的完成标记。已完成10%可通过只读复用清单引用或带来源复制，保留原设备/二进制身份。
4. 重算日志校验现在写死长输入必须有`KVMEM_BLEND`且ratio=.1，必须按arm区分：blend10保留原校验；blend0核验有效ratio=0且无实际重算事件。不要为通过off组而删除两组所有验证。
5. `identity.json` 对源码SHA、模型、命令、binary、库等有严格resume检查。远端是新硬件/新二进制身份，应新建identity并记录来源链、迁移差异和复用依据；不要改旧SHA来绕过检查。
6. 当前 `--limit` 的运行末尾会标成 `limit_complete`，不能冒充全组完成。两组各1818最终结果与正式评分完整才算完成。
7. scorer内 `c.OUT`、MAB历史reference目录、解释器路径需适配。也可将远端原始结果同步回本机，使用本机冻结评分环境；为每组独立输出和judge缓存。不要改评价函数的语义。
8. 当前 `summarize()/local_score()` 会写共享文件；推理运行时单独评分优先用LME-only入口，避免多个进程写同一summary临时文件。多组各自目录/锁，评分缓存按输入、答案hash和judge身份验证。
9. 保留反作弊/失败原则：评分前冻结答案；不按正确性挑选重试；模型输出错误、拒答、空答计入本组分母；基础设施错误独立归档、确认无完整答复后再恢复。

可先抽取每个suite一条预先固定的样本验证接口/资源/日志；这些答案属于所选arm正式结果，无论对错都保留，不能按通过情况换题。build/model-backed blend regression与正式样本分开，日志留档。

## 7. 安全迁移和执行顺序

1. 重新检查当前机器和3090进程/显存/磁盘，阅读新目录适用的AGENTS说明。冻结源代码、模型、输入、配置清单、现有控制器和已有结果hash；传输只含实验必需数据，不含SSH私钥或API key。
2. 在3090独立目录编译、核验模型SHA和输入SHA，安装冻结runner依赖，验证render/tokenize/memory端点与采样实际生效。不要下载同名但不同量化/不同权重代替。
3. 远端预备完成后，再协调本机控制器于完整样本保存边界停止。当前runner没有stop-file接口；若必须中断在途请求，记录其ID并归档请求/已有response/日志片段。存在完整response但缺result时先恢复解析，不能直接重生成。
4. 停止后再制作最终完成清单、同步最后新增结果，核验无同一 `(arm,suite,id)` 被两台机器重复执行。不要杀其他任务。
5. 在3090优先续完blend10缺失部分，再运行blend0完整1818问；如改为按suite交替，也须每arm独立server/cache并保留MAB同context顺序。单卡不并发两个大模型server。
6. 最终结果回传本机的独立arm目录，保留原10%目录完整，保留每条result的execution_host/build身份。维护迁移映射清单，不能把PRO6000结果标成3090生成。
7. 旧judge waiter只监控原根目录的`status.json`；迁移后不会自动理解远端新目录。明确安排评分接续，不伪造原目录`inference_complete`来触发它。只有真实全组完成才写完成阶段。

重启MAB中途context会导致首条缺失问重新prefill。必须记录cold/warm/prefix-hit与迁移边界，不能将冷启动延迟误认为重算开销。不为了计时重生成已完成答案。

## 8. DeepSeek正式评分与当前障碍

### LongMemEval

使用历史DeepSeek Flash协议：model=`deepseek-flash`，thinking disabled，temperature=0，max_tokens=16，stream=false；官方每题型judge prompt；只发送问题、reference/rubric和最终answer，不发送method标签或reasoning。保留12例历史校准。只接受正常stop且严格yes/no；错误、截断、非二值留待评分，不计为答错。实际返回model要验证并记录。

- `score.py:grade_lme` 已接入上述协议；保留历史模型返回检查，不因接口失败静默换成Pro。
- `judge_longmemeval_now.py` 已准备并通过102题preflight，可独立正式评分当前完成的102题，不碰GPU推理。
- **还没有发送正式DeepSeek评分请求**。现有waiter PID2942386只在内存持有凭据，没有提前触发接口；当前交互环境没有`DEEPSEEK_API_KEY`。不要读取任意进程内存或在文档/日志保存密钥。
- 用户已授权使用DeepSeek；接手环境有环境变量即可直接运行，否则本机终端隐藏输入：

```bash
/home/chaidi/qw3/.venv/bin/python /home/chaidi/kvmem-freetoken/benchmarks/cacheblend-memory-pro6000-20260928/judge_longmemeval_now.py --prompt-key
```

- 结果：原根目录 `judge_deepseek_v41_flash/{standalone_status.json,report.json,report.md,scores/,requests/}` 及 `longmemeval_judgments.jsonl`。各arm需独立。
- 成功judge缓存校验答案SHA；错误请求重试不等于允许重生成reader答案。

### MemoryAgentBench

- 原官方本地scorer + `judge_memoryagentbench_special.py`，特殊LongMemEval/InfBench子集用历史 **DeepSeek V4 Pro**。
- 主指标为四大类别宏平均：题目先汇总到source，各source等权汇总到category，四category等权。ReDial用Recall@5；InfBench为summary F1等原指标。
- 不用仅含1100个binary题的accuracy冒充全1316问结果。区分题量、空答、待judge和各source覆盖。
- scorer参考输入 `/data/chaidi/kvmem_eval/results/memoryagentbench_over256k_raw_query_k64_b32_semantic_chunk_20260812_110226`；迁移需复制必要的reference manifest或在原机评分，不能因为路径缺失重建不同选择。

### AgentLongBench

冻结 `canonical_alb.evaluate_response`，分别报告250/100/50三档以及原任务分项。保留exact等原指标定义，不替换为LME judge或统一substring评分。

## 9. 已有分数仅供定位，不是正式judge

102题10%组已于2026-09-28 13:00:39完成，102个finish_reason均stop、0空答。

Codex语义复核暂计84/102=82.35%，不是DeepSeek成绩：

| 类型 | 暂计正确/总数 |
|---|---:|
| single-session-user |17/17|
| single-session-assistant |8/17|
| single-session-preference |14/17|
| multi-session |13/17|
| temporal-reasoning |15/17|
| knowledge-update |17/17|

4个边界暂判正确：58ef2f1c、e9327a54、3a704032、gpt4_2f8be40d。全部判错时80/102=78.43%；这不是统计置信区间。正式judge应盲评全部102题，不只裁判这4题，也不以该复核分数为预期目标。

复核产物：`partial_reviews/matched102_20260928T072104Z/`。与归档S500结果严格区分。

## 10. 结果比较与验收

硬件切换需要标注：现有部分blend10来自PRO6000，之后来自3090，而新blend0计划在3090。即使输入/模型/参数相同，不同架构/内核的浮点和采样轨迹也可能不同。

交付至少两种比较：

1. **完整1818问配对结果**：按 `(suite,id)` 配对，明确blend10混合硬件来源，不称纯单变量实验。
2. **同3090配对子集**：只包含两个arm都在同一3090构建下生成的共同题目，报告数量和题型覆盖。不要挑选效果好的子集。现有LongMemEval10%全部来自PRO6000，因此目前没有LME同3090对照子集。

不得直接把PRO6000和3090耗时差解释为重算开销。耗时分析仅同硬件、同构建、同冷暖cache口径；报告wall time、实际输入/输出token数、已记录的prefill/decode和blend时间。可比性不足时明确留空，不推算成精确开销。

若之后需要严格同3090的完整1818问单变量AB，需要额外完整重跑已完成10%部分；这不包含在当前“保留已完成结果、续跑剩余”的默认迁移范围内，先明确额外工作量再决定。

验收文件建议：

- `migration-manifest.json`：源码/权重/输入/二进制身份、硬件、复用结果hash、迁移边界。
- 每arm `status.json/results or cases/summary/report`，各1818唯一最终答案或显式终态失败；judge覆盖单独核算。
- `paired-results.csv`：输入hash、两组设备/答案hash/score、仅10%正确/仅0%正确/均正确/均错；非binary任务保存原连续分数与delta。
- `comparison.md`：LME总体及六类、MAB四类与source、ALB三长度与任务，数据/模型/预算/judge标签齐全。
- 无未说明的duplicate/遗漏/输入漂移；off组确实ratio0且无重算，on组确实ratio.1并满足原trace断言。
- 所有实验终态与正式评分完成后才结束，不把Codex语义复核代替DeepSeek，不自动归档用户会话或新建周期任务。

## 11. 可粘贴给接手任务的指令

> 请先读 `/home/chaidi/qw3/docs/memory_eval_3090_ab_handoff_20260928.md`。按文档将kvmem-llama-cacheblend的剩余memory评测迁移到SUFE-3090（dell@10.1.139.69）。保留当前Qwen3.8-27B IQ3_S、MTP3、36,864 KV预算和其他推理参数，历史样本/提示/评分不变。10%重算组只续跑缺失结果，另建ratio=0关闭重算组跑同一完整1818问清单，包括LME102。先检查本机仍在推进的队列和远端环境，完成清单以迁移时实际结果为准；保留旧结果和hash，不重复挑选答案。正式用历史DeepSeek协议评分两组；当前LME的84/102仅是Codex复核，不是DeepSeek。交付全量配对比较和同3090配对子集，并标注设备来源及计时限制。用户已授权服务器迁移和新增off对照，不必再次确认这些操作；如需改变模型/预算/输入/策略，明确说明影响再确认。

## 12. 补充核验：超过256K可运行，但3090主存可能限制最长样本

当前服务实际逻辑context为2,097,152，并校验prompt_tokens+max_tokens不超过此值；36,864为GPU选中历史窗口，不是总输入上限。PRO6000上已经完成实际输入516,349-token的ALB512K、1,030,722-token的ALB1M，以及755,778-token的MAB样本，均正常stop。这仅证明原机器执行成功，不证明3090资源足够或答案正确。

特别注意3090机器约62GiB系统内存：实际1,030,722-token样本生成后，memory_after记录stored_tokens=1,031,730、CPU KV=40,107,507,712bytes（37.35GiB）、CPU index=2,113,011,712bytes（1.97GiB），另有checkpoint约0.44GiB。`--kvmem-cpu-gb 0`不能理解为整个实现不使用CPU存储，实际历史KV和索引仍占主存。

MAB最末recsys_redial_full的旧准备清单有archive_tokens_unpadded=1,665,281、padded=1,667,072；当前模型实际token长度尚需render确认。按已测1M样本线性估算，1.665M的CPU KV+index约63.46GiB，尚未计模型映射、Python输入、临时buffer、系统等。该估算不是实测峰值，但已经超过目标主机总物理内存，故**不能承诺原参数下3090单机可顺利完成最长ReDial**。swap也不是已验证解决方案。

迁移前优先以相同tokenizer做最大输入长度/内存测算并检查后端真实存储增长；若不足，优先让超长样本的两个arm都留在内存更大的原机器，按执行设备分别配对报告。不要截断输入、缩短测试或调低KV预算来假装完成全量。确需更换存储路径或其他执行策略应说明算法/性能影响并记录新配置。
