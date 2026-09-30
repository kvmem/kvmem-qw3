# QW3 接入 Qwen3.8-27B DFlash 2 的评估

日期：2026-09-24。评估对象是当前工作区（HEAD `4954b871aa37b3e8530aef1cfaade2c2bc3d60f3`，包括已有未提交改动），不是只审查 HEAD。本文只做源码、公开模型配置和权重头部审计；未实施推理改动、下载完整草稿权重或运行性能测试。

## 结论

建议支持，而且直接以 **DFlash 2** 为第一目标。已经存在与 `Qwen/Qwen3.8-27B` 匹配的公开 checkpoint，不需要先自行训练。合适的接入方式是在原生 CUDA runtime 中新增一个草稿后端，复用主模型的批量验证、接受/拒绝、混合模型状态回滚及服务调度。

DFlash 和内置 MTP 都可以提出多个 token，但两者不是相同的模型格式。当前单层、逐 token 的 Qwen MTP 执行器不能通过换权重变成 DFlash 2。也不建议为此将整个 QW3 生成流程委托给另一套推理引擎，否则现有 KVMem 和状态管理难以保留。

优先级建议：单请求文本正确性 → 随机采样 → KVMem/前缀复用 → continuous batching。DFlash 2 的动态卷积与 selector 从第一版就应实现；只有传统 DFlash backbone 的版本不能称为完整 DFlash 2。

## 已核实的上游状态

- 官方发布名称为 DFlash 2，发布于 2026-08-18；本次确认的匹配权重为 [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)，模型卡说明它是 `incoai/Qwen3.8-27B-DFlash2` 的镜像。
- HF revision：`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`。配置为 `DFlash2DraftModel`；独立的单文件 `model.safetensors`，81 个 BF16 tensor，1,924,404,480 个参数，不含 target embedding / LM head。参数数量通过权重 header 和 HF API 交叉核对。
- [官方配置](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/blob/50307d4c4cde6860d4eee73e2547cd786fe8e8a4/config.json)：5 层，hidden 5120，FFN 17408，32 个 Q heads、8 个 KV heads、head dimension 128；非因果 sliding attention，窗口 2048。块长 8，即一个 anchor 加最多 7 个 draft token。target layer ids 为 `[5, 19, 33, 47, 61]`，selector top-k 16、rank 256，动态卷积 kernel 2、group size 16，mask token id 248070。
- [官方实现](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py) 可作为算子数值参考。其 Transformers 生成后端当前未声明支持 Qwen3.8 混合主模型，不能把该脚本直接当作完整 Qwen3.8 端到端基准；完整基准优先选支持混合模型的 serving 实现。
- 已检查 SGLang main 的 [`DFlash2DraftModel`](https://github.com/sgl-project/sglang/blob/9f6fc553322e897a22f30320dae750f4a40a0b84/python/sglang/srt/models/dflash.py)、vLLM main 的 [DFlash 2 模型代码](https://github.com/vllm-project/vllm/blob/cf78797cb79e73bb798746c6cce7c34f56860a0a/vllm/model_executor/models/qwen3_dflash2.py)，以及 llama.cpp master 的 [conv/selector 实现](https://github.com/ggml-org/llama.cpp/blob/9710a32175b3b8f04636aaac4aa3cc28651b505d/src/models/dflash.cpp)。上线参考应锁定验证过的 commit，不能将首发博客里的旧 PR 安装命令视为当前唯一方案。

## 当前框架能复用什么

| 环节 | 当前实现与证据 | 接入判断 |
| --- | --- | --- |
| 主模型执行 | `src/qwen_executor.cpp` 中 `forward_one_token` / `forward_n_tokens` | 保留 Qwen3.8 target；新增选定层的隐藏状态输出 |
| 批量验证 | `src/qwen_executor.hpp` 的 `forward_n_tokens` 已输出逐行 argmax / logits | 复用 anchor + draft 的验证格式 |
| 混合状态回滚 | `StateSnapshot` / `StateCheckpointSet` / `restore_state_checkpoint` | 复用 target attention KV、DeltaNet、conv 的回滚；追加草稿状态边界 |
| 采样 | `src/qwen_native_backend.cpp` 的 `speculative_accept_pointmass` | 适用于确定性草稿；完整随机 selector 需要一般 q 分布版本 |
| 基础算子 | `include/qw3/device_backend.hpp`、`src/kernels_cuda.cu` 已有 BF16/Q8 linear、norm、embedding、RoPE 等 | 复用基础算子，不复用完整 MTP 层 |
| 草稿权重 | `QwenMtpWeights` 绑定一层与 `nextn.*` | 新增独立 DFlash 配置和权重对象 |
| 草稿 attention | FlashInfer prefill 适配器现有路径使用 causal mask，部分模板固定 target head dimension | 新增 128-dim、无 Qwen3.5 attention gate、非因果块与 sliding mask 的路径 |
| 服务与 KVMem | 已有 MTP continuous batching、paged KV、prefix cache 和 working-set generation | 可复用基础设施，但各调用点对 MTP 的直接依赖需要拆开 |

当前 `src` / `include` / `scripts` / `docs` 未发现已有 DFlash 实现。框架存在 `qwen3` 的部分 GGUF 支持，也不等价于支持带特殊输入、卷积和 selector 的 `DFlash2DraftModel`。

## 建议的实现边界

### 1. 独立草稿后端

新增 `DFlashConfig`、`DFlashWeights`、`DFlashExecutor`，由 `SpeculativeDecoder` 编排；将现有 MTP 包成另一种 drafter。第一步只提取必要的调用边界，避免同时重构整个大型 backend。

建议接口语义：

```text
observe_target_features(committed_rows, positions)
propose(anchor, max_draft_tokens) -> tokens + optional sparse proposal q
commit(accepted_target_rows)
invalidate(reason, position_frame)
snapshot / restore                 # 前缀缓存、请求恢复需要时
```

配置可设计为 `--speculative-method none|mtp|dflash2`、`--draft-model PATH`、`--draft-block-size N`，旧 MTP 参数继续兼容。以上为建议接口，目前尚不存在。运行时块长不得超过 checkpoint 支持的 8；可以对 4、6、8 做性能扫描，尾部按剩余输出长度缩短。

选择 DFlash 时不应要求 target 带内置 MTP 权重，也不应默认同时分配整套 MTP prefix KV。启动校验 target 架构、层数、hidden/vocab、tokenizer 与 checkpoint 身份，并记录量化格式；只校验 tensor 形状不足以保证 draft 与 target 匹配。

### 2. 权重加载先走 BF16

现有 HF 路径在 `src/model_source.cpp` 要求 `model.safetensors.index.json`、`compressed-tensors` 和 `text_config`，`src/qwen_config.cpp` 也要求 `qwen3_5_text`。官方 DFlash 文件不满足这些要求。因此需要单独的 safetensors source / binder，复用现有低层解析、上传及 BF16 GEMM。

需要覆盖 `fc`、`hidden_norm`、5 层 attention/FFN/norm、每层两套动态卷积、selector 的两张 codebook 和 hidden projection。注意 HF codebook tensor 名称没有 `.weight` 后缀；norm 也应按 DFlash/Qwen3 语义处理，不套用 Qwen3.5 的权重变换规则。

target 的 embedding 和 LM head 直接借用。QW3 现有 Q8/BF16/NVFP4 dispatch 应作为共享 head 的实现入口，不需要为了草稿另存一份全词表 BF16 head；后者会额外占约 2.37 GiB。

官方也提供 [Q8_0 GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/tree/main)，可作为第二阶段降低显存与带宽的路线。但现有 `QwenConfig` 不接受 `dflash` architecture，当前 GGUF source 的 dtype 映射也仅覆盖 Q8_0/F32，需要独立的 metadata/tensor binder 与实际 dtype 审计。Q4_K_M 目前不属于框架可直接运行的格式。

### 3. 主模型采集五层输出

在层的 attention、FFN 和 residual 完成之后采集 `[5,19,33,47,61]` 的输出，先明确这些是 0-based layer id。参考代码取 `hidden_states[layer_id + 1]`，这里容易发生一层偏移。

采集要覆盖 prefill、单 token decode 和 batch verify；continuous batching 的独立层执行路径后续也必须覆盖。现有 `h_` / `h_batch_` 主要保留最后一层，无法事后反推出五层特征。

只保存必要的 chunk / 最近窗口 / 本轮 verify 行，随后用 `fc: 25600 → 5120` 和 norm 得到草稿上下文特征。prefill 不应保留整条历史的五层 hidden；可以滚动保留末尾窗口，并对实际有效位置建 draft KV。

### 4. 完整 DFlash 2 执行图

每轮先将 anchor 与 mask embedding 作为一个块，结合已提交 target 特征，一次运行 5 层 draft backbone。草稿使用独立 KV；不能借用 target KV，后者的投影权重、heads、维度都不同。

必须实现两个新增模块：

- **分组动态卷积**：根据块内 hidden 生成系数，在 attention 和 MLP 子层的输入与输出两侧做卷积。这里是块内计算；不是将 target DeltaNet 的跨 token conv state 直接拿来复用。短块、padding、CB 中不同请求之间不得串扰。
- **候选 selector**：每个位置从共享 LM head 获取 top-16，通过前驱/后继 codebook 和 hidden projection 打分，再从 anchor 出发选择一条路径。可以先实现短循环，后优化 GPU kernel；无需再对每个候选位置做一遍完整 LM head。

attention 在块内允许看见其他草稿位置，并遵守 checkpoint 的 sliding mask、位置和 norm 定义。target 的验证仍保持因果 attention。不能只将现有 causal prefill 的开关改成 false，就认为完成了整个 DFlash attention。

### 5. 验证与采样

验证输入为 `[anchor, d1, ..., d7]`。第 i 行 target logits 验证下一个 draft，接受连续匹配前缀后，提交 anchor 和已接受 draft 对应的真实状态；拒绝后的 speculative suffix 不得进入下一轮上下文。

Qwen3.8 是 hybrid 模型，回滚不仅是截断 KV 长度，还包含 DeltaNet state、target conv state、位置、paged KV、KVMem window。已有 checkpoint 机制是重要的复用资产。DFlash 还要截断临时 draft KV、丢弃被拒绝行的五层特征，并仅将实际已提交 target 特征用于下一轮。

温度 0 时复用 greedy accept。温度大于 0 时可以分两步：

1. selector 仍确定性选路径，此时 q 是点质量，现有 `speculative_accept_pointmass` 仍能保持 target 采样分布，但接受长度不代表官方随机草稿性能。
2. 完整 selector sampling：保留每行 top-16 token ids 与条件概率 q，接受概率为 `min(1, p(d)/q(d))`；拒绝时从归一化的 `max(p-q, 0)` 采样，全接受时从最终 target 行取 bonus。必须使用实际经过温度等变换后的 proposal q，不能把 q 当成 LM-head 原始 softmax。

target 的 top-p、top-k、min-p、重复/存在惩罚与普通生成保持一致；惩罚按已提交前缀加本轮已接受 token 更新。最初可复用 host logits 验证；后续应避免每轮复制约 7.58 MiB 的 8 行 FP32 全词表 logits，以减少同步与传输开销。

“无损”指相对于同一权重、量化与上下文策略的 target。随机采样验收比较分布，不要求同一 seed 逐字一致。greedy 验收要求与普通 target 路径一致，并诊断 batch/单 token 的浮点差异；量化 target 上接受率变化不能解释成验证正确性的放宽。

### 6. KVMem 与前缀复用

草稿 sliding window 仅 2048，因此不应默认把它扩展成随完整 agent 历史增长的第二套 KVMem。保留独立有界 draft cache，记录已提交边界、位置映射和 working-set generation。

关键事件包括：新请求命中 target prefix cache、历史分叉、KVMem reselect / rebase、归档恢复、thinking compaction、请求取消。现有 target checkpoint 和 MTP 最后一行 hidden 并没有保存 DFlash 所需的五层特征；不能仅凭 target KV 命中就声称 draft cache 也命中。

首版保守策略：相容状态可恢复；缺失或位置帧改变时失效 draft cache，使用正常 target 路径建立新的特征，之后恢复草拟。需要重放时从合适 checkpoint 开始，不能在 live target state 上重复执行旧前缀。后续再增加 draft 最近窗口/投影 KV 的 checkpoint，以减少跨轮预热。

草稿历史不足本身可以只影响接受率，但 target 的回滚、检索决策、状态提交边界必须与不启用 speculation 时一致。KVMem 选集切换应发生在提交边界；不能让最终被拒绝的 verify 行触发持久检索/归档副作用。验收基线应为相同 KVMem 策略的 target，不能与 full-context target 混为一谈。

## 显存与性能评估

本次只读查询显示开发机为 NVIDIA RTX PRO 6000 Blackwell Server Edition，总显存约 96 GiB。以下是布局估算，不是运行峰值：

| 项目 | 估算 |
| --- | --- |
| 完整 BF16 draft 权重 | 3.58 GiB，全请求共享 |
| draft 5 层、2048 窗口、FP16/BF16 KV | 40 MiB / 请求，另加临时草稿尾部与元数据 |
| 最近 2048 token 五层 raw hidden | BF16 100 MiB / 请求；投影后单份特征约 20 MiB |
| 错误地保存 262144 token 五层 raw hidden | BF16 12.5 GiB；FP32 25 GiB，应避免 |
| target 每个额外 recurrent checkpoint | 仅 DeltaNet 约 144 MiB；7 个约 1008 MiB，另加 conv/hidden |

最后一项按 target 的 48 个 recurrent 层、48 个 value heads、128×128 state 和当前 FP32 checkpoint 计算。它是既有回滚机制的成本；相对 MTP-4 的新增部分取决于 checkpoint 策略，不应将整项全部算成 DFlash 新增。CB 下这些 per-request 开销会随并发增加。首版可用单个快照加拒绝后 replay 换显存，之后再优化逐行 checkpoint。

官方 H200/SGLang、concurrency 1、指定采样与任务集的结果，DFlash 2 相比 autoregressive 约 2.67–3.43 倍，按表中吞吐计算相对 MTP 约 1.32–1.48 倍；这是上游环境的参考，不是 QW3 的预期承诺。[官方模型卡](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2#evaluation)

QW3 的收益条件是：

```text
(draft 时间 + target batch verify 时间 + 回滚/特征/调度时间)
---------------------------------------------------------------- < 普通 target 单步时间
                    每轮平均推进 token 数
```

更高接受率不保证更快。仓库 `docs/mtp_paged_kv_optimization.md` 的 2026-06-16 历史测试曾出现长上下文 verify 主导、MTP 反而变慢；模型为 Qwen3.6，代码也早于当前版本，因此只能作为瓶颈提示，不能当作当前 Qwen3.8 性能结论。DFlash 减少串行草拟，但不会消除 target verify 对长上下文的读取。

## 实施与验收顺序

| 阶段 | 交付与验收 |
| --- | --- |
| A：单请求原型 | BF16 loader；五层 capture；含 conv/selector 的完整 draft；greedy verify/rollback；先用短文本、无 KVMem、无 CB。与上游逐层结果及同一 QW3 target 输出对齐。 |
| B：采样与性能 | sparse-q rejection sampling；top-p/top-k/penalties；GPU top-k/selector；测块长与接受长度，量化 draft 可在数值基准建立后加入。 |
| C：KVMem / prefix | warm continuation、分叉、reselect、rebase、archive、thinking compaction；已提交状态恢复后与无 speculation 路径对齐。 |
| D：CB / 多模态 | ragged row hidden capture，逐请求 draft KV / checkpoint，加入与取消隔离；多模态另验 image-prefill 与后续文本生成的位置及 embedding。 |

工程量判断：熟悉本仓库和 CUDA 的工程师，阶段 A 约 1–2 周；包含随机采样、KVMem 与 CB 的可用版本约 4–6 周。它是按本次代码审计给出的工程估算，不是完成承诺；性能调优和多模态完整兼容可能另需时间。

必要验证：

- 每个拒绝位置 0–7、全部接受、EOS/stop、短尾块，检查恢复后的 target state 和继续生成。
- 动态卷积与 selector 的算子对齐；注意 block 边界、mask id、norm 和 layer index。
- 随机采样用小词表可枚举分布及真实模型抽样检验一般 q 的接受/残差实现。
- benchmark 用同一 target 权重、量化、KV dtype、上下文和任务，对比无 speculation、已有 MTP、DFlash 2；上下文至少 4K/32K/128K，并发 1/4/16，再单独测 KVMem working-set 场景。
- 同时报告 TTFT、decode tok/s、端到端输出吞吐、平均推进长度、draft/verify/rollback 分段时间和峰值显存。不能只报告接受率，也不能只拿普通 decode 作唯一对照。

建议先用阶段 A/B 决定是否继续做深度整合。若代表性任务在相同条件下对已有 MTP 没有稳定收益，应先查 verify 和共享 LM-head 开销，再扩展 KVMem/CB 范围。
