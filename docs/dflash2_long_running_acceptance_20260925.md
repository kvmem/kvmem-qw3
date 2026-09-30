# DFlash 2：长任务单请求验收与转入并发的条件

状态：GPU 已释放并完成首轮部分重放；流式工具截断导致该轮停止。真实任务和非流式
完整重放正在串行执行。最新进度见 `dflash2_deepswe_first_gpu_20260925.md`。
本文记录用户 2026-09-25 的最新指令，
优先于旧目标中“继续追平公开 H200 绝对吞吐、补齐全部公开任务后才能转入并发”的要求。
原始结果、失败记录和冻结协议继续保留，不追认旧实验为新验收通过。

用户的新收敛条件：长历史下真实 long-running 任务（例如 DeepSWE）若能维持约
130+ tok/s 的 decode 吞吐，GPU 总占用控制在 48GB 以下，且整体正确性和状态行为
没有问题，就停止单请求速度优化，开始并发优化。GPU 空闲前只做 CPU 准备工作。

## 固定边界

- Qwen3.8-27B Q8_0 目标权重，E4M3 FP8 目标 KV；官方 BF16 草稿保持原样。
- 只在 `/home/chaidi/qw3-dflash2` 工作，主目录的任务、venv、历史轨迹可只读复用；
  新结果、脚本和冻结构建全部放本工作区。当前阶段不替换已有服务。
- 单请求先验收，多并发后置。不得关闭其他任务的 GPU 进程，不争抢显存。
- 135.536 tok/s 是 GSM128 完整请求任务窗口的历史记录，不是长历史 decode 结果。
  32K/W256 的 88.431 tok/s 同样不能代替真实 agent 长任务验收。

## 工作负载与配置准备

优先复用本地已固定版本的 DeepSWE 任务、Pier 0.3.1、mini-swe-agent 2.4.6、
官方工具循环及 verifier。任务选择必须在新模型结果出现前冻结，不根据已有本地成功率
挑题。不读取或向推理请求注入 solution/、hidden tests 或 verifier 答案。

先审计已有真实轨迹的长度和工具轮次，用固定输入自由续写定位各历史区间的吞吐与
内存。真实工具内容可重放，但须标记为 trajectory replay，不把它称为任务完成测试。
再运行真实 agent 工具循环和官方 verifier，检查多轮状态、格式、终止和任务结果。
没有达到长历史的任务仍保留，按预先冻结顺序补充任务，不能删掉慢请求或失败。

现有 48GiB A40 脚本采用 K64K/G32K，现有本地主线还存在 K128K/G64K 配置。
准备阶段需审计这两种配置的 GPU 索引、CPU 配额与当前实现，冻结本次主测配置；
不能默默用 256-token 小窗口替代实际长任务预算。DFlash 默认七草稿；32K/W256
筛出的五草稿尚不能直接推广到长任务。

## 指标与通过判据

1. 记录实际历史长度、GPU 工作窗口、生成预算、输出长度、终止原因、重放和刷新。
   按 <32K、32–64K、64–128K、128–256K、>=256K 分桶；未覆盖的区间明确标空。
2. 主指标是各长历史区间的总输出 token / 总 decode 时间，目标 >=130 tok/s。
   同时报告每请求 decode 的 p10/p50/p90、TTFT、完整请求耗时，以及检索/重放/
   草稿/验证时间。分别列出短响应，不能用单次高峰或请求 tok/s 的简单平均代替持续吞吐。
3. 长历史结论至少要有多条真实轨迹和足量自然输出；准备完成时冻结任务数、长度
   覆盖、输出量和重复规则。固定输入的 none/MTP/DFlash 对照与真实任务结果分别记录。
4. 显存以更严格的 48,000,000,000 字节为验收上限，同时显示 GiB。记录模型加载、
   prefill、decode、检索切换、恢复阶段的 NVML 观测峰值和采样间隔；区分进程占用与
   整卡占用，明确采样是否可能漏掉瞬时峰值。有 allocator 峰值时一并报告。
5. GPU 独占必须在运行期间监测，而不只在启动前检查。如果其他 GPU 进程进入，
   保留原始记录并取消受影响阶段的性能资格，不把共享 GPU 数据当作正常性能。
6. 保留已有 greedy 逐 token、拒绝回滚、CPU/GPU 采样与恢复检查，并针对长任务
   覆盖实际发生的 query replay、历史增长、取消/恢复和刷新。新任务的 verifier 结果、
   截断、超时、格式错误与循环全部报告；不以响应文本相同或 verifier 进程 exit=0
   冒充任务通过。

满足这些用户关注的单请求吞吐、内存和正确性条件后，转入并发阶段。此时无需为了
236.1 tok/s 的跨硬件公开绝对数字继续单请求优化；仍不宣称取得公开同条件排名。
若未满足，依据实测缺口修复或优化，不自动转入并发，也不将目标标为完成。

## 本轮已冻结的执行方案

冻结输入：`benchmark/dflash2/deepswe_long_requests01/manifest.json`。从既有独立
stratified20 样本中仅依据历史长度及 task ID 哈希选择三条轨迹：
quill-shared-toolbar-focus、effect-sse-httpapi-streaming、
dynamodb-toolbox-lazy-recursive-schemas；不依据本地结果或吞吐选择。
45 个请求在当前 native renderer 下，各长度桶数量为 9、9、9、15、3，
最长 264,494 token。原始日志的较大 token 数不能替代当前实际 token 数。

主配置固定为 K=65,536、prefill=65,536、G=32,768、chunk=2,048、block=32，
GPU Mean-K 索引，query-conditioned，guided-reselect=both，guided-query=4,096，
immutable K，CPU KV 预算 64 GiB。逻辑 ctx 上限本轮为 1,048,576，明确不代表
已验证无限历史或旧服务的 3M 上限。GPU ratio=.44 给索引、草稿及临时内存留余量；
它不是 allocator 硬上限，仍由独立 NVML 监测检查 48,000,000,000 字节。
使用 build17 及已验证的 DFlash query-context reset，七草稿。

重放顺序为 DFlash(seed=1000)、MTP4(seed=1000)、DFlash(seed=1001)，每次新服务，
各执行全部 45 个请求，任务内部按原始请求顺序，保留自然 EOS，max_tokens=4096。
不同 seed 是两次独立采样重复，不冒充相同输出的重复计时。所有短输出、length 终止、
错误均保留；后续冻结前缀不使用本轮生成的答案。因此它不等同于真实 agent 的严格
prompt+response 续接。真实 agent 阶段另验前缀复用和工具状态。

重放吞吐资格要求：两个 DFlash 重复都完成且无资源/计时违规；每个 >=64K 桶
两次合计至少 4,096 个输出 token、每次至少三个有效请求，并报告各重复吞吐。
输出不足只能判为证据不足，不能强制忽略 EOS 补长。长桶 token 加权 decode
均 >=130 tok/s 后，再结合真实 agent 阶段结果决定是否转并发。
同配置 MTP4 用作参照。由于当前 guided self-query 要求 MTP 或 DFlash，
无投机且 guided=off 的结果不能称为相同策略下的 AR 基线；本轮不据此报告加速比。

真实 agent 阶段按以上三个 task ID 的顺序各运行一次，使用同一 KVMem 配置，
官方工具循环和 verifier、原任务 90 分钟 agent deadline。不能根据成败重试模型；
基础设施失败单独保留并诊断。真实任务未产生足够长历史则只报告实际覆盖，
必要时按照已冻结的候补顺序补充，不能删除慢请求或失败任务。

执行记录入口：`benchmark/dflash2/deepswe_long_replay01/manifest.json` 与
`benchmark/dflash2/deepswe_long_queue02/manifest.json`。前者等待当前实验主进程
473828 退出，且 GPU 连续空闲 60 秒后运行；后者在重放完整通过计数/资源检查后
运行已冻结的 `deepswe_long_live_plan01/plan.json`，输出到 `deepswe_long_live01`。
这里接续真实任务只要求重放完成、计数一致、GPU 独占和显存合格，不要求重放先达到
130 tok/s 或足够输出量。真实 agent 本身就是用户要求的测试，不能用重放速度/样本量
代替。旧 queue01 因该条件修正主动终止于 CPU 等待阶段，记录保留；没有重启重放
进程，也没有加载 GPU 模型。
资源或执行失败时保留记录、停止自动推进，等待诊断；并发开发仍需结合
真实任务结果和状态检查后决定，脚本不自动宣告总体通过。

已完成的 CPU 检查包括全部 45 个输入哈希、CLI 参数来源、既有 native 日志解析、
时间分项守恒、UTF-8/tool SSE、截断 SSE 拒绝、加权统计和未启动结果不可通过。
主模型/草稿哈希及 pinned Pier/mini-swe/Python/任务清单均已核对。
CPU 检查不等于 GPU 正确性或吞吐验收；首轮部分 GPU 结果见最新进度文档。

真实任务汇总器 `benchmark/dflash2/summarize_deepswe_long_live.py` 已用三条历史轨迹
共 500 轮请求逐轮核对 native/HTTP/轨迹 token 计数（全部一致），并测试缺失计数、
时间不守恒、HTTP 失败、未完成生成和轨迹缺失不能被当成有效测量。任务结果同时
核对 reward 与 F2P/P2P passed/total，reward=1 单独不能证明测试通过。这些历史
轨迹仅作解析器验证，不提供本轮吞吐、质量或显存结论。

显存汇总器 `benchmark/dflash2/summarize_long_resources.py` 从压缩的原始 NVML 时间线
重新计算进程生命周期、启动阶段和逐请求峰值，核对样本数、真实采样间隙、进程/GPU
身份及其他进程干扰。已通过既有只读 host observer 的 50 个样本和反例检查。
逐请求窗口包含 prefill、检索重放与 decode 的整体；当前 native 日志只有各阶段时长，
因此不能据此虚构每一阶段的精确显存峰值，也不能把 10ms 采样称为 allocator 硬上限。
