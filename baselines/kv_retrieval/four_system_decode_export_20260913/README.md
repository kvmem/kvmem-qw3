# 四系统 decode 吞吐绘图数据

本导出仅整理已有实验；没有重新运行 GPU 测试。吞吐单位 token/s，均为三次重复的中位数。

## 文件与选择规则

- `latest_confirmed.csv`：32K、240K 最新复核。qw3/KVMem 使用前一轮原生 MTP=4 数据；InfLLM 使用最新复核 MTP=0；InfiniGen 使用最新交错线程复核的 optimized_both、8 线程、MTP=0。按方法固定配置，没有逐长度挑选最快 MTP 设置。
- `latest_confirmed_mtp4.csv`：同样的来源，四者均 MTP=4。InfiniGen 32K 此轮三次范围为 9.92–19.27 token/s，不能只展示中位数而忽略波动。
- `full_length_earlier.csv`：8K、32K、120K、240K 的完整较早测量。qw3/KVMem MTP=4；外部两方法 MTP=0、2 线程。InfiniGen 未应用后续 optimized_both 优化，不能标为最新或最佳性能。
- `full_length_earlier_mtp4.csv`：上述完整较早测量，四者均 MTP=4。
- `measurements.csv`：33 条汇总记录，包含各自 MTP、实现版本、重复数、中位数、最小/最大值和来源行号。没有混入插桩、oracle、GPU 全历史数据源或 GIL 调参结果。
- `sources/` 与 `provenance.json`：原始汇总 CSV 的副本与 SHA256。InfiniGen 的最新表采用交错复核三次数据，与此前答复相同；源文件还保留其他控制组。更早的八模式正式矩阵不混入该三次复核。

优化后的 InfiniGen 尚无 8K、120K 对应正式复核。请勿用较早版本这两点与优化后 32K、240K 拼成一条固定实现的曲线，也不要插值作为实测值。

## 共同测量设置

- Qwen3.8-27B，同一 Q8_0 GGUF 权重；FP8 E4M3FN 历史 KV。
- 同一 RTX PRO 6000 Blackwell Server Edition，测量时 GPU 独占，CPU affinity 0–15、32–47。
- 同一条真实 DeepSWE 历史的不同前缀，batch=1，greedy，固定生成 128 tokens，统计后 127 个有用输出。
- 8K=8192、32K=32768、120K=122880、240K=245760 个输入历史 tokens。横轴不是 active window，也不是多个独立任务。
- decode 计时计入草稿、验证、拒绝恢复及生成期间的检索/搬运；模型加载、初始 prefill、校准和首 token 前的历史维护不计入。

## 比较边界

- KVMem：128K 选择预算、128K 生成预算、block=32、2M CPU 历史，无 NVMe，GC 关闭；请求提供显式 query span，保留 query replay。本组未实际生成私有 guided query。其选择/replay 在首 token 前，因此不在 decode 吞吐分母内。
- 普通 qw3：262144 context；本组是固定前缀微基准，没有 Claude Code compaction。
- InfLLM：128 initial + 4096 local + 16×128 retrieved tokens，GPU 历史块缓存容量 32 blocks。使用作者 ContextManager 和缓存机制，Qwen/Q8/FP8/MTP 是本地适配。
- InfiniGen：每 query head 最多 2048 个历史 tokens，alpha=4、partial ratio=0.2；实际选择量可变。优化版保持选择与计算语义，改变线程配置、索引处理与无用 gate 投影计算。
- 本地适配使用 HF/FLA 与独立 Q8 算子路径，原生 qw3 使用不同的算子组织和残差精度。原生 MTP 基于 KVMem 工作集，外部 MTP draft 保留完整 GPU FP8 历史。工作集与总内存预算没有对齐。
- 后续瓶颈实验还为诊断预分配了不参与普通 CPU-source 路径的 GPU shadow；不能从这些记录的显存占用推导部署内存效率。
- 外部优化/MTP 的逐 token 一致性检查只证明各方法内部实现一致；不是四方法输出一致或任务质量相同。KVMem 240K 三次输出也不同。
- 这些数据支持当前实现的稳态 decode 吞吐对比，不代表端到端 agent 加速、guided-query 摊销吞吐或同质量/同显存的纯检索方法加速。

重建：在本目录运行 `python3 export.py`。
