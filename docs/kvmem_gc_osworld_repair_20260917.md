# 512K OSWorld GC 实验修复

旧运行：`benchmark/osworld_v2/gc512k_all45_20260917`。
修复验证：`benchmark/kvmem_gc_repair_20260917`。
恢复运行：`benchmark/osworld_v2/gc512k_all45_repaired_20260917`。

## 问题与修复

1. CPU Mean-K 索引的 tiled softmax 评分工作区默认为 512 MiB。
   27B 的 24 个 attention heads、16,384 个仓库槽位使每个 Q token
   需要 3 MiB block statistics；216-token query 已需 648 MiB。
   GC 的两遍旧实现不能屏蔽分母中的删除空洞，因此原代码拒绝该路径，
   上层却退回 window-profile，改变了实验方法。

   现在按工作区容量对 Q 行分批。每行仍对完整候选块域计算 softmax，
   原有 deleted/forced-slot masks 保留，各行分数依原定义累加。没有截断
   query，也没有减少 replay 或改变选择预算。分批可能多次传输 CPU 索引，
   不声称其零开销；工作区维持原上限。成功的 bounded scorer 也记录实际
   requested/used 策略；已有完整 Q 时评分失败直接报错，禁止静默退化。

2. MTP 一次可提交 1–5 tokens；只在 position%32==0 时捕获重选后的 P，
   可能连续跨过所有边界。030 的最新 M=1071476，比改写后 LCP=1071408
   晚 68 tokens；旧 P=1071232 的 selection generation 已失效，造成冷 prefill。

   现在仅当重选后 checkpoint 缺失或周期 checkpoint 到期时，限制下一批
   draft 不越过边界；离边界一个 token 时走现有单 token target 路径。
   原捕获、MTP/source-index 完整性检查及 LCP/generation 验证仍执行。
   正常批次不受影响；这不能保证任意早期历史改写都可热恢复。

3. 原分析脚本假设所有调用都有 streaming usage。030 的非流式重试回复
   实际完成，但 usage 没进入 API 事件字段，导致分析退出。新分析器从
   归档 JSON 回复恢复真实 usage；超时且无回复的调用保持未知，不补造
   TTFT，也不将传输故障当作模型失败。

## 验证与恢复原则

- 主机回归 25/25 通过。
- CUDA 数值测试覆盖 FP32/scalar 与 FP16/GEMM：521 行 Q、两层、不等长
  最后一批、额外 layer stride、散布删除空洞及全空 tile；分批/整批评分
  误差 <1e-4，删除槽位得分为零，总概率质量保留全部 Q 行。
- 27B Q8、FP8 KV、MTP=4：32K 仓库、8K 选择预算的真实模型回归，32 轮
  生成中 guided refresh，随后截掉并重写最后 68 个生成 tokens；31 HIT、
  0 MISS、32 个重选后 P、104 个实际回收块，累计历史 33,020 tokens。
  首次小预算探针因测试配置的 mandatory pins 超过选择预算而退出，其日志
  保留为 `27b_middecode.log`，不计入通过结果。
- 512K 仓库、128K 选择/生成预算的 27B 长 query 验证通过：输入历史
  270,620 / 271,222 tokens，526 行 Q 全部参与 Mean-K，实际 Q tile 为170行，
  block statistics为510 MiB，第二次请求 HIT、0 MISS、0策略回退。
  这是实际27B推理/评分测试，不包含视觉编码器。完整验证及程序哈希见
  `benchmark/kvmem_gc_repair_20260917/validation.json`。

旧 22 题中 10 题存在 scorer fallback（含030），其尝试保留为诊断；
12 题无这些故障，保留其原始结果与原二进制身份。新修订先重跑030，
再跑其余9个受影响样本，最后继续23个尚未开始的任务。按故障记录决定
重跑范围，不根据得分重试。合并报告须保留修复前后版本来源，不能称为
45题全部使用同一程序。两批均与同一45题历史2M结果配对。

新启动器每10秒检查新增日志；发现 scorer fallback、评分硬失败或中途
prefix MISS 时停止本次尝试和后续队列，保留故障记录，避免长时间客户端
重试。该检查属于实验进程，不恢复已取消的全局30分钟定时检测。

2026-09-17 18:31 +08 已启动修订队列，首题030。程序SHA256：
`e2324eea6ca0ba74f07d9d4e4e93c2b02c275c2cad0a53a4f852ef61dc0d51cc`。
冻结参数与历史运行一致（512K物理容量、128K选择/128K生成、block32、
CPU历史KV/索引、无NVMe、FP8 KV、MTP4、完整guided query/replay）。
