# Q8/FP8 公开实现对照资格检查

本轮没有修改原生运行时代码，也没有开启多并发。原生仍以 Q8_0 主模型、E4M3 FP8 KV、
单请求 KVMem 为主线。结论是：当前 vLLM + GGUF 插件配置可以生成文本，
但**尚不具备正式性能对照资格**，不能用其吞吐宣称接近或超过公开结果。

## 实际配置

- vLLM 0.30.0，V2 runner，GGUF 插件 0.0.5，真实 Q8 GGUF 目标权重与 FP8 E4M3 目标 KV。
- 官方 BF16 DFlash2 草稿，7 个草稿 token；参考草稿 KV 为 BF16，原生草稿 attention/cache 为 FP32。
- tokenizer/HF config 来自 BF16 模型目录，只读取配置与 tokenizer，不加载 BF16 目标权重。
- 单请求，max-model-len=8192，max-num-batched-tokens=1024，关闭 prefix cache，eager 执行。
- 保留阶段 5 中明确记录的 draft loader import overlay；未改动采样或执行内核。
  因此它是带加载适配的参考端，不能称为未经修改的公开发行版。

## Greedy 检查失败

使用冻结五任务各前两条输入，每条最多生成 256 token，最后重复第一条 GSM8K 输入。
无投机与 DFlash 各完成 11 次请求，输入 token ID 与冻结数据全部一致；
各自第一条与最后重复请求的 choices/usage 一致。

跨路线只有 1/11 次完整 choices/usage 相同。去重后是 1/10 条输入相同、9 条不同。
首个差异出现于第 25–233 个输出 token（从 0 开始计数）；重复的 GSM8K 也在同一位置分叉。
没有把这些差异改判为通过，也没有在失败后放宽 greedy 验收标准。

GGUF 插件源码根据输入行数选择 MMVQ 或 MMQ，因此单步与八行验证可能走不同算子。
这是后续定位线索，尚未通过固定前缀 logits 或算子消融证明为上述分叉的根因。
不能由此断言 DFlash 算法本身错误，也不能把参考端分叉归因于原生 KVMem。

证据：`benchmark/dflash2/vllm_q8_greedy_{none01,dflash01,comparison01}.json`，
对应服务日志，以及 `run_vllm_q8_greedy.py`、`compare_vllm_q8_greedy.py`。

## 采样分布检查也未通过

参考 V2 的显式 seed 普通采样与 DFlash 验证均调用 Triton top-k/top-p 过滤，
顺序都是先 top-k、再 top-p。但相同顺序不足以证明边界处理一致。

独立 GPU 检查直接调用已安装 vLLM 的 `apply_top_k_top_p`，与编译自原生
`src/target_sampling.hpp` 的 CPU 分布比较；使用相同 FP32 logits、温度、k、p。
共 24 个用例，14 个满足候选集合完全一致且概率最大绝对误差小于 2e-6，10 个不同。
9 个未舍入随机 FP32 用例全部通过；差异包括 5 个 BF16 舍入用例和 5 个人工边界用例。

例如 logits=[0,0,-100,-100]、k=4、p=0.5 时，原生保留两个高概率 token，
参考端只保留一个。248320 词表的 BF16 舍入用例在 T=1、k=20、p=.95 时，
原生保留 20 个，参考端保留 18 个。结果保留了全部输入、稀疏分布和源码哈希。

该检查针对过滤后的目标分布，不是完整拒绝采样器测试；人工数据不能估计真实请求中
发生边界差异的频率。这与 T=0 greedy 分叉是两个独立问题，不能相互作为根因证明。

证据：`benchmark/dflash2/vllm_sampling_audit_policy01/{report,inputs}.json`、
`vllm_sampling_audit_policy01.log`、`audit_vllm_sampling.py`、`native_distribution_audit.cpp`。

## 后续优先级

继续原生单请求验证与优化，不因参考端失败而停止可独立完成的工作。
阶段 5 的默认关闭候选已通过三组独立进程对工程计时与五任务小样本输出一致性检查；
当前进一步验证完整 GSM128。完整五任务质量、独立重复、剩余 KVMem 生命周期及矩阵仍需完成。

正式同机公开对照还需解决以上资格问题、容量策略及优化执行配置，并作协议修订。
`public_execution_protocol_v8.json` 没有被悄悄改成 vLLM 协议。H200 公开数字仍只作
跨硬件参照，不把 eager 参考日志中的吞吐作为最终对照。多并发继续后置。
