# DFlash 2 / KVMem 宽 Q8 SwiGLU 候选

2026-09-27。用户要求集中优化，跳过不必要的重复效率测试。
本候选已独立编译，尚未运行 GPU 正确性或吞吐检查。

## 改动和依据

`BatchedPrefillExecutor::prefill_ragged` 使用 `Q8VerificationMath` 保持验证的 Q8_1
激活量化和四warp的FP32归约顺序。已有宽 Q8 fanout 可以同时处理多个请求的验证行；
FFN down 已经通过这个接口，但 gate/up 的 `silu_mul` 仍按8行分别调用融合算子。

新路径给 `silu_mul` 提供调用者已有的 gate/up scratch，在超过8行、权重Q8_0、输入和
输出FP32、输出紧密排列、CUDA宽投影可用时，执行一次两输出宽fanout，再执行原有
FP32 `silu_mul_n`。不新建每层临时张量，不增加BF16目标权重，不改变FP8 KV。
潜在收益是减少量化/启动次数并扩大权重复用；代价是写入gate/up中间值以及一次独立
逐元素调用。最终是否更快必须以测量决定。

开关全部显式：

```text
QW3_Q8_VERIFY_WIDE=1
QW3_Q8_VERIFY_WIDE_TILE=32
QW3_Q8_VERIFY_WIDE_SWIGLU=1
```

新增SwiGLU开关默认关闭；缺少scratch、8行及以下、padding或不支持的类型继续走原
实现。常规prefill和单请求验证不进入新分支。算子计数以及
`native continuous_wide_swiglu` 日志用来确认真正执行了新分支，避免只验证回退路径。

## 精简验证

1. 定向GPU算子检查：21个尺寸/行数组合，覆盖8行回退、9/31/33行尾块、32/64行宽块、
   不规则维度、padding回退和真实17408×5120的FFN尺寸。与原融合路径逐位比较；检查
   输出和scratch尾部哨兵，并核对新路径调用次数。已有宽投影CUDA kernel未修改。
2. 短C4：只切换SwiGLU开关，宽线性投影两侧均开启。CPU/GPU采样、取消和不同生成长度
   的32条响应逐回调比较。调度开关与S1a相同，私有query让出关闭，隔离数学改动。
3. 一个冻结的长历史C4请求组：前4条真实DeepSWE输入，seed1000，K128K/G64K、自然EOS、
   上限4096、Q8_0/FP8和原资源策略不变。“仅宽投影”“宽投影+SwiGLU”各执行一次，复用
   S1a完整测试第一组的原始结果。检查token ID，再比较整组完成墙钟、服务吞吐、统一
   callback窗口吞吐、TTFT/最大输出停顿。不同阶段的等待不能从分母中删除。

GPU和模型错误、输出不一致、外部GPU干扰都停止当前自有测试并保存证据。首轮没有
独立重复，只作候选筛选；有明确收益才扩大输入集合、补另一种子和最终复测。

## 构建与证据

- 构建脚本：`benchmark/dflash2/build_wide_swiglu01.py`。
- 冻结源码、差异和独立二进制：`benchmark/dflash2/wide_swiglu_build01/`。
- CPU语法检查、CUDA条件分支编译、CUDA算子探针及模型探针链接：已通过。
- 执行脚本：`benchmark/dflash2/run_wide_swiglu01.py`。
- 后台队列：`qw3-dflash-wide-swiglu01.service`；状态及不可变计划位于
  `benchmark/dflash2/wide_swiglu01/`。

队列依赖S1a成功；私有query候选是独立开关，其特有检查失败不会自动归咎本数学路径。
当前仍无GPU数值、性能或默认启用资格；尚未实施FP8 ragged规划优化和完整会话复用。
