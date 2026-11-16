# KVMem 进程内多会话历史

`--kvmem-session-capacity` 将可保留的会话历史数量与 `--max-active` 分开。
例如 active=2、capacity=6 时，服务最多同时执行两个请求，并保留六个具名历史。
需要给新请求让出共享 KV 页时，服务将闲置会话的目标/MTP KV 页按原始内容保存到
CPU，释放其共享页；下次请求恢复这些页，再按已有前缀执行后缀。

这是**默认关闭的进程内功能**。默认 `capacity=0` 保留原有 idle cache 路径及其淘汰行为。
它不是重启持久化、独立的磁盘会话格式或多进程共享会话服务：退出进程后，调用方需要
凭自己保存的完整消息重新建立历史。`--kvmem-nvme-dir` 的存在不改变这一点。

## 启用条件

| 设置 | 含义 |
| --- | --- |
| `--max-active A` | 同时执行的请求槽数 |
| `--kvmem-session-capacity N` | 保留历史的硬上限，包含活跃、warm、suspended 和尚未删除的 invalid 历史；N 必须不小于 A，默认 0 |
| `--kvmem-session-swap-gb F` | 精确保留 suspended 目标/MTP KV 页的额外 CPU 预算，单位 GiB；启用会话管理时必须大于 0 |
| `--kvmem-cpu-gb` / `--kvmem-nvme-gb` | 原有 KVMem 历史 tier 的总额度；与 session swap 分开计量 |

当前实现要求 `serve`、bounded repository、continuous batching、完整 cooperative
controller、cooperative prefill 和 keyed prefix reuse。需要下列环境开关：

```bash
export QW3_EXPERIMENTAL_BOUNDED_CONTINUOUS=1
export QW3_CONTINUOUS_KVMEM_CONTROLLER=1
export QW3_CONTINUOUS_KVMEM_COOPERATIVE_PREFILL=1
export QW3_CONTINUOUS_KVMEM_SESSION_REUSE=1
```

下面是小规模功能检查配置；模型路径和资源额度应按实际机器调整。
它不代表长上下文或高并发性能已得到验证。

```bash
./build/qw3 serve \
  --model models/Qwen3.5-4B-Q8_0-with-MTP.gguf \
  --host 127.0.0.1 --port 8080 \
  --speculative-method none \
  --continuous-batching --max-active 2 \
  --paged-kv --kv-page-size 16 --kv-dtype fp8 \
  --ctx 8192 --prefill-chunk 128 \
  --kvmem --kvmem-query-conditioned --kvmem-prefix-cache \
  --kvmem-update-mode step --kvmem-block-tokens 32 \
  --kvmem-repository-blocks 256 --kvmem-repository-gc delete \
  --kvmem-budget 1024 --kvmem-prefill-budget 1024 --kvmem-gen-budget 2048 \
  --kvmem-sink-blocks 1 --kvmem-recent-blocks 4 \
  --kvmem-index-placement cpu --kvmem-cpu-gb 7 \
  --kvmem-session-capacity 6 --kvmem-session-swap-gb 2
```

该模式沿用 bounded controller 的兼容性限制，不能与 generic `--prefix-cache`、
layered MTP verifier 或 `q8` KV scale planes 混用。FP8 与 Q8 模型权重是不同设置：
这里拒绝的是 `--kv-dtype q8`。旧 `kvmem_session_id/start/append/finish` 和
`kvmem_cache` API 不接入这一多会话路径。不支持的请求明确报错，不退回共享主 executor。

## 请求和所有权

Chat Completions 和 Completions 请求继续使用 `kvmem_prefix_cache_key`，并提交
**完整历史**。这不是只发送增量文本的协议。相同 key 对应一个可变历史；不同 key 可以
同时进入 continuous batching。每个 key 同时只能有一个排队或执行中的操作。

```json
{
  "model": "test",
  "kvmem_prefix_cache_key": "app-user-42:conversation-7",
  "messages": [
    {"role": "user", "content": "Remember that this project's code is amber."}
  ],
  "max_tokens": 32,
  "temperature": 0,
  "enable_thinking": false
}
```

下一轮保留相同 key，并将上轮完整输入、模型回复及新的用户消息一起发送。
是否可复用、复用到哪个边界仍由原有 token/prefix 检查决定。只恢复底层页不代表
可以跳过前缀身份检查，也不保证任意修改历史都能复用。

Key 为 1–128 个字符，可使用字母、数字、`.`、`_`、`-`、`:`。
应由可信应用根据用户与会话生成和管理 key。此接口不提供身份认证、tenant ACL 或
每用户配额；前端不能仅凭一个客户端任意提交的 key 判断用户归属。查询、删除和
取消接口也应通过应用自己的身份验证层。

## 查询、删除与取消

```text
GET    /v1/kvmem/sessions/{key}
DELETE /v1/kvmem/sessions/{key}
```

GET 返回该进程拥有的会话信息。例如：

```json
{
  "object": "kvmem.session",
  "id": "app-user-42:conversation-7",
  "state": "suspended",
  "history_tokens": 1024,
  "swap_bytes": 16777216,
  "retained_device_bytes": 33554432
}
```

上面的数字仅解释响应结构，不是容量估算。

| 状态或结果 | 语义 |
| --- | --- |
| `warm` | 保留历史及共享 KV 页的闲置会话 |
| `suspended` | 精确 KV 页在 CPU swap 中，已归还共享 GPU KV 页；恢复保留原历史 |
| `invalid` | 执行中的取消或异常使该历史不能安全续接；仍占用历史名额，必须显式删除 |
| HTTP 409 | 同 key 已有排队/执行中的操作，或尝试续接 invalid 历史 |
| HTTP 429 | 历史容量、swap 预算或其他准入额度不足 |
| HTTP 404 | 该进程不存在这个 key，或此功能未启用 |

GET/DELETE 同样遵守 key 的排他规则：正在执行或排队时返回 409，不读取半提交状态，
也不销毁正在使用的 executor。成功 DELETE 返回 `{"id":"...","status":"released"}`，
归还该历史及其资源，随后可以使用相同 key 从完整输入重新建立会话。

请求可携带 `x-qw3-request-id`，再通过原有
`POST /v1/requests/{request-id}/cancel` 取消。若执行中的请求取消、断连或失败，
历史会保留为 invalid，后续请求返回 409，直到显式 DELETE。服务不会悄悄丢掉原历史
并把下一轮当成冷请求。尚未获得执行权的排队取消与已开始修改历史的取消应分别检查。
正常命中请求的 `stop` 序列属于成功结束：在新会话模式下，Chat/Completions 流式接口
保留有效历史，允许后续轮次继续；断线或显式取消仍优先判为 invalid。

非流式请求通过 HTTP 状态返回错误；流式响应若已发送响应头，后续错误通过流中的
错误事件返回。调用方必须同时处理 HTTP 状态和 SSE 错误。

## 内存边界

压力挂起只释放**共享的目标/MTP GPU KV 页**。executor 私有的 recurrent/conv、
scratch、checkpoint、部分投机状态等仍可能驻留 GPU。因此，增加 capacity 仍会增加
内存占用；history 数量不能无限扩大，也不能把 suspended 理解为该用户完全不占显存。
GET 的 `retained_device_bytes` 用于观察该会话保留的已统计 device allocations，
包含 prompt/end/replay checkpoint 的保守计数，不能代替整卡峰值或进程显存测量。

CPU/NVMe 历史 tier 的额度按 `capacity + 1` 个 owner 分配：最多 capacity 个历史 owner，
外加 setup executor。增加 capacity 会降低每个 owner 的份额，需要重新核对 repository
容量。Session swap 是另一个显式 CPU 总预算；它不包含全部历史 tier、索引、模型映射、
I/O staging 和其他 host scratch。

历史容量或 swap 空间不足时返回 429，保留已有历史。挂起失败不会以驱逐其他用户历史
来掩盖容量不足。调用方可以重试、调节工作负载或显式 DELETE 已结束的会话。会话管理层
不做静默 LRU 历史淘汰；bounded repository 内部已有的块级 GC 策略仍独立运行。

若恢复发生设备故障，且无法确认相关 DMA 已完成，涉及的页会被隔离，不交给其他会话
复用；该 engine 随后拒绝生成和 DELETE（500），GET 仍可用于检查状态。此类设备故障
需要重启 engine，不属于可通过普通重试或 DELETE 恢复的容量拒绝。

## 验证和观测

设置 `QW3_CONTINUOUS_BATCHING_TRACE=1` 和 `QW3_KVMEM_PREFIX_CACHE_TRACE=1` 后，
检查 `native kvmem_session_memory: event=park|resume|release|create`、对应 key、
`swap_bytes`，以及后续 `prefix-cache HIT` 的正数 reuse 边界。单有 HTTP 成功或 resume
日志不能证明没有重算完整历史，也不能证明并发吞吐提高。

无需模型或 GPU 的 CLI 回归：

```bash
python3 tests/kvmem_session_cli_test.py /path/to/new/qw3
```

GPU 空闲时运行独立生命周期探针；它只启动并关闭自身的 localhost 服务，要求新的输出目录：

```bash
python3 tests/kvmem_multi_session_http_probe.py \
  --binary /path/to/new/qw3 \
  --model models/Qwen3.5-4B-Q8_0-with-MTP.gguf \
  --output /tmp/qw3-session-memory-probe
```

默认 active=2、history=4、target-only controller。探针比较关闭新功能时的热会话参考与
开启后的轮换恢复，核对完整回复和 usage，检查 GET 状态、实际 park/resume 与前缀复用，
再覆盖正常流式 stop 后续接、同 key 冲突、取消、invalid、显式删除、容量拒绝及容量回收。
可使用 `--route mtp` 或 `--route dflash2 --dflash /path/to/draft` 复验投机路径。
`--dry-run` 只打印启动配置，不绑定端口、不启动服务、不访问 GPU。

CPU/CLI 验证不能替代真实 GPU 页恢复和多请求正确性测试；小规模生命周期测试也不能
替代长历史、GC 压力、批宽数值一致性与端到端吞吐测试。
