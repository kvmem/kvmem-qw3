# 完成条件修复：真实 Codex 传输资格（2026-10-06）

新的私有候选已通过 Windows、Linux、Android 三种工具目录，各普通/验证模式共六次真实 Codex 传输检查。它在本地模型请求发出前恢复动作工具的内嵌完成条件；普通模式及工作流工具保持原样。这轮后端只返回固定文本，没有 QW3 推理、桌面执行器或业务成功率结果。

| 工具目录 | 实际客户端发出的验证模式条件 | 候选发给后端的条件 |
| --- | --- | --- |
| Windows | 内嵌 `items` 为空 | 原可信六种条件，原必需字段与 1–12 项限制 |
| Linux | 内嵌 `items` 为空 | 同上 |
| Android | 字段保留，`const` 被转成单元素枚举，描述与 `maxItems` 丢失 | 精确认出这一已知改写后恢复原条件 |

Android 首次验证被严格保护拒绝，原失败保留。修复只接受空定义、原定义或**与可信定义的已知 Codex 改写完全相同**的结构；未知非空定义、不同上下限和改变过的可信声明仍被拒绝。没有使用宽松匹配绕过保护，也不修改运行期完成判据。

候选由原生按钮候选的完整、固定源码清单构建，只修改 `protocol.py` 并增加不依赖 jsonschema 的私有 `condition_schema_wire.py`。当前运行中的原生八次对照、其他五组对照、协议、评分及所有原失败均未修改。CPython 3.12 的实际私有导入与恢复检查通过，无需安装新依赖。

最终十项 CPU 检查通过，包括完整源码范围、三平台转换、实际保留的 Android 请求、未知定义拒绝和字段语义。六次真实 Codex 0.159.2 检查均只有一次固定文本后端请求，工具调用为零。独立审计核对 125 个来源，使用原转换器重建预期请求，证明转发请求只改变内嵌条件和沿用原有工具过滤；消息、模型参数、权限与其他参数定义保持原行为。实际 Codex、MCP 后代已退出，本轮监听不可达。六次固定文本回复不能算六次模型任务。

两次早期失败保留：检查 01 把未过滤转换结果与已过滤后端结果比较，忽略原桥接排除的四个宿主工具；它只有一次固定回复，且当时尚未在比较失败前保存原 Responses 请求。检查 02 前五种模式通过，Android 验证模式在后端请求前被拒绝。两次原登记的十四个进程代次均已独立确认退出；旧九项 CPU 通过记录不是最终源码的资格，不累加为覆盖。

最新核验六个原控制器仍是同一代次。图像保留 1/12，其余原组 0/4、0/8、0/8、0/4，原生组 0/8；当前可用 55711 MiB，低于原 56000 MiB 门槛，没有干预用户 GPU 工作。目标仍 active。

下一步先对候选做实际私有 VM/客户端栈资格，再准备独立的同条件原转换/修复对照；原八次研究保持不变。只有取得实际 QW3 的产物正确率、步骤错误、时延和连续录像，才能决定是否设为默认。现有复杂 OSWorld 和 KVMem 尚无业务改善的结论保持不变。

来源见 [完整资格报告](computer_use_condition_schema_transport_20261006.json)。主要原始记录：

- `.runtime/condition-schema-candidate-20261006-02/build.json`
- `.runtime/condition-schema-wire-cpu-02/result.json`
- `.runtime/condition-schema-transport-20261006-03/result.json`
- `.runtime/condition-schema-transport-independent-audit-01/result.json`
- `.runtime/condition-schema-private-import-01/result.json`

在仓库根目录，用新的输出目录复现构建和传输检查；传输检查只启动本机固定文本服务及隔离 Codex，不启动模型或设备：

```bash
PYTHONPATH=apps/computer_use:python .venv/bin/python -m condition_schema_candidate \
  --base-build apps/computer_use/.runtime/native-control-candidate-20261005-02/build.json \
  --output apps/computer_use/.runtime/condition-schema-candidate-new
PYTHONPATH=apps/computer_use:python .venv/bin/python -m qualify_condition_schema_transport \
  --build apps/computer_use/.runtime/condition-schema-candidate-new/build.json \
  --output apps/computer_use/.runtime/condition-schema-transport-new
PYTHONPATH=apps/computer_use:python .venv/bin/python -m audit_condition_schema_transport \
  --report apps/computer_use/.runtime/condition-schema-transport-new/result.json \
  --output apps/computer_use/.runtime/condition-schema-transport-audit-new
```
