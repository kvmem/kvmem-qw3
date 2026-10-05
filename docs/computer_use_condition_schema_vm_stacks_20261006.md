# 完成条件对照：真实 Linux VM 服务栈资格（2026-10-06）

原转换器侧和完成条件修复侧各新建一台独立 Linux VM，两侧均通过原来的十三项只读服务栈检查。真实客户端、网关观察、截图和连续录屏可用；独立审计确认两侧 SQLite 动作数均为零，没有模型请求，服务进程与 Codex 后代已退出、VM 已不存在，各自六个端口不可达。两份原连续录像均完整解码七帧。这是服务栈资格，不是两次 QW3 任务成功。

新准备的独立对照保留原任务材料、初始化、提示、评分器和预算。两侧都使用相同的原生按钮执行器，只改变传给模型的内嵌完成条件定义；原有原生键盘／按钮八次研究及运行队列不改。

| 条件 | 原转换器侧 | 条件修复侧 |
| --- | --- | --- |
| 文本任务 | 360 秒、100 次调用，重复两次 | 相同 |
| Calc 任务 | 900 秒、100 次调用，重复两次 | 相同 |
| 模型配置 | thinking、DFlash2、KVMem OFF、16K 输出、524288 上下文、保留全部历史 | 相同 |
| 工具路线 | 纯 GUI，observe/act/workflow，无软件工具或云顾问 | 相同 |
| 执行器 | 原生按钮候选 | 相同 |
| 处理顺序 | 第一次先原转换器，第二次先修复侧 | 反转顺序，合计八次 |

五项研究资格检查通过：完整双臂清单和原材料、预算、顺序、来源与前序依赖不可被改动；临时研究映射在异常退出后仍恢复原 lookup 和研究哈希。实际服务栈复用原冻结实现，只在本轮独立进程里映射新研究；未修改原文件、控制器或运行中进程。

正式模型试验尚未派发，新的串行控制器也未登记。两侧资格没有绑定真实引擎，客户端上下文为未绑定时的默认值，不能将这次只读检查称为已验证的 524288 模型栈。正式入口仍需绑定原 OFF 引擎，核验实际上下文，并在原八次研究全部结束、原进程退出且产物/回执/录像齐全后交接。

首次独立审计因为绝对归档路径与相对命令行目录比较而失败，源码和原失败保留；修正审计路径后只重读同一批原始产物，没有重启 VM 或重复测试。最终独立核验两侧全部通过。源环境启动时报告缺少可选代理配置，代理已禁用，没有因此下载或修改代理文件。

最新实际原六个控制器仍是同一代次。图像 1/12，其他原组 0/4、0/8、0/8、0/4，原生组 0/8；可用显存 55711 MiB，低于原 56000 MiB 门槛。没有干预用户 GPU 工作。目标 active；复杂 OSWorld 和 KVMem 尚无业务改善的结论保持。

来源见 [完整报告](computer_use_condition_schema_vm_stacks_20261006.json)与 `.runtime/condition-schema-model-stack-independent-audit-02/result.json`。研究准备清单位于 `.runtime/condition-schema-model-study-20261006-01/protocol.json`；两侧原始记录位于 `.runtime/condition-schema-model-stack-control-01` 和 `.runtime/condition-schema-model-stack-repair-01`。

下一步完成原 pilot 的新试验入口与串行交接资格，再按冻结清单执行八次真实 QW3 对照，收集业务正确率、工具拒绝与步骤错误、模型请求成本及连续录像。组件通过和安全暂停都不算任务完成。

只读服务栈可在仓库根目录用新的输出目录与空闲的六连端口复现。固定镜像必须已安装；此入口不下载镜像或启动模型。两侧使用各自的私有包，避免选错转换器：

```bash
PYTHONPATH=apps/computer_use/.runtime/native-control-candidate-20261005-02/framework:apps/computer_use:apps/computer_use/.runtime/media-qualification \
  third_party/OSWorld-V2/.venv/bin/python -m qualify_condition_schema_model_stack \
  --study apps/computer_use/.runtime/condition-schema-model-study-20261006-01/protocol.json \
  --trial-id condition-schema-linux_text-wire-control-r1 \
  --output apps/computer_use/.runtime/condition-schema-model-stack-control-new --port-base 19140
PYTHONPATH=apps/computer_use/.runtime/condition-schema-candidate-20261006-02/framework:apps/computer_use:apps/computer_use/.runtime/media-qualification \
  third_party/OSWorld-V2/.venv/bin/python -m qualify_condition_schema_model_stack \
  --study apps/computer_use/.runtime/condition-schema-model-study-20261006-01/protocol.json \
  --trial-id condition-schema-linux_text-condition-repair-r1 \
  --output apps/computer_use/.runtime/condition-schema-model-stack-repair-new --port-base 19150
```
