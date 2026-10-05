# 原生控件同条件模型复测

截至本轮独立核验，串行等待进程已经实际登记，八次 QW3 任务尚未派发。调度与结果收集资格通过，不等于业务成功率改善。完整记录见 [JSON 报告](computer_use_native_model_queue_20261006.json)。

## 固定的比较

| 任务 | 控制组 | 候选组 | 重复 |
| --- | --- | --- | --- |
| Linux 文本编辑与保存 | 已核验原生键盘 | 同一键盘能力，加原生按钮核验与激活 | 各两次 |
| Linux Calc 表格编辑与保存 | 已核验原生键盘 | 同一键盘能力，加原生按钮核验与激活 | 各两次 |

任务复用原 `platform_pilots` 的提示、初始化和独立评分。文本期限360秒，Calc期限900秒；工具上限100、单请求输出16K、thinking开启、上下文524288、DFlash2、KVMem OFF、全部视觉历史、单模型并发。第二轮反转两组顺序。两组使用相同的中性任务目录名，避免把候选名称带入材料。仅开放观察、动作和工作流工具；不调用顾问或软件作业工具。

原研究位于 `/home/chaidi/qw3/apps/computer_use/.runtime/native-control-model-retest-20261005-01/protocol.json`。本轮声明另存，绑定当前实际查询控制器，不改原研究或已运行框架。

## 执行与交接

等待进程完整身份为PID354765、UID1001、start_ticks377284282；注册文件还保存实际命令和工作目录。PID不能单独作为身份。实际声明是 `/home/chaidi/qw3/apps/computer_use/.runtime/native-control-model-queue-20261006-01/declaration.json`。

交接先确认查询控制器仍是登记实例；只有其四次回放完整终止、原生产者实际退出，且更早库存、原结果和最终OFF引擎核验通过后，才运行八项。查询未证明检索收益也可以正常交接，但不能存在未确认请求、输入或结果。调度复用原连续显存准入和引擎监督逻辑：至少56000MiB持续30秒，再经过原启动与就绪检查，不改变任务预算。

每项启动私有Linux VM、网关和真实Codex客户端，由原任务执行器管理期限。子任务观察超时只继续核对同一子进程，不重启或重复派发。动作送达或任务退役未确认时停止后续任务并保留现场。

## 结果收集

每项保留原 `integration-result.json`，包括失败成绩；再检查实际任务配置、模型请求、提示一致性、自有服务和录屏进程退出、同一VM消失。连续录像须与原录屏回执哈希一致，并被独立解码至结束，帧数和尺寸也须一致。业务结果或后续证据不完整都不能补写为通过。

最终比较完整通过率、产物正确率、任务时间、请求时间、工具调用数、输入拒绝及恢复原因，并逐项回放。组件检查数不会计入任务成功率。此前实际服务栈录像的两份7帧解码，只验证解码入口，并非新QW3任务录像。

## 当前机器上的验证入口

下面的只读核验读取现有进程身份和队列，不会重新派发任务：

```bash
cd /home/chaidi/qw3
PYTHONPATH=apps/computer_use:python .venv/bin/python - <<'PY'
from pathlib import Path
import native_control_model_queue as q
path = q.RUNTIME / 'native-control-model-queue-20261006-01/declaration.json'
registration, state = q.registered_waiter(path)
actual = q.base.require_identity(registration['identity'])
print({'pid': actual['pid'], 'status': state['status'],
       'completed': len(state['trials']), 'planned': state['planned_trials']})
PY
```

新增调度源码和测试已冻结于CPU资格档案中。需要重跑CPU资格时，选择新的输出目录：

```bash
PYTHONPATH=apps/computer_use:python .venv/bin/python -m qualify_native_control_model_queue \
  --output apps/computer_use/.runtime/native-control-model-queue-cpu-NEW
```

34项最终检查涵盖实际调度、原成绩收集、未知动作、源码与任务边界、同句柄观察超时，以及完整前序验证逻辑。CPU中的模型、宿主及历史40项终止探针是明确夹具，实际运行仍须通过真实交接。第一轮夹具错误及26项旧版本通过记录均保留，检查数不累加。

当前显存仍未满足原冻结实验的准入要求。原五个控制器保持同一实例，新增八次队列等待前序；复杂OSWorld和KVMem收益结论保持，完整目标尚未完成。
