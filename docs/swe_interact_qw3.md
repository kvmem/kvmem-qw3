# SWE-Interact 在 QW3 + Claude Code 上的测试方案

调研日期：2026-09-24。本文是源码核对后的接入方案；尚未安装 Harbor、启动测试容器、调用模拟用户 API 或跑出 benchmark 成绩。

## 1. 确认 benchmark 与评测方式

这里指 Scale AI 的 [SWE-Interact](https://github.com/scaleapi/SWE-Interact)，对应[论文](https://arxiv.org/html/2606.30573v1)。不要混用搜索结果中另一个 `sweinteract/sweinteract`、125 条数据的 Hugging Face 数据集。

本次核对版本：

- SWE-Interact：`b32f98c3b8f76ca65e84341d1f30e5af7135f85d`
- Harbor：`cdb76bae6dc88d5bca1c8f0754bbba300d6574b4`，源码版本 0.23.0

官方仓库同时提供 `data/multiturn` 和 `data/singleturn`，各 75 个任务：DeepSWE、SWE-bench Pro、SWE Atlas Refactoring 各 25 个。

多轮任务的完整需求和 persona 由独立 `user-server` 容器持有。Claude Code 使用 MCP `ask_user` 提问、提交计划和请求审阅；模拟用户通过主容器的 `repo_exec_server.py` 查看仓库和 diff，逐步补充要求。最终执行任务自带 verifier，不能以“用户满意”或 Claude 进程成功退出代替评分。

当前公开代码中，全部 75 个多轮任务都有以下五个阶段：

| 阶段 | Agent 行为 |
| --- | --- |
| `01_plan` | 用 `ask_user` 沟通需求和计划，获认可后保存 `/tmp/plan.md` |
| `02_implement` | 读取计划，实现、按反馈修改，每轮实现先 commit 再请求审阅 |
| `03_handoff` | 导出实现的 commit history、diff 和工作区状态 |
| `04_write_tests` | 为自己的实现补测试，保存为独立 commit |
| `05_test_handoff` | 最终交接，运行实际评分 |

`multi_step_reward_strategy = "final"`：中间阶段的 reward=1 是流程占位，不能计为解题成功。以所检查的 Bandit 任务为例，最终 verifier 要求工作区干净、实现和测试分开提交，然后撤下 agent 的最后一个测试 commit，对实现运行 canonical hidden tests。DeepSWE 示例要求 baseline/new tests 都通过；RF 示例还要求测试通过和 rubric 的 must-have 条件满足。应以各任务的原始 verifier 为准。

**会话连续性需要显式记录。** 当前官方脚本没有传 `--resume-trajectory`；上述 Harbor 版本默认 `resume_trajectory=False`，各阶段启动新 agent 会话，但共享工作区与模拟用户状态。每个阶段内部仍可多次 `ask_user`。如增加 `--resume-trajectory` 来测试跨阶段连续上下文，应单独命名实验，不能无说明地与默认流程混报。

任务根目录的 `instruction.md` 含原始完整需求；不能拿它直接填入现有单轮 runner，然后把结果叫 SWE-Interact 多轮成绩。应由 Harbor 使用 `steps/*/instruction.md`。

## 2. 当前机器已具备什么

只读检查发现：

- Claude Code 2.1.148、Docker 29.5.0、uv 已安装，Docker daemon 可访问。
- uv 已有 Python 3.12；系统 `python3` 是 3.10，Harbor 要求 Python >=3.12。
- `build/qw3` 和 `models/Qwen3.8-27B-Q8_0.gguf` 存在。
- `qw3 serve` 已支持 Claude Code 使用的 Anthropic Messages、SSE、tool_use/tool_result 和 count_tokens；见 [claude_code.md](claude_code.md)。
- PATH 中没有 Harbor/Modal；本地 Docker 方案不需要 Modal。
- 检查时已有 QW3 服务在 `127.0.0.1:18978` 运行，ctx=524288、KVMem budget=32768。没有修改或重启它，也没有向它发推理请求。
- 检查时磁盘可用约 123 GiB，不宜同时预拉全部 75 个任务镜像。

缺少的是 Harbor 接入和可用的外部模拟用户/评分 API 配置。此次没有读取私人凭据，因此没有判断现有凭据是否可用。

建议结构：

```text
Harbor（宿主机）
  ├─ main 容器：Claude Code 2.1.148 + 任务仓库
  │    ├─ Anthropic Messages → 宿主机 QW3 / Qwen3.8-27B
  │    └─ MCP ask_user → user-server 容器
  ├─ user-server：完整需求 + persona + 外部模拟用户模型
  │    └─ 仓库只读检查 → main:8765
  └─ verifier：原任务测试；RF 任务额外使用 rubric 模型
```

只替换 coding agent 的模型。模拟用户与 RF grader 保持独立；将它们也切换为同一个 QW3 模型会同时改变评测条件，而且交错请求会干扰当前单轨迹 KVMem 状态。

## 3. 安装和固定版本

以下是后续执行命令，不是本次已经执行的操作。先等现有 GPU 实验结束，再启动新的 QW3 服务。

```bash
mkdir -p /home/chaidi/qw3/benchmark/swe_interact
cd /home/chaidi/qw3/benchmark/swe_interact
git clone https://github.com/scaleapi/SWE-Interact.git upstream
git -C upstream checkout b32f98c3b8f76ca65e84341d1f30e5af7135f85d
git clone https://github.com/laude-institute/harbor.git harbor-src
git -C harbor-src checkout cdb76bae6dc88d5bca1c8f0754bbba300d6574b4
uv tool install --python 3.12 ./harbor-src
harbor run --help
harbor agent schema claude-code
```

主机上的 Claude 安装不会自动成为容器内版本，后面的 `--ak version=2.1.148` 用来固定容器安装。

## 4. 配置 API 和容器网络

在 `upstream/harbor/.env` 写入以下配置，替换占位值：

```dotenv
# coding agent：本机 QW3，使用 Anthropic 接口，不加 /v1
ANTHROPIC_API_KEY=local-qw3
ANTHROPIC_BASE_URL=http://host.docker.internal:18080

# 模拟用户：OpenAI-compatible API；这里的地址不带 /v1
SIM_USER_MODEL=openai/gpt-5.5
LITELLM_API_KEY=REPLACE_WITH_SIMULATOR_KEY
LITELLM_BASE_URL=https://YOUR_GATEWAY_HOST

# RF rubric：这里的地址带 /v1
OPENAI_API_KEY=REPLACE_WITH_GRADER_KEY
OPENAI_API_BASE=https://YOUR_GATEWAY_HOST/v1
EVAL_MODEL=anthropic/claude-opus-4-5-20251101
```

`user-server/server.py` 自己在 `LITELLM_BASE_URL` 后追加 `/v1`；填成 `.../v1` 会产生 `.../v1/v1`。官方运行脚本还把该变量硬编码成 `https://api.openai.com`，所以使用网关时应采用下面的自定义命令，不要原样执行脚本。API 必须支持所填 model ID 和模拟用户需要的 tool calling。

先跑 DeepSWE 或 SWE-bench Pro 样例可以暂不使用 RF grader。若用 OpenAI 直连接口作为模拟用户，`LITELLM_BASE_URL=https://api.openai.com`；RF 默认 Anthropic rubric 模型仍需要支持它的网关，或明确变更 `EVAL_MODEL` 并记录协议差异。

在 `upstream/qw3-network.yaml` 写入：

```yaml
services:
  main:
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

Linux Docker 中，容器内的 `127.0.0.1` 指容器自身。当前主机服务只监听 loopback，不能直接给主容器使用。新服务可绑定 Docker bridge 的宿主机地址；本机检查值为 `172.17.0.1`。保留 Compose 默认网络，让 `main` 与 `user-server` 继续通过服务名互相访问。

## 5. QW3 服务示例

下面沿用 QW3 文档中的 1M logical context、64K selection、32K generation reserve，作为接通流程的示例，不代表最优质量配置。正式比较应冻结模型、QW3 二进制和采样参数，并使用独占服务。

```bash
cd /home/chaidi/qw3
qw3_docker_gateway=$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')
./build/qw3 serve \
  --model models/Qwen3.8-27B-Q8_0.gguf \
  --host "$qw3_docker_gateway" --port 18080 --ctx 1048576 \
  --enable-thinking --native-mtp-speculate --mtp-chain 4 \
  --kvmem --kvmem-budget 65536 --kvmem-gen-budget 32768 \
  --kvmem-query-conditioned --kvmem-update-mode step \
  --kvmem-guided-reselect both \
  --kvmem-guided-thinking-tokens 0 \
  --kvmem-guided-query-tokens 256 \
  --kvmem-middecode-trigger-tokens 28672 \
  --kvmem-middecode-max-refreshes 2 \
  > /home/chaidi/qw3/benchmark/swe_interact/qw3-server.log 2>&1
```

该命令前台持有服务；在另一个终端启动 Harbor。首次试跑前应检查所用容器能访问该地址及 Messages/count_tokens 路径。

## 6. 单任务试跑命令

```bash
cd /home/chaidi/qw3/benchmark/swe_interact/upstream
harbor run \
  --env-file harbor/.env \
  -p ./data/multiturn \
  -i deepswe_bandit-structured-nosec-directives \
  -a claude-code \
  -m Qwen3.8-27B \
  -e docker \
  --extra-docker-compose ./qw3-network.yaml \
  --force-build \
  -n 1 -k 1 -r 0 \
  --ak version=2.1.148 \
  --ak reasoning_effort=high \
  --ak disallowed_tools=Agent,EnterPlanMode,AskUserQuestion \
  --ae ANTHROPIC_API_KEY=local-qw3 \
  --ae ANTHROPIC_BASE_URL=http://host.docker.internal:18080 \
  --ae CLAUDE_CODE_MAX_CONTEXT_TOKENS=1048576 \
  --ae DISABLE_COMPACT=1 \
  -o ./results/qw3 \
  --job-name qw3-kvmem-bandit-smoke
```

参数解释：

- `-n 1` 是并发任务数，`-k 1` 是每任务采样次数，`-r 0` 是基础设施失败的重试次数。官方脚本的 `-n 25` 不是“只跑 25 题”，`-r 3` 也不是三次采样。
- `--force-build` 沿用官方设置，构建包含仓库检查服务的任务环境；首次构建可能耗时，后续可利用构建缓存。
- 禁用 `Agent` 沿用当前 QW3 runner 的单轨迹约束。此限制应在报告中声明；baseline/KVMem 两组都保持一致。
- `AskUserQuestion` 是 Claude 内置交互工具；benchmark 使用的是另一个 MCP 工具 `ask_user`，必须保持可用。禁用内置工具和 `EnterPlanMode` 可避免 print 模式等待真实终端输入；任务的规划阶段照常通过 MCP 完成。
- Harbor 从 task.toml 注册 `user-server` MCP，不需要改写 Claude 的主 agent 循环。
- reasoning effort 示例采用官方脚本的 `high`；现有 DeepSWE runner 用 `xhigh`。可以改为 `xhigh`，但需在所有对照组固定相同设置。
- `DISABLE_COMPACT=1` 是 KVMem 实验设置。普通 dense 对照若保留 Claude 自动压缩，应移除此变量；不要设成字符串 `0`。

本次只核对了上述选项对应源码，没有运行这条命令。首次集成仍需验证镜像可构建、容器网络、模型 ID、MCP 注册与外部 API 可用性。

## 7. 正式测试和结果判定

建议按以下顺序推进：

1. 单个 DeepSWE 任务接通五阶段流程，确认模拟用户实际给了多轮回复，而非 API 异常 fallback。
2. 三类任务各跑一个，验证尤其是 RF 的 rubric 路由和 hidden verifier。
3. 对固定任务集比较 singleturn 与 multiturn；singleturn 使用官方 `data/singleturn`。正式全量每种设置 75 个任务，按来源分别报告和总体汇总。
4. 若研究 KVMem，再做固定 solver、simulator、grader、预算和超时的 dense/KVMem 配对实验。外部模拟用户会依赖当前实现给不同反馈，因此两组是任务配对，不是相同对话轨迹。

至少保存：

- 每任务最终 reward/resolved、各来源通过率、基础设施失败及重试记录。
- 每阶段 Claude 原始 stream/session、Harbor trajectory 和 simulator 的 `ask-user.jsonl`、`tool-calls.jsonl`、`events.jsonl`。
- 实现 diff、独立测试 commit、canonical verifier 输出、RF rubric 分数。
- 最大单请求 prompt 长度、累计 input/output tokens、ask_user 次数、阶段重置或 compaction 事件。
- QW3 服务端请求时延、prefill/decode、KVMem refresh/cache、显存；模拟用户和 grader 的时间/费用独立计量。
- QW3、模型、Claude Code、Harbor、数据集版本和完整启动配置；当前工作区有未提交修改，仅记 Git HEAD 不足以冻结 QW3 实现。

公开论文的多轮主结果平均了 GPT 5.5 与 Opus 4.7 两种模拟用户的两次 trial，公开示例脚本只指定 GPT 5.5。单个 GPT 5.5 配置的运行应标明这一差异，不能声称完整复现论文表格。论文报告的累计 tokens/trial 也不等于单次上下文窗口长度。

## 8. QW3 KVMem 特别需要检查的点

当前 [anthropic_adapter.cpp](../src/anthropic_adapter.cpp) 把 Anthropic `tool_result` 转换为内部 `role=tool`；[harness_semantics.cpp](../src/harness_semantics.cpp) 用真实 `role=user` 寻找 root/current query。

因此从源码推断，MCP `ask_user` 返回的新需求一般不会单独触发“新 user query”路径；它先作为工具结果进入 live tail，随后依赖既有 guided reselection 等机制保留和检索。不是说模型看不到反馈，也不能在未实测前断言它一定遗忘。

首轮应在官方 MCP 协议下记录：新反馈进入后 current query、mandatory spans、live tail、guided refresh 如何变化。如果后续增加“识别 ask_user 语义边界”的 QW3 优化，应单独测试，并保留原始 MCP transcript；不要悄悄把反馈改成普通用户消息来宣称同一配置结果。

另一个相关变量是跨阶段会话：默认阶段重置会截断 Claude 原生历史。若目的是检验持续长上下文，需要额外开 `--resume-trajectory` 对照，并验证两组实际达到了所比较的上下文压力。不能仅凭任务叫 long-horizon 就推断触发了 256K 或 1M context。

## 主要源码入口

- [官方 README](https://github.com/scaleapi/SWE-Interact/blob/b32f98c3b8f76ca65e84341d1f30e5af7135f85d/README.md)
- [官方 Claude Code 多轮脚本](https://github.com/scaleapi/SWE-Interact/blob/b32f98c3b8f76ca65e84341d1f30e5af7135f85d/run_configs/multiturn/opus-4p8-high_claude-code.sh)
- [示例 task.toml](https://github.com/scaleapi/SWE-Interact/blob/b32f98c3b8f76ca65e84341d1f30e5af7135f85d/data/multiturn/deepswe_bandit-structured-nosec-directives/task.toml)
- [模拟用户 server.py](https://github.com/scaleapi/SWE-Interact/blob/b32f98c3b8f76ca65e84341d1f30e5af7135f85d/data/multiturn/deepswe_bandit-structured-nosec-directives/environment/user-server/server.py)
- [最终 verifier 示例](https://github.com/scaleapi/SWE-Interact/blob/b32f98c3b8f76ca65e84341d1f30e5af7135f85d/data/multiturn/deepswe_bandit-structured-nosec-directives/steps/05_test_handoff/tests/test.sh)
- [Harbor Claude Code adapter](https://github.com/laude-institute/harbor/blob/cdb76bae6dc88d5bca1c8f0754bbba300d6574b4/src/harbor/agents/installed/claude_code.py)
- [Harbor 多阶段会话默认配置](https://github.com/laude-institute/harbor/blob/cdb76bae6dc88d5bca1c8f0754bbba300d6574b4/src/harbor/models/trial/config.py)
