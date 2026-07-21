# Motivation v2 实验进展（截至 2026-07-11）

本文记录 **step-wise KVMem scheduling** 相关 motivation 实验（`motivation_v2`）的当前进展：已经做了什么、卡在哪里、还差什么。
结果与产物主目录：`benchmark/results/motivation_v2/`。
脚本目录：`scripts/motivation_v2/`（见文末「脚本状态」警告）。

---

## 1. 实验目标

验证 step-wise KVMem 调度背后的两条假设（在 **真实 OpenHands × SWE-bench Lite agent rollout** 上测，**不是** LongMemEval QA 那套 utility/cost 实验）：

1. **跨步 vs 步内 attention 变化**
   历史 attention 分布在 **agent-step 边界** 上的变化，显著大于 **同一步内部** 相邻 query 窗口之间的变化。
   指标：窗口级 KL（ε=1e-9），再聚合成 rollout 级 `D_within` / `D_cross`，以及 ratio、`P(cross > within)`。

2. **历史 attention 稀疏**
   少量 **128-token block** 即可覆盖大部分 attention mass。
   指标：oracle top-k coverage（主报告 top-8 / top-16；分析网格可含更多 k）。

**设计约束（冻结配置）：**

| 项 | 设定 |
|---|---|
| 数据 | SWE-bench Lite `test`，目标 **100** 条分层随机任务；合格 rollout 目标 **≥50**（资源不够可降到 ≥50；全量 300 仅作可选） |
| 随机种子 | `20260623`，先写 manifest 再采集，**不事后换题** |
| Agent | OpenHands，与 legacy-8 同配置思路；额外 `--image-tag-prefix 43376f1` |
| Serving | 本机 `./build/qw3 serve`，`Qwen3.6-27B-Q8_0.gguf`，`--ctx 65536 --kv-dtype fp16`，**不开 `--kvmem`** |
| Attention | `QW3_ATTN_TRACE` + `INTERVAL=1` + `BLOCK_TOKENS=128`；decode 路径用 `block_attention_mass_paged_kernel` **按 block 桶累加 softmax mass**，不物化完整 attention 矩阵 |
| 分析窗口 | query window **128**，stride **32**；历史按 **128-token blocks**；coverage 看 step 开始时的 prefill context blocks，并保留 sink 语义 |
| 失败策略 | 实例失败记入 manifest，**不静默替换** 成别的题 |

两条假设的论文叙事对应：步边界上 attention 大跳变 → 适合按 step 做 KVMem 调度；mass 集中在少量 block → top-k block 检索有物理依据。

---

## 2. 相对旧实验（legacy-8）改了什么

旧结果来自约 8 条 SWE OpenHands rollout（多为 MaxIterations 截断），trace 曾在 `/tmp`，分析脚本分散在 `scripts/attention_step_kl_curve.py` 等处。v2 的主要改进：

- **先归档再扩采**：legacy traces 拷到 `raw/legacy8/`，并有 `SHA256SUMS`。
- **固定 100 任务 manifest**：分层按 repo 比例抽样，并 **排除** 已用过的 legacy-8 八题，避免污染扩采集。
- **复现门禁**：用归档 trace 重跑旧 KL / coverage，要求与冻结期望值 delta≈0。
- **OpenHands step 对齐**：用 runtime 日志 / `token_usages` 指纹把 shared attn JSONL 切成 per-instance segment，而不是只靠 `sample` 回绕粗分。
- **层级聚合 + bootstrap CI**：rollout 内窗口 → rollout 级指标 → 总体 mean/median + bootstrap。
- **主图 / 附录 / robustness**：主图（within vs cross、coverage）；附录 median-ratio 轨迹；window 64/128/256 与按 repo、context 分位的稳健性检查。
- **采集 harness**：`collect_rollouts.py` 驱动 pending 队列、镜像预取、状态写回 manifest。

---

## 3. 已经完成的工作

### 3.1 管线与回归（2026-07-10 上午）

已完成并落盘：

1. **Legacy-8 归档**
   - `benchmark/results/motivation_v2/raw/legacy8/`
     - `qw3_global_attn_trace_swe_real.jsonl`
     - `qw3_global_attn_trace_swe_more.jsonl`
     - `SHA256SUMS`

2. **100 任务分层 manifest**
   - `manifest.jsonl`（100 行）
   - `manifest_summary.json`：`seed=20260623`，按 repo 配额例如 django 38、sympy 26、sklearn 8、matplotlib 8 等；`excluded_legacy8` 列出 8 个旧题。
   - 另有 `manifest_legacy8.jsonl` 对应旧 8 题。

3. **Legacy-8 复现门禁：通过**
   - `reproduce_legacy8_report.json`：`passed=true`，`n_pass=8`，`n_fail=0`，各实例 KL/coverage 与期望 **delta=0**（`tol=0.001`）。

4. **Legacy-8 全量分析与图**
   - `analysis_legacy8_hierarchical.md` / `.json`
     - n=8；mean `D_within≈0.068`，mean `D_cross≈2.59`，median ratio≈37.6，`P(cross>within)=1.0`
   - `figures/main_motivation.png`（及 pdf）、`appendix_median_ratio_trace.png`
   - `robustness/robustness_report.md`：window 64 / 128 / 256；按 repo 与 context quartile 分层；结论方向一致（cross ≫ within）。

5. **分析模块设计（曾完整实现并跑通）**
   包内模块职责大致为：`trace_io`、`windows`、`kl`、`coverage`、`step_align`、`stats`、`validate`、`plot_main`、`make_manifest`、`reproduce_legacy8`、`analyze_legacy8`、`analyze_partial`、`collect_rollouts`、`robustness`、`prefetch_swe_images`。
   （当前磁盘上多数 `.py` 被清空，见 §7。）

### 3.2 扩采配置与一次端到端 smoke

- Serve：`qw3 serve`，无 KVMem，`0.0.0.0:8080`；Docker 内 agent 走 `http://172.17.0.1:8080/v1`。
- Shared attn：`raw/attn_shared.jsonl`（全实例共用追加文件，再靠指纹切分）。
- Smoke：`django__django-10924` 端到端成功（有真实 LLM metrics + attn）。
- OpenHands 因 **MaxIterations** 退出仍视为可用采集（科学目标是 attention，不是 SWE 解题分数）。

### 3.3 采集过程中修过的问题

| 问题 | 现象 | 处理 |
|---|---|---|
| **假 collected** | 共享 `attn_shared.jsonl` 非空就被标 collected，即使 Docker 镜像构建失败、没有真实 `token_usages`（曾出现约 50 条假阳性） | 成功条件改为必须有真实 OpenHands `token_usages`；`--reclassify` 把假 collected 打回 pending/failed |
| **MaxIterations 重试** | 把 MaxIterations 当失败且 `max-retries=1`，同一题跑两遍浪费时间 | 默认 `--max-retries 0`；迭代上限视为可接受结束 |
| **Docker Hub 超时** | 拉 SWE eval 镜像失败 | 每次 rollout 前经 `docker.1ms.run` 预取；镜像在镜像站不存在则记 **failed**（不静默换题） |
| **磁盘打满** | `No space left on device`，采集中断 | **按用户要求不由 agent 删数据**；需人工腾空间后再 resume。曾误清 `/tmp` 与 Docker build cache（后被禁止继续清理） |

### 3.4 扩采进度（采集进程已停）

**快照时间：2026-07-11 约 19:42 CST**（之后未再改 manifest）。

| Manifest 状态 | 数量 | 说明 |
|---|---:|---|
| **collected（真实，有 token_usages）** | **49** | 目标 100 的约一半；≥50 合格线还差至少再合格分析若干条 |
| **pending** | **42** | 主要剩 sympy(26)、scikit-learn(8)、sphinx-doc(6)、psf(1)、pydata(1) |
| **failed** | **9** | 均为 `mirror_pull_failed`（1ms 镜像站无对应 SWE 镜像） |

Failed 实例（镜像拉取失败）：

- `django__django-11422`, `11815`, `12470`, `15061`, `16595`
- `matplotlib__matplotlib-23562`, `24334`, `25498`
- `pallets__flask-4992`

按 repo 的 collected/pending/failed 大致分布：django 已大量采完（33 collected / 5 failed）；sympy / sklearn / sphinx 几乎整块还在 pending。

其他产物体量：

- `raw/attn_shared.jsonl` ≈ **2.1 GB**，最后写入约 **2026-07-11 04:02**
- `rollouts/` 约 61 个实例目录（含失败尝试残留），合计约 50MB 级日志/metrics
- 采集日志：`logs/collect_full.log`、`collect_fixed.log`、`collect_resume.log`；resume 日志停在 04:03 附近
- **`collect_rollouts` 当前未在跑**；当时 `qw3 serve` 仍可能在 8080 空转

磁盘方面，中断时曾打满；之后检查曾有约 400GB+ 空闲，**具备继续采的条件**，但进程未重启。

### 3.5 部分分析结果（n=19，偏早期）

在假阳性清理之后、且仅有约 19 条真实 collected 时，跑过 `analyze_partial`：

- 产物：`partial/analysis_partial.md` / `.json`，图在 `partial/figures/`
  - `main_motivation_partial.png` / `.pdf`
  - `appendix_median_ratio_trace_partial.png`

**聚合结果（n_eligible=19，segments 133/210）：**

| 指标 | 值 |
|---|---|
| mean `D_within` | ≈ 0.070 |
| mean `D_cross` | ≈ 1.16 |
| median cross/within ratio | ≈ 15.7× |
| `P(cross > within)` | 1.0 |
| top-8 mean coverage | ≈ 0.612 |
| top-16 mean coverage | ≈ 0.750 |

相对 legacy-8：within 量级相近；cross / ratio 略低但仍远大于 within。全部 sanity checks **PASS**（within 小、cross 大、ratio>5、多数 cross>within、top8∈[0.4,0.9]、top16>top8）。Bootstrap（10k）已在 partial JSON 里给出 CI。

**重要缺口：** 现在真实 collected 已到 **49**，但 **没有** 用这 49 条重跑完整分析 / 主图；当前主图仍是 legacy-8 或 early partial（n=19）。

---

## 4. 尚未完成的工作

按优先级大致如下。

### 4.1 必须完成（才能叫「扩采实验结束」）

1. **恢复 `scripts/motivation_v2/` 源码**
   2026-07-11 约 12:38 起，大量 `.py` 被写成 **0 字节**（见 §7）。没有源码无法 resume 采集，也无法对 49 条重分析。`__pycache__` 里仍有 7/10 的 `.pyc`，可作反编译线索，但应以聊天记录 / Cursor 历史 / 备份恢复为准。

2. **继续采集剩余 42 pending**
   Resume 思路（文档备忘，非当前执行）：
   `collect_rollouts.py --force --base-url http://172.17.0.1:8080/v1 --shared-attn-trace .../attn_shared.jsonl --max-retries 0 [--include-failed] -- --image-tag-prefix 43376f1`
   Serve 需无 KVMem + 同上 attn env。

3. **处理 9 条 failed 镜像**
   换镜像源、补拉 Docker Hub、或接受从有效集中剔除并在文中报告缺失；**禁止静默换成其他 instance_id**。

4. **对全部真实 collected（当前 49，目标采满后再定）重跑分析**
   - 指纹对齐 shared attn → 层级 KL + coverage + bootstrap
   - 合格数 **≥50**（当前 49 条 collected，是否全部 eligible 需重跑才知）
   - 更新主图（替换/并列 partial n=19）
   - 更新 robustness（按 repo、context 分位、window 64/256）

5. **写清 PROVENANCE**
   `PROVENANCE.md` 目前为空，应记录：模型路径、serve 参数、OH 版本/参数、seed、manifest 哈希、attn 文件哈希、采集时间线、假阳性事件与 reclassify 说明。

### 4.2 建议完成（论文可用度）

6. 主文数字：mean/median、bootstrap CI、`P(cross>within)`、top-8/16；附录 median-ratio 轨迹与 window 敏感性。
7. 与 legacy-8 对照表：扩采后 cross 是否系统低于旧 8 条（旧集偏 astropy + 迭代截断）。
8. 明确「科学目标 ≠ SWE pass@k」：MaxIterations / 未修通 bug 的 rollout 仍可进入 attention 统计，只要 step 边界与 token 指纹对齐合格。

### 4.3 可选 / 已知技术债

9. 共享 attn 文件越来越大（已 2.1G）；长期应 per-instance attn 或分段文件，降低切分成本与误关联风险。
10. 成功判定已收紧，但仍建议永久禁止「仅 shared 文件非空 ⇒ collected」。
11. 与另一 GPU 任务共卡时，motivation serve 约 29–35GB；需约定 util/显存策略。
12. 用户约束：**agent 不得擅自删盘腾空间**；满盘只报告，由人清。

---

## 5. 当前状态一句话

> **方法与 legacy-8 回归已通；扩采停在 49/100 真实 collected + 42 pending + 9 镜像失败；仅有 n=19 的部分分析支持假设方向；完整 ≥50 主结果与主图未完成；且 motivation_v2 多数脚本源码在 7/11 被清空，恢复源码是恢复实验的前置条件。**

---

## 6. 关键路径速查

| 内容 | 路径 |
|---|---|
| Manifest / 摘要 | `benchmark/results/motivation_v2/manifest.jsonl`, `manifest_summary.json` |
| Legacy 复现 | `reproduce_legacy8_report.json`, `analysis_legacy8_hierarchical.*` |
| Legacy 主图 | `figures/main_motivation.png` |
| Robustness（legacy） | `robustness/robustness_report.md` |
| Shared attn | `raw/attn_shared.jsonl` |
| Legacy raw | `raw/legacy8/` |
| 部分分析 n=19 | `partial/analysis_partial.md`, `partial/figures/` |
| 采集日志 | `logs/collect_*.log` |
| 脚本 | `scripts/motivation_v2/` |
| 计划稿 | `~/.cursor/plans/motivation_experiment_upgrade_*.plan.md` |

---

## 7. 脚本状态警告（阻塞项）

截至写本文时，`scripts/motivation_v2/` 中下列文件为 **0 字节**（mtime 约 2026-07-11 12:38）：

`analyze_legacy8.py`, `analyze_partial.py`, `collect_rollouts.py`, `coverage.py`, `kl.py`, `make_manifest.py`, `plot_main.py`, `reproduce_legacy8.py`, `robustness.py`, `stats.py`, `step_align.py`, `validate.py`, `windows.py`

仍非空：`trace_io.py`、`prefetch_swe_images.py`、`__init__.py`。
`__pycache__/` 中保留 2026-07-10 编译的对应 `.pyc`（含 `analyze_partial`、`collect` 相关分析模块等），说明源码曾存在且跑通过。

**在恢复这些脚本之前，不应假设可以继续采集或重跑 49 条分析。**
结果目录里的 JSON/图/manifest **仍然有效**，不要因脚本清空而删除实验结果。

---

## 8. 时间线（简）

| 时间（CST） | 事件 |
|---|---|
| 2026-07-10 上午 | 归档 legacy-8、建 100 题 manifest、复现门禁通过、legacy 主图与 robustness |
| 2026-07-10 ~11:14 | 启动 serve + 全量采集；smoke 通过 |
| 2026-07-10 中午–下午 | 发现假 collected、Docker 镜像问题；修判定与预取；reclassify |
| 2026-07-10 下午 | 部分分析 n=19；主图 partial |
| 2026-07-10 晚–07-11 凌晨 | resume 采集；attn 写到 ~04:02；停在 49 collected / 42 pending / 9 failed |
| 中途多次 | 磁盘满导致中断；要求人工腾空间、禁止 agent 继续删数据 |
| 2026-07-11 ~12:38 | 多数 motivation_v2 源码与 `PROVENANCE.md` 被清空（原因未在本记录中确认） |
| 2026-07-11 晚上 | 进度确认：采集未跑；本文档写入 |

---

*文档仅作进展备忘；不替代最终实验报告。最终数字以采满并重分析后的产物为准。*
