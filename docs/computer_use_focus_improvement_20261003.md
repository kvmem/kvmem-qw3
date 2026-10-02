# Windows 应用焦点修复与同条件复测

固定操作诊断中，原执行器三次切窗均失败；最终候选三次全部通过，九次应用启动
的焦点回执也全部确认成功。这证明了一个执行层缺口得到修复，尚不证明模型规划、
复杂任务或 Python/LaTeX 工作流成功率有所提高。

## 试验和结果

恢复同一个 `qw3-osworld-client15-20261002` 快照，每次准备中性应用状态，顺序执行
Edge → Notepad → WPS Spreadsheets。独立检查操作返回时及两秒后前台进程/窗口标题。
测试不使用模型、KVMem 或顾问；UIA 仅用于基础设施准备。历史原始记录不改写。

| 版本 | 前台检查通过/有效试次 | 新增焦点回执确认/启动次数 | 发现 |
|---|---:|---:|---|
| 原执行器 | 0/3 | 未提供此字段 | Edge 启动后前台仍为 Notepad |
| 第一次修改 | 0/3 | 5/9 | 直接请求激活仍被拒绝；过早接受过渡窗口 |
| 第二次修改 | 3/3 | 8/9 | 画面正确，但一次把 Edge 通知气泡当成第二个应用窗口 |
| 最终候选 | 3/3 | 9/9 | 画面检查和新的回执检查均通过 |

原执行器三次有效试次来自 `local-diagnostic-p0-executor-01` 的第二、三次切窗和
`local-diagnostic-p0-executor-02` 的补测。首轮 16 次尝试中的 4 次准备无效继续保留，
不混入此表的有效操作分母。三个修改版本共新增九次试次，没有删除第一次修改的失败。

## 实际修改

- 启动前后识别窗口及进程，优先匹配新窗口或启动进程；不能消除歧义时返回原因。
- 确认窗口及标题稳定后返回，观察总等待有六秒上限；不会重复执行启动命令。
- 直接激活被拒绝时，检查菜单和已按住的输入，最多发送一次 ALT 按下/释放来尝试
  恢复焦点。结果仍以真实前台窗口为准，不修改系统全局焦点策略，不附加输入线程。
- 依据 Win32 窗口归属和工具窗口样式过滤通知气泡，避免依赖“Restore pages”等
  特定语言文案；被模态窗口阻塞的主窗口仍不会被强行激活。
- 回执区分“命令已投递”和“焦点已确认”。确认失败时，应用返回带当前截图的可恢复
  工具错误，不盲目重启、不把已知的焦点失败误报为未知投递。

实现参考 Microsoft 对 [前台激活条件](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setforegroundwindow)
及 [ALT 与前台锁关系](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-locksetforegroundwindow)
的说明；能否奏效以本次 Windows 记录为准。

可靠性改善需要额外等待。此小样本中 Edge 旧回执在 1.29–1.54 秒返回但未获得焦点，
最终候选在 2.26–2.81 秒返回并确认焦点。不能将两者当作相同成功语义的速度排名。
最终候选保留 127.90 秒连续 VNC 录像；两个中间版本也都有完整录像。

## Windows 客户端验证

新归档 `windows-native-client-17` 增加 `start.ps1 -Thinking`，默认仍关闭，运行器
只允许支持该能力的新归档启用。Windows 原生 Python 3.13.13、Codex 0.159.2 下
**89 项运行时测试通过**，包括 thinking 多轮历史、取消和焦点失败回执。

归档 SHA256：`b58cbdb3ae9aee0a78740979e5133b08137f7ac5384bc17173cd5ef3b880ac63`。
这是脚本化推理后端的真实 Windows/Codex 合约测试；此前真实 QW3 thinking 探针运行
在 Linux。尚未把两种证据合称为“Windows 上真实 QW3 thinking 业务验收”。

## 证据与复现

- 双栏连续回放：`apps/computer_use/.runtime/focus-comparison-replay-20261003/index.html`，可选择三次试次和三种应用；仅复制了所需视频，不包含凭据。使用支持 Range 的本机服务访问 `http://127.0.0.1:18108/index.html`；可用 `.venv/bin/python apps/computer_use/.runtime/focus-comparison-replay-20261003/serve.py` 重启。
- 汇总与原始结果哈希：`apps/computer_use/.runtime/local-diagnostic-p0-focus-audit-01/result.json`
- 三个版本：`.runtime/local-diagnostic-p0-focus-v1-01`、`-v2-01`、`-v3-01`。
  每个目录含冻结脚本/执行器、逐次回执、截图、SQLite、连续录像和录像清单。
- 通知气泡归属实测：`.runtime/local-diagnostic-p0-window-inventory-01.json`。
- Windows 合约结果：`.runtime/windows-client-17-qualification-01/result.json` 及 `native.stderr.log`。

上述相对路径均以 `apps/computer_use` 为根。旧快照的源码校验仍保留，不能拿当前
修改后的执行器直接冒充原基线。诊断运行器通过 `--baseline-sources` 指定保存的旧源码，
严格恢复后再部署候选；每次准备还核对实际 guest 执行器 SHA256。

```bash
PYTHONPATH=apps/computer_use .venv/bin/python apps/computer_use/local_diagnostic_executor.py \
  --output apps/computer_use/.runtime/NEW-UNIQUE-RUN \
  --baseline-sources apps/computer_use/.runtime/local-diagnostic-baseline-vm-daaaf6c \
  --categories launch_focus --repetitions 3 \
  --ffmpeg third_party/OSWorld-V2/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
```

这里未证明所有应用、多窗口文档选择、权限弹窗或物理桌面的兼容性。P1/P2、故障恢复、
真实安装配置和顾问接入仍按持续目标继续；原 OSWorld V0 未因本次局部通过而冻结。
