# Mac 测试接入

Windows 虚拟机阶段已完成，结果见[验收记录](computer_use_acceptance_20261001.md)。
Mac 桌面执行器、打包及实际任务验收仍未完成；
本页和检查脚本用于准备测试环境，不构成 Mac 支持已经完成的证明。

## 提供连接信息

提供这台 Linux 宿主机能够连接的既有 SSH 别名，或 `用户名@内网地址`，
以及 Mac 的芯片类型、macOS 版本即可。无需在聊天中提供密码或私钥。
用专门的测试账户登录 Mac 的图形桌面，并在测试时保持桌面解锁。

如果尚未打开 SSH，在 Mac 的“系统设置 → 通用 → 共享”中打开“远程登录”，
并选择允许连接的测试账户。[Apple 的远程登录说明](https://support.apple.com/zh-cn/guide/mac-help/mchlp1066/mac)

SSH 用于传送构建、运行检查和取回结果。桌面截图和输入还需要实际执行器的
屏幕录制、辅助功能权限；这些权限会在该 Mac 上通过正常的系统授权流程确认。
不同启动方式可能具有不同的进程授权身份，因此 SSH 登录成功不等于桌面控制已经可用。

## 先检查环境

仓库提供标准库脚本 `scripts/computer_use_macos_preflight.py`。在 Mac 上使用已有
Python 3 运行它，可以记录系统版本、当前用户与图形控制台用户、活动显示器、
显示模式，以及当前进程的屏幕录制和辅助功能授权状态。

```bash
python3 scripts/computer_use_macos_preflight.py --output qw3-macos-preflight.json
```

有可用别名后，也可以从本仓库所在宿主机运行以下命令；将 `qw3-mac` 换成
已配置的真实别名：

```bash
ssh qw3-mac 'python3 -' < scripts/computer_use_macos_preflight.py
```

脚本返回 `0` 表示本次进程的初步条件满足，返回 `2` 表示条件不满足、平台不符
或检查出错。它只做只读查询，不发起权限申请。报告明确保留
`desktop_actions_tested=false` 和 `screenshot_coordinate_mapping_tested=false`：
显示模式的尺寸不能代替实际截图与坐标映射验证。

实现依据：[CGPreflightScreenCaptureAccess](https://developer.apple.com/documentation/coregraphics/cgpreflightscreencaptureaccess%28%29)、
[AXIsProcessTrusted](https://developer.apple.com/documentation/applicationservices/1460720-axisprocesstrusted)、
[活动显示器查询](https://developer.apple.com/documentation/coregraphics/cggetactivedisplaylist%28_%3A_%3A_%3A%29)。

## 后续实际验收

接入后先部署与 Windows 客户端相同协议的 Mac 执行器，核对真实截图和输入坐标，
再验证 Retina 缩放、焦点变化、中英文输入、权限拒绝、暂停接管和恢复。
随后运行浏览器表单、文档保存、跨应用复制等同类任务，并独立检查输出文件。

第一阶段仍可使用当前 Linux 宿主机的本地 GPU 推理。这样能够验证 Mac 客户端和
桌面执行器，但不能据此宣称模型已经在 Mac 自身的 CPU/GPU 上运行。

当前验证仅为 Python 语法检查、命令行帮助，以及 Linux 上正确返回“不支持该平台”。
Apple API 调用和 Mac 图形会话尚未在真实 Mac 上运行。

物理 Windows 目标的接入边界见[物理机测试接入](computer_use_physical_test_access_20261001.md)。
