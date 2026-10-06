# Mac 实机测试准备：2026-10-06

已通过用户提供的 Tailscale 地址，用专用 SSH 密钥连接实际 Mac。机器为 Intel
架构，macOS 13.7.7，测试账户已登录图形桌面。私钥和连接地址留在被 Git 忽略的
运行目录，没有提交到代码库。

机器尚未安装 Command Line Tools。此前调用系统 Swift、xcrun 和 Python 启动器
触发了 Apple 开发工具安装提示，不能据此认为这些工具可用。当前准备步骤使用
系统自带的 JXA 和 osacompile，不需要安装开发工具或本地模型。

## 已完成的准备和实际验证

独立的只读应用 **QW3 Mac Test.app** 已安装在 Mac 的
`/Users/sunny/Applications/QW3 Mac Test.app`。打包、Info.plist 验证和本地 ad-hoc
签名验证通过；实际启动到 sunny 的图形会话，并在该应用自身进程内检查权限。
首轮打包因生成的 plist 没有 CFBundleIdentifier 而中止，补齐应用身份后验证通过；
原失败记录保留，不计作桌面任务或模型失败。

真实应用报告主显示器逻辑尺寸 1680 × 1050，backing scale factor 为 2。这个信息
只证明显示器元数据可读，不能代替真实截图尺寸与输入坐标映射验证。首次运行时，
屏幕录制和辅助功能检查均为 false。随后独立复查发现应用已退出，原因未确定，
保留的 status.json 没有被当作当前授权状态或存活证明。用户回复已开启两项权限
后，重新启动并核对了新 PID、实际应用路径和新时间戳；应用自身进程内的两项
权限均实测为 true。接入与授权准备已完成。

应用仅检查权限和显示器元数据，不截图、不发送键鼠输入、不请求模型、不修改
系统权限、不触发授权对话框。Windows、Linux、Android 执行器和正在运行的冻结
复测源码未修改。源码见
[`permissions.js`](../apps/computer_use/macos_probe/permissions.js)，验证快照和证据哈希见
[`computer_use_macos_preparation_20261006.json`](computer_use_macos_preparation_20261006.json)。

## 已完成的 Mac 本机授权

在“系统设置 → 隐私与安全性”中为 **QW3 Mac Test.app** 开启屏幕录制和辅助功能。
需要添加应用时，使用实际路径 `/Users/sunny/Applications/QW3 Mac Test.app`。
本次用户已完成这两项设置，重新打开后已验证生效。后续测试仍须保持测试账户
桌面登录、解锁并避免休眠。

这些是 macOS 的正常本机授权要求：[屏幕录制说明](https://support.apple.com/guide/mac-help/mchld6aa7d23/13.0/mac/13.0)、
[辅助功能说明](https://support.apple.com/guide/mac-help/mh43185/13.0/mac/13.0)。
如果授权列表无法添加或找不到应用，需要调整应用的权限请求流程；不能直接改变
系统权限数据库，也不能把 SSH 进程的授权视为图形应用已经授权。

## 尚未完成的测试

Mac 桌面执行器仍待实现和验收；目前仅完成连接和权限准备。后续按顺序执行：

1. 接入 Mac 执行器；再次核对实际执行应用的进程身份和两项权限，取得真实截图。
2. 实测 Retina 下截图像素、系统逻辑坐标、控件坐标之间的换算，处理前台变化和
   旧截图失效。
3. 验证默认暂停、单一输入所有者、持久操作回执、拒绝后无输入、结果未知时不
   重发，以及用户接管。
4. 在独立测试文件中验证文字输入、保存和重新打开；再接入现有本地 QW3 harness，
   执行浏览器和跨应用任务，记录视频并独立验收业务结果。

模型继续运行在现有 Linux GPU 主机。Mac 端负责桌面操作和记录，这些测试不能
证明模型已在 Mac 自身的 CPU 或 GPU 上本地运行。后续更换或重新打包执行应用后，
仍须重新核对实际应用的授权和进程身份。
