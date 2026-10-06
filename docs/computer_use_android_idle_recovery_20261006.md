# Android 空闲会话恢复：独立候选与实际验证（2026-10-06）

新增可单独启用的 Android 恢复组件，解决 Appium 长时间没有命令后会话失效、读屏无法继续的问题。最终私有包的十四项 CPU/本机 HTTP 检查和十一项实际模拟器检查通过。独立核验确认全部四十三项系统设置和前台应用保持一致，两次 Appium 会话创建分别是初始连接和一次恢复，没有任何 Appium 输入命令。新模拟器、生产者、客户端和录屏资源已退役，两个端口拒绝连接。

恢复只接受读屏返回的明确 `invalid session id`，要求输入已暂停、没有未确认动作或未决会话创建，并核对同一个模拟器启动代次和自有 Appium 实例。创建连接前持久保留一次性预留；超时不重复 POST。旧截图立即失效，恢复后重新读屏并保持暂停。活动状态、恢复中的 resume、旧截图和未确认输入都被拒绝；不会重放动作或自动继续模型任务。

组件复用原 AndroidDriver、Gateway、设备租约和公开接口。候选复制原生按钮框架的完整固定清单，只增加 `android_idle_recovery.py`，原有文件零修改。入口为 `android_idle_recovery.serve(args, expected_identity, read_identity)`；调用方提供已选择设备的启动身份及只读核验回调。冻结的客户端和队列仍使用原网关，**尚未把候选设为默认**。跨恢复连接的模型任务检查点续跑仍需实际 QW3 验证。

最终使用一台独立模拟器、已有固定镜像和未运行的 AVD 种子，软件渲染，不占 GPU。测试把 Appium 闲置期限设为五秒来制造真实失效；组件默认仍为 3600 秒。未确认输入检查使用明确的本地数据库夹具，没有制造真实未知输入。完整回放为十六帧、1080×2400 的实时 ADB 采样循环，目标 1 Hz，记录了实际采集时间；不是历史截图重建，也不是全帧率视频。

早期记录保留：首次准备误用了缓存中不存在的 `userdata.img`，未创建容器；第二次新容器缺少 KVM 数字组权限，未进入测试；第三次恢复已成功，但完整设置不变检查失败，观察到新增的默认声音配置项。后续先等待明确的首次初始化标记和全部配置稳定，仍比较完整键值，没有忽略变化字段。先前未打包版本通过并保留十七帧录像；最终交付包另行验证，不能累加为模型任务成功。旧十二项 CPU 记录也不是最终十四项的额外覆盖。所有已创建的早期容器和生产者均独立确认退役。

最新模型测试状态已变化：原七个控制器均达到停止状态且实际退出。图像组保留 1/12，其余为 0/4、0/8、0/8、0/4、0/8、0/8；图像组因原六小时显存等待期限到达而停止，其余按前序保护停止。实际可用显存 55711 MiB，低于冻结的 56000 MiB 连续三十秒门槛，没有干预用户 GPU 进程。下一步需取得可用测试窗口，再资格验证保留完成前缀的续跑与新部署，不重放首个已完成样本。

此轮证明连接和读屏恢复能力，没有新增 QW3 业务任务，没有复杂 OSWorld 或 KVMem 成功率改善的结论。目标保持 active。

证据：[完整报告](/home/chaidi/qw3/docs/computer_use_android_idle_recovery_20261006.json)、[最终实际独立核验](/home/chaidi/qw3/apps/computer_use/.runtime/android-idle-recovery-live-independent-audit-02/result.json)、[最终十四项资格](/home/chaidi/qw3/apps/computer_use/.runtime/android-idle-recovery-cpu-02/result.json)、[私有候选清单](/home/chaidi/qw3/apps/computer_use/.runtime/android-idle-recovery-candidate-20261006-01/build.json)、[最终实时回放](/home/chaidi/qw3/apps/computer_use/.runtime/android-idle-recovery-live-05/continuous.mp4)与[七个原代次的停止核验](/home/chaidi/qw3/apps/computer_use/.runtime/android-idle-final-seven-controller-status-01.json)。

在仓库根目录，用新的输出目录复现构建和实际验证；固定镜像与 Android 依赖须已经存在，入口不下载镜像，不启动模型：

```bash
PYTHONPATH=apps/computer_use:python .venv/bin/python -m qualify_android_idle_recovery \
  --output apps/computer_use/.runtime/android-idle-recovery-cpu-new
PYTHONPATH=apps/computer_use:python .venv/bin/python -m android_idle_recovery_candidate \
  --base apps/computer_use/.runtime/native-control-candidate-20261005-02/build.json \
  --cpu apps/computer_use/.runtime/android-idle-recovery-cpu-new/result.json \
  --output apps/computer_use/.runtime/android-idle-recovery-candidate-new
PYTHONPATH=apps/computer_use/.runtime/android-idle-recovery-candidate-new/framework:apps/computer_use:apps/computer_use/.runtime/media-qualification \
  third_party/OSWorld-V2/.venv/bin/python -m qualify_android_idle_recovery_live \
  --output apps/computer_use/.runtime/android-idle-recovery-live-new \
  --cpu apps/computer_use/.runtime/android-idle-recovery-cpu-new/result.json \
  --build apps/computer_use/.runtime/android-idle-recovery-candidate-new/build.json --port 19680
PYTHONPATH=apps/computer_use:apps/computer_use/.runtime/media-qualification \
  third_party/OSWorld-V2/.venv/bin/python -m audit_android_idle_recovery_live \
  --source apps/computer_use/.runtime/android-idle-recovery-live-new \
  --output apps/computer_use/.runtime/android-idle-recovery-audit-new
```
