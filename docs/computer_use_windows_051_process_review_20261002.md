# 051 完整过程审核入口

GIMP 的最终 XCF 可以通过原图层评分，却无法证明过程中没有裁剪人物、反复旋转、
移动背景或临时改名。现有评分因此保留两个结果：原成果分和包含完整过程审核的
严格验收。不能仅凭最终图层把严格结果改为通过。

宿主机新增两阶段入口，审核输入和评分器不进入 Windows 客户端：

1. `osworld_windows.review051 prepare` 核对已停止的有效试次、文件收集回执、
   完整录像及其覆盖的启动至终态区间；从独立副本读取 SQLite 和 WAL，导出全部
   动作与观察，并核对截图。原始数据库、录像、任务结果和原评分不修改。
2. 审核者查看整个录像和轨迹，再填写独立审核文件：六项约束各自的决定与具体
   证据、审核身份、完整录像范围。初始模板保留未知项，不填入成功答案。
3. `osworld_windows.review051 apply` 重新核对原试次、成果及审核包哈希，要求
   整段录像审核和每项依据，用冻结录屏二进制解码完整视频，随后调用原图层评分
   及过程审核。结果写入新的 `independent-process-grade.json`；裁剪或旋转违规
   可以正常记录为失败，缺失证据不能成为通过。

两侧都采用这个入口。任务到时停止仍属于超时，即使文件或后续审核通过，也不
修改原执行终态。后续总表必须分别列出成果正确、过程约束结果与按时完整结束。

## 已执行与未执行的资格

六项宿主机控制通过：保留 WAL 中的完整动作、不完整录像/覆盖不足拒绝、篡改
媒体或缺失截图拒绝、外来/不确定动作拒绝、局部录像审核拒绝、旧审核包/变动
原证据拒绝。这些是合成合同检查，不是实际视觉审核或模型任务通过。

原真实模型预跑 `osworld-native051-on-02` 已生成第二版审核包，绑定 26 次动作、
44 次观察和 48 份来源。该试次没有 `final.xcf`，原严格失败不变；审核包状态仍为
`pending_full_visual_review`。第一版包保留，第二版修复读取副本数据库的资源关闭，
并记录准备工具哈希。完整从原始素材操作至合格成果的 Windows 过程资格仍待
执行；打开既有答案再保存的旧序列不能代替它。

- [实际预跑的审核包](/home/chaidi/qw3/apps/computer_use/.runtime/osworld-051-process-review-02/manifest.json)
- [未填写的审核模板](/home/chaidi/qw3/apps/computer_use/.runtime/osworld-051-process-review-02/review-template.json)
- [六项审核合同检查](/home/chaidi/qw3/apps/computer_use/.runtime/osworld-051-process-review-qualification-01/result.json)
- [队列、存储及过程审核的 23 项控制](/home/chaidi/qw3/apps/computer_use/.runtime/host-preparation-contracts-02/result.json)

## 使用方式

准备包使用宿主机应用环境即可；应用完整审核使用已验证的 OSWorld 评分环境，
因为它需要原评分的 OpenCV、PyMuPDF 及固定的无网络 XCF 解码器。
`--output` 必须是新目录；`--directory` 指向对应的完整原生试次，不能指向
只打开并保存答案的资格目录。

```text
PYTHONPATH=/home/chaidi/qw3/apps/computer_use \
  /home/chaidi/qw3/.venv/bin/python -m osworld_windows.review051 prepare \
  --directory <原生051试次绝对路径> --output <新审核包绝对路径>

PYTHONPATH=/home/chaidi/qw3/apps/computer_use \
  /home/chaidi/qw3/third_party/OSWorld-V2/.venv/bin/python -m osworld_windows.review051 apply \
  --package <审核包绝对路径> --review <完成视觉审核后的决定文件绝对路径> \
  --output <新独立评分绝对路径> \
  --ffmpeg /home/chaidi/qw3/third_party/OSWorld-V2/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
```
