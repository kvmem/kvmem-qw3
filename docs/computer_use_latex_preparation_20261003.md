# Windows LaTeX 产品流程资格检查

2026-10-03。已下载、校验安装资产并通过用户便携安装及程序版本检查；中文公式编译
仍在开发验证。以下全部为脚本资格检查，不计入本地模型产品试次，也不是正式验收。

## 已核实资产

下载使用无代理 HTTPS，两个文件均匹配官方下载页的 SHA-256。
来源：[官方下载页](https://miktex.org/download)。

| 文件 | 官方 SHA-256 |
|---|---|
| basic-miktex-25.12-x64.exe | `14b42dd9f4b4a7813a8bfd69c8f99316c2888cc4ee26f631f397e163d85d6c62` |
| miktexsetup-5.5.0+1763023-x64.zip | `0571e90f6d94353089b4f189fd82a532f9fe559a388c7e7f1102b14b3c1ae27d` |

Windows Authenticode 检查两者均为 `NotSigned`；不能将官网哈希校验表述为签名通过。
证据 `.runtime/latex-assets-01/download-result.json`、预检中的 asset-metadata 输出。

## 实际失败、修正与边界

1. `latex-preflight-01`：便携模式与 `--user-install` 参数互斥，即使无人值守仍弹出错误框。
   已记录超时、终止进程及独立核对：无残留安装器、无安装目录、PATH 未变。
2. `latex-preflight-02`：改用 `--portable=目录` 后命令退出 0，但 `--dry-run` 仍写入
   18537 个便携目录文件。缺少 xelatex 启动程序，不能视为可用安装，也不能把 dry-run
   当作无副作用检查。该资格检查判为失败；后续真实安装使用另一空目录。
3. `latex-install-qualification-01`：实际安装失败。日志显示管理员测试控制台触发了
   默认共享安装，后续 `mpm --admin` 与便携模式冲突。关闭错误提示后主安装器仍退出 0，
   但 xelatex 不存在；再次说明不能以退出码或进度条判定成功。无残留进程、PATH 未变。
   关闭错误提示属于人工调试，记录在 `manual-dismissal.json`，不计作无人干预任务。
4. `latex-install-qualification-02`：新增 `--private`，从另一空目录完成安装，
   xelatex 和 miktex 版本命令通过、PATH 未变。实际报告为 MiKTeX 25.12 Portable、
   MiKTeX-XeTeX 4.16。

已验证的基础安装参数是：

```text
basic-miktex-25.12-x64.exe --unattended --private --portable=<独立目录> --no-registry --no-additional-roots
```

参数语义参考[官方安装说明](https://docs.miktex.org/manual/setupwiz.html)及
[固定提交的参数实现](https://github.com/MiKTeX/miktex/blob/76d1b3bd2b69857f30133c62ff9ab66c83dcf80f/Programs/MiKTeX/Setup/MFC/Setup.cpp)。
该源码作为排查参考，不声称其提交与下载二进制完全一致；最终以以上真实二进制检查为准。

## 中文编译及后续接入

`latex-compile-probe-01` 因尝试读取 dry-run 树中的 xelatex 失败；原记录保留。
`latex-compile-probe-02` 在实际安装后执行，确认中文文档因 `ctexart.cls` 缺失而失败。
其中 package-info 多参数查询也失败，后续改为逐包查询。旧探针的 passed 仅表示探针
顺利收尾，不表示命令均成功或 PDF 通过；以各命令退出码为准。

固定文档使用 ctexart、amsmath、SimSun 与 Microsoft YaHei；关闭自动装包和 shell escape，
缺包不弹出交互安装请求。中文字体身份已采集哈希。基础安装的 LaTeX 内核为 2025-11-01，
当前 CTAN 中文包要求 2026-06-01，因此正在验证同一冻结仓库中的内核及依赖更新。

候选仓库由 CTAN 镜像直连下载，包文件先核对仓库 MD5/大小，再记录本地 SHA-256；
这些包的 SHA-256 是下载后固定的校验值，不冒称官网公布。`latex-compile-candidates-01`
发现仅有旧格式数据库不足以识别本地仓库，已保留失败，补充 pr.ini 和新版 zzdb3 再验证。
来源：[官方包管理](https://docs.miktex.org/manual/miktex-packages.html)、
[格式生成](https://docs.miktex.org/manual/miktex-formats.html)。

后续依次完成：两次新进程编译、PDF 内容/页数/渲染独立检查；冻结实际需要的包集合；
将已验证命令接入软件作业的安装、诊断、取消和恢复；真实模型运行及同条件新建/修复
各三次正式验收。新建任务不能复用已经安装好的最终环境；可用下载缓存须明确标注。

上述 `.runtime` 均相对 `apps/computer_use`。所有失败目录保留，不覆盖或删除。
