# 9B 摘要取消独立 thinking 限制

2026-09-20 用户授权：“10K thinking budget 可以去掉这个限制”。

512K Compaction+RAG 的前置两轮 Q8 摘要和后续 FP8 补充摘要均取消独立
thinking token 上限。llama.cpp 启动参数显式设为 `--reasoning-budget -1`，
避免环境默认值重新引入限制；vLLM 摘要请求省略 `thinking_token_budget`，
本地协议默认值为 None。冻结的原客户端和提示代码不修改，由现有摘要适配器
处理请求及配置记录。配置及 checkpoint 中的摘要 thinking budget 记录为 None。

仍开启 thinking，仍使用总输出 `max_tokens=32768`，包含思考和正文，
不是32768正文加无限思考。温度1、top_p=.95、top_k20、min_p0、
presence_penalty1.5、repeat/repetition_penalty1、摘要输入228336保持不变。
最终回答仍为温度.6、top_p=.95、seed20260722、thinking8192、输出32768。
这次调整不能保证解决提前 EOS 或空正文；不修改解析、不屏蔽EOS、不将思考
字段冒充摘要、不按答对与否重试。每个正式模型终态继续保留在固定分母中。

上一目录 `512k/compact_rag_samplingfix_20260920/` 保存15题成功摘要、
2题摘要生成失败以及中断中的请求，尚无最终回答。完整产物哈希保存在其中的
`superseded_nothinkcap_20260920.json`。中断请求不作为模型失败。
新目录 `512k/compact_rag_nothinkcap_20260920/` 从新摘要开始运行同一100题，
只复制模型无关的样本及检索产物，不复制旧摘要、失败或答案。

4项 CPU 回归测试覆盖真实客户端发出的摘要参数、FP8回答客户端隔离、
上下文容量及配置指纹、错误预算的拒绝。启动前核验锁定源码和100份检索哈希、
100题身份、后端命令差异，以及后续1M配置不变。线上 Q8 参数和返回值写入
新目录的 `launch_runtime_validation.json`；FP8线上记录须等该阶段实际启动，
当前用冻结客户端的真实序列化请求做离线核验。

已有 KVMem、≤256K 和27B结果不重跑。1M使用锁定的第三方摘要，没有新增摘要
生成。顺序 campaign 继续执行剩余阶段，每30分钟检测定时器保持禁用。

14:54 已恢复 campaign（PID 2127580）。实际 Q8 服务进程的参数为
`--reasoning-budget -1`；首个正式摘要请求已进入 prefill，审计确认总输出32768、
thinking开启、未发送独立thinking限额，运行配置中的摘要限额为None。
尚未进入FP8阶段，其请求省略预算字段及回答8192限制隔离已通过CPU测试。
