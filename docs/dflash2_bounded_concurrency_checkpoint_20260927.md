# Bounded并发的checkpoint前置修正

2026-09-27。当前只完成前置源码修改，bounded与continuous batching入口仍拒绝组合。
既有MMA长历史测量运行封存二进制，不包含本次修改。

串行与ragged verifier原来分别设置checkpoint ready标志，后者没有填写bounded append
身份。新增 `QwenExecutor::seal_state_checkpoints`，统一填写repository所有者、append
序号及工作集generation。序号只在一个repository内唯一，因此恢复时同时检查所属
repository；共享所有者引用也防止销毁、重新分配后地址复用导致误判。
串行重用checkpoint容器时清除旧owner，错误恢复在任何GPU状态写入前拒绝。

`ContinuousBatchActive`、`ContinuousPrefillBatchEntry`和输出中的历史offset/total
改为64位，避免把bounded历史位置提前截断到32位。GPU局部window与page metadata
仍为int32，构造时显式检查范围；历史区间加法检查uint64溢出。
这不等于前端、repository和所有独立prefill路径已经具备完整的64位历史并发支持。

模型探针增加实际有效checkpoint被换成另一repository owner后必须拒绝恢复的用例，
同时确认位置不变，再恢复原owner继续正常投机流程。该GPU用例尚未执行。
本次executor/backend和探针的CPU语法检查已通过；第一次手工检查缺少构建系统提供的
`QW3_SOURCE_DIR`宏，补齐原构建定义后通过，并非放宽源代码检查。

`bounded_checkpoint_build01`已完成冻结源码的完整CPU构建，40项CTest全部通过。
构建取得现有GPU实验锁后执行，未将重编译负载混入性能测量。后续CUDA构建必须同步重编译使用
`StateCheckpointSet`的backend、executor及所有相关探针，禁止与旧布局混链。
GPU检查仍未执行；CPU通过不代表bounded并发已经可用。

仍需实现共享arena下的bounded准入及配置、逐请求repository/GC/CPU/NVMe资源预算、
私有query与回滚取消生命周期，以及跨slot回收和长历史GPU正确性；这些限制没有删除。

后续工作区已准备executor层的共享arena接入：
`QW3_EXPERIMENTAL_BOUNDED_CONTINUOUS=1`允许private repository目录映射到共享物理页池，
要求显式allocator及匹配的物理容量；MTP若启用则独立验证其arena。每请求仍通过
`SharedKvPagePool`持有自己的页配额。ragged verifier能力查询接入这个开关，普通body
batch和联合prefill仍不开放。CPU语法检查通过，尚未进行该变更的CUDA/GPU验证。
server/backend的bounded并发准入限制仍保留；这不是已可用的bounded并发入口。
此修改发生在`concurrency_mma_diagnostic_build01`冻结之后，未混入该数值诊断。

## 实验服务接入（源码完成，GPU待验证）

后续`bounded_concurrency_build01`冻结以下改动；不再仅停留在executor准备：

- backend/server在`QW3_EXPERIMENTAL_BOUNDED_CONTINUOUS=1`下允许bounded与
  continuous组合，要求完整cooperative controller和ragged验证；generic页级前缀
  缓存、layered验证、已有不兼容请求仍拒绝。无法路由的请求明确失败，禁止落到主
  executor串行执行而越过并发资源预留。
- 每个bounded controller都有独立`KvmemPrefixState`，无key的临时请求也不借用
  backend全局warm state。仅有key且正常完成的状态可park；取消/异常先展开栈、
  释放checkpoint再销毁executor。各controller的semantic-budget作用域在本executor。
  当前请求支持检查仍只接受默认semantic budget，未声称开放任意逐请求预算API。
- CPU/NVMe tier配置在此实验模式下作为所有owner的总额度，按`max_active + 1`
  分配：至多max_active个运行或驻留会话，加现有setup executor。关闭请求executor
  的无界idle host-tier缓存，换入前先销毁超出槽数的驻留会话。此额度是KV tier额度，
  不把模型映射、索引、I/O staging和其他host scratch误称为包含在内。
- 共享GPU arena在load时验证至少容纳每槽K/G工作集及private-query预留，不降低K/G。
  K128K/G64K、8192 private reserve、page16时每槽12800页，C4至少51200页。
  共享arena已经分配，executor不再用当前空闲显存重复计算这部分页的容量。
- continuous路由的RoPE保护使用本地K/G/private窗口上界，不把bounded历史逻辑位置
  误当成RoPE位置。历史仍保留原64位所有权/范围检查。
- NVMe临时arena和overlay改用原子唯一文件名，并在预分配前unlink。修复同目录
  多会话构造器在`open(O_TRUNC)`到`unlink`之间共享同一inode的竞争，durable
  archive路径保持原命名。CPU测试覆盖16个同时创建的owner及同名已有文件保留。

新增CPU资源测试验证总tier额度、GPU页数、向上取整和溢出拒绝。冻结构建排在现有
`concurrency_replenish01`性能A/B之后取得实验锁，避免编译负载混入计时。
`bounded_concurrency01`随后先跑临时请求刷新/取消、驻留复用的C4 GPU一致性，再跑
K128K/G64K的四个独立会话（物理repository为262144 tokens，CPU总tier为64GiB）。
短fixture的K/G只用于功能检查。长阶段是首次可运行性验证，不是bounded前后性能
对照，也未覆盖历史超过physical repository后的GC压力。

`bounded_http_build01`使用相同冻结core编译HTTP入口，尚未启动或替换服务。
跨请求全局I/O调度、实际GC压力、私有query内取消、HTTP实测及独立重复仍未完成。

冻结CUDA链接构建现已完成，3个受影响core单元和3个模型探针重编译；NVMe tier
CPU检查（含16个并发文件owner）与资源额度检查通过。HTTP二进制也完成编译链接，
未部署。`bounded_concurrency01`已启动`bounded_ephemeral_refresh` GPU阶段；其余
GPU阶段必须等待该检查通过，当前不声称bounded并发已经通过正确性验收。

后续GPU进展：`bounded_ephemeral_refresh`的16条刷新/取消响应、
`bounded_resident`的16条驻留/取消响应均通过固定种子的重复输出检查；前者的日志
确认query让出与同行ragged验证同时发生，后者确认实际prefix HIT。
这是32条GPU功能检查，不替代长历史和不同并发顺序下的状态隔离对照。
`bounded_long`已开始真实K128K/G64K C4独立会话回放，日志确认每owner CPU tier
13743895347 bytes，共至多5个owner，且每请求预留12800个GPU页。

`bounded_controller_reference01`已排队等待C4成功，保持同样的max_active=4、GPU
arena、CPU配额、repository和算术，但单一客户端按会话顺序逐条提交，检查完整输出
能否与四个独立stream一致。它用于隔离/批宽正确性，不是新一轮C1/C2/C4效率矩阵。

冻结短探针旧汇总字段只反映parity切换，而没有反映外部环境设定，因此部分
cooperative字段显示0。逐次query-yield/prefill日志和manifest环境记录为实际路径
证据；工作区已统一改为读取实际开关，未替换运行中的冻结二进制。
