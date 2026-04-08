# Mooncake Store 多租户隔离在 Serving 框架中的生产接入调研与方案

## 1. 目的

本文档聚焦一个非常具体的问题：

> 在生产环境中，SGLang、vLLM、TensorRT-LLM、LMCache 风格系统怎样才能把 Mooncake 的多租户隔离能力真正用起来？

这里的重点不是重新定义 Mooncake Store 内部的 metadata schema，也不是展开完整的 quota / fairness / bandwidth scheduling 算法，而是回答以下问题：

- 现有 serving 框架在生产里通常如何表达缓存隔离与共享边界
- 哪些做法需要侵入框架，哪些做法可以低侵入甚至不侵入框架
- Mooncake 如果希望在生产环境中被 SGLang / vLLM 等系统使用，推荐采用什么接入路径
- Mooncake 的 `tenant/domain`、`sharing_scope/cache_salt`、`qos_tier` 应如何映射到这些框架的现有能力中
- 在不同接入方式下，Mooncake 能得到什么级别的隔离能力，又会失去什么能力

## 2. 背景

Mooncake 当前的多租户隔离设计已经明确了两层语义：

- 复用隔离：`sharing_scope` / `cache_salt`
- 治理隔离：`tenant_id` / `domain_id` / `qos_tier`

但在真实生产环境中，Mooncake 往往不会直接面对“抽象客户端”，而是被嵌入或对接到：

- vLLM
- SGLang / HiCache
- TensorRT-LLM
- LMCache 风格控制器
- 自研网关、router、prefill/decode 编排层

因此，一个关键现实问题是：

> 即使 Mooncake 在 Store 内部已经支持 tenant-aware metadata，如果上层框架根本没有把这些语义传下来，那么生产里仍然很难真正使用这些能力。

这意味着，除了 Store 内部设计以外，还需要一套面向生产集成的“低侵入接入方案”。

## 3. 评估标准

为了比较不同系统的生产接入方式，本文使用以下几个维度：

1. **框架侵入程度**
   - 是否需要修改框架核心调度/缓存逻辑
   - 是否只需要增加请求字段或部署配置

2. **隔离语义强度**
   - 只能表达 reuse isolation
   - 还是能表达 tenant/domain 级治理边界

3. **控制面依赖程度**
   - 是否依赖独立 controller / namespace / instance-group / policy center

4. **与 Mooncake 的对齐程度**
   - 能否自然映射到 `sharing_scope`
   - 能否进一步映射到 `tenant/domain/qos_tier`

## 4. 调研对象在生产中的典型接入方式

## 4.1 vLLM：最典型的是 request-level `cache_salt`

### 已有生产模式

vLLM 当前最成熟的隔离机制，本质上仍是 `cache_salt`。

它的生产特征是：

- `cache_salt` 作为 request-level metadata 进入请求
- `cache_salt` 被纳入 KV block hash
- 只有 salt 相同的请求才能复用对应 KV cache

### 优点

- 对生产接入非常友好
- 基本不要求重构框架核心逻辑
- 只需在请求入口或网关层补充 request metadata
- 适合快速解决“不同 tenant 之间不要错误复用”问题

### 局限

- 它解决的是 reuse isolation
- 不自然表达对象 ownership
- 不能直接提供 per-tenant quota / fairness / lifecycle governance

### 对 Mooncake 的启发

如果希望 Mooncake 先被 vLLM 类系统低侵入接入，最自然的第一步是：

- 对外兼容 `cache_salt`
- 内部映射到 `sharing_scope`
- 将 `tenant/domain/qos_tier` 保留为可选增强项，而不是第一天就要求 vLLM 全量理解

## 4.2 TensorRT-LLM：request-level policy surface 更丰富，但仍属于低侵入扩展

### 已有生产模式

TensorRT-LLM 相比 vLLM，除了 `cache_salt` 以外，还支持：

- retention priority
- duration
- request-level cache policy hint

这类能力的典型生产接入方式是：

- 在请求对象或 executor API 中附带 cache policy metadata
- 后端 cache 子系统按这些 hint 调整保留与驱逐策略

### 优点

- 不一定要深改框架核心架构
- 只需扩展请求参数或执行器接口
- 比纯 salt 更容易承载更丰富的 cache policy surface

### 局限

- 仍然更偏 request-level cache policy
- 不等于 tenant-native object governance
- 不能自动提供 tenant/domain 资源树

### 对 Mooncake 的启发

Mooncake 的生产 API 可以设计成分层模式：

- 最小接入：只传 `cache_salt/sharing_scope`
- 增强接入：再传 `tenant_id/domain_id`
- 高级接入：再传 `qos_tier` 或 retention hint

这样可以减少对现有 serving 框架的改动压力。

## 4.3 SGLang / HiCache：更偏架构层与层次缓存边界

### 已有生产模式

HiCache 的核心边界是：

- L1 GPU cache：私有
- L2 host cache：私有
- L3 distributed cache：集群共享

它在生产里最典型的接入方式不是要求每个请求携带完整租户治理信息，而是：

- 先通过层次缓存架构划定 private/shared 边界
- 再通过 L3 backend 的能力承担跨实例共享

### 优点

- 架构边界清晰
- 与现有 serving runtime 的耦合较弱
- 容易通过 backend 接口替换不同 L3 系统

### 局限

- 它本身不提供 object-level tenant governance
- 如果 L3 backend 没有 tenant-aware metadata，shared L3 仍可能缺少治理边界

### 对 Mooncake 的启发

如果 Mooncake 接入 SGLang，低侵入方式更可能是：

- 保持 HiCache 的 private L1/L2 + shared L3 模型不变
- 在接入 Mooncake L3 backend 时，按 deployment / instance / model-group / gateway identity 绑定默认 tenant/domain
- 必要时再在 request path 上补充 `sharing_scope`

换句话说，SGLang 的最佳接入点更像是：

> L3 backend integration point，而不是 request scheduler 核心。

## 4.4 LMCache：更偏 controller / deployment 驱动

### 已有生产模式

LMCache 的公开资料更强调：

- controller
- instance / worker / chunk 管理
- 分布式 metadata 组织
- 生产部署与 K8s 编排

这类系统在生产里的典型接入方式通常是：

- 通过 controller / orchestrator 管理实例和缓存归属
- 应用本身不一定直接感知完整治理模型

### 优点

- 容易在控制面注入策略
- 适合多实例部署和中心化编排
- 框架侵入通常小于“改请求执行器核心”

### 局限

- 公开资料里并没有很强的一等公民 tenant/domain 模型
- 更适合 instance/controller 级别治理，而不是完整 object-native 多租户治理

### 对 Mooncake 的启发

Mooncake 如果未来引入更完整的控制面，那么：

- tenant/domain 绑定
- policy 下发
- rollout / override
- quota / fairness 配置

都可以优先落在控制面，而不是强迫上层每个框架都理解全部语义。

## 4.5 阿里 Tair KVCache Manager：实例 / 实例组边界优先

### 已有生产模式

其公开模式更接近：

- `instance_id`：可见性边界
- `instance_group`：容量和策略边界
- quota / reclaim policy：控制面配置

### 优点

- 非常符合生产部署与运维方式
- 不要求请求级传递复杂治理语义
- 上层应用只需要接入某个实例或实例组

### 局限

- 粒度更偏实例级
- 不适合在同一个 serving instance 内部做更细粒度的 tenant/domain 治理

### 对 Mooncake 的启发

这是“不侵入框架”的强参考：

- 可以把隔离边界前置到 namespace / tenant endpoint / deployment unit
- serving 框架只知道连哪个 Mooncake namespace/endpoint
- 更细的 quota / fairness / policy 在 Mooncake 控制面完成

## 4.6 火山引擎 / 字节 EIC：Namespace + QoS 是最接近共享基础设施生产治理的模式

### 已有生产模式

从公开资料看，EIC 的核心原语是：

- `Namespace`
- per-namespace quota
- per-namespace IOPS / 带宽 QoS
- 队列/调度机制抑制 noisy neighbor

### 优点

- 显式管理边界清晰
- 资源治理直接落在共享基础设施层
- 上层框架不一定需要知道完整治理细节，只需落到某个 Namespace
- 很接近“低侵入 + 生产可运营”的共享基础设施模型

### 局限

- Namespace 模型仍然偏粗粒度
- 公开资料不足以说明它天然支持 object-native tenant/domain hierarchy

### 对 Mooncake 的启发

EIC 是 Mooncake 生产接入方式最重要的参考之一：

> 不要求上层框架先理解全部治理语义，而是要求它至少落到一个显式 namespace，再由共享基础设施完成 quota / QoS / fairness。

## 4.7 Agent Bucket / ObjectSet 模型的启发

字节跳动在 Agent Bucket 设计中提出的关键创新，不是简单重命名 Bucket，而是在传统 `Bucket/Object` 之间引入一个新的原生资源层级：`ObjectSet`。

也就是说，其逻辑模型不再只是：

- Bucket
- Object

而是扩展为：

- Bucket
- ObjectSet
- Object

这带来的核心变化是：多租户系统中真正需要的高级治理能力，不再只能停留在 bucket 级别，也不需要再退化成依赖前缀约定的“伪层次”，而是可以落在一个可规模化、可运营、可独立治理的中间资源层上。

### 4.7.1 Agent Bucket 的核心价值

从文中描述看，ObjectSet 承担的不是简单目录语义，而是一个真正的原生治理单元。它天然适合承载：

- 权限边界
- 容量配额
- 带宽与 QPS 限制
- 生命周期策略
- 监控与计量
- 计费与成本分摊
- 独立接入点与差异化加速/隔离策略

换句话说，在 Agent Bucket 模型里：

> ObjectSet 才是面向海量终端用户的基本治理单位。

这与传统对象存储中“Bucket 太粗、Object 太细、Prefix 只是命名技巧”的困境形成了鲜明对比。

### 4.7.2 对 Mooncake 的直接启发

这对 Mooncake 的启发非常直接：Mooncake 当前虽然已经明确了：

- `tenant/domain` 作为治理隔离边界
- `sharing_scope` 作为复用隔离边界
- `group/generation` 作为生命周期与批量管理边界

但这几个边界之间，仍然缺少一个更适合承载生产运营语义的“对象集合容器层”。

Agent Bucket 的设计说明，生产系统往往不仅需要回答：

- 对象属于谁
- 对象能不能共享
- 对象如何批量清理

还需要回答：

- 哪一组对象应该作为一个整体被治理
- 哪一组对象共享同一套 quota / monitoring / access / bandwidth policy
- 哪一组对象应成为生产接入、资源切片和成本归因的稳定单元

这意味着，Mooncake 后续可能需要显式承认一种 ObjectSet-like 的中间抽象。

### 4.7.3 它和 `group/generation` 的关系

ObjectSet-like 抽象与 `group/generation` 有相似之处，但两者并不等价。

`group/generation` 更偏：

- workflow 级组织
- batch remove
- generation trim
- 生命周期管理

而 Agent Bucket 中的 ObjectSet 更偏：

- 运营与治理边界
- 配额与计量边界
- 独立接入与权限边界
- 水平扩展与切片边界

因此，更合理的理解是：

- `tenant/domain` 负责 ownership / governance root
- ObjectSet-like 容器负责面向生产的对象集合治理单元
- `group/generation` 负责更轻量的生命周期组织关系
- `sharing_scope` 负责安全复用边界

### 4.7.4 Set Tagging / Set Slice / Set AccessPoint 的进一步启发

Agent Bucket 文中最值得 Mooncake 继续吸收的，不只是 ObjectSet 本身，还包括围绕 ObjectSet 展开的三类能力：

#### Set Tagging

通过 tag/profile 统一定义一批 ObjectSet 的策略，例如：

- 容量上限
- 公网带宽
- 单流速度
- 附加增值能力

这启发 Mooncake 后续不要只停留在 per-tenant / per-domain 的硬编码策略，还可以考虑：

- policy class
- qos profile
- tagging 驱动的批量治理

#### Set Slice

Set Slice 的思想是：

- 逻辑上保持一个统一的 ObjectSet
- 物理上将其数据和元数据切分到多个 slice / cluster
- 实现“逻辑不拆、物理拆分”

这对 Mooncake 的启发很强：未来如果某一类对象集合规模非常大，Mooncake 也需要考虑：

- 如何保持对象集合治理边界稳定
- 同时允许底层 placement / metadata / query / reclaim 水平扩展
- 让上层看到的仍是一个逻辑统一容器

#### Set AccessPoint

ObjectSet 级独立接入点进一步说明：

- 生产接入不一定要从 per-object metadata 开始
- 也可以从 container-level access boundary 开始
- 接入点、权限与隔离可以先在容器层落地，再由底层系统展开到对象层

这与 Mooncake 当前正在考虑的 namespace / endpoint / adapter / sidecar 路线高度一致。

### 4.7.5 对 Mooncake 生产接入方案的意义

ObjectSet-like 抽象最现实的价值之一，是它可能成为 Mooncake 低侵入接入 serving 框架的最佳桥梁。

也就是说，未来上层 SGLang / vLLM / Agent SDK 不一定需要一开始就显式传递完整的：

- `tenant_id`
- `domain_id`
- `qos_tier`

它们也可以只先落到一个更高层的 container，例如：

- object_set_id
- workspace_id
- agent_bucket_id
- session_bucket

然后由 Mooncake 的 control plane / gateway / adapter 根据这个容器级标识，自动补齐：

- `tenant_id`
- `domain_id`
- `sharing_scope`
- `qos_tier`
- 未来的 bandwidth / traffic policy

这种方式有助于显著降低上层框架的侵入成本，同时保留 Mooncake 后续演进为 tenant-aware shared infrastructure 的空间。

### 4.7.6 当前阶段的设计建议

基于上述启发，Mooncake 当前阶段更合理的策略不是继续把 `object_set` 仅仅视为接入层近似概念，而是应正式将其纳入设计语义中，作为 `tenant/domain` 之下、`object` 之上的独立抽象层。

推荐的职责划分是：

- `tenant`：最外层 ownership / quota root / governance root
- `domain`：tenant 内业务边界
- `object_set`：面向生产工作负载的对象集合治理单元
- `object`：具体对象元数据与 placement 单元

同时继续保留：

- `sharing_scope`：复用隔离边界，而不是资源层级
- `group/generation`：生命周期与批量管理边界，而不是治理容器

这意味着，Mooncake 当前阶段应先在设计上明确承认：

- 除了 `tenant/domain` 这种 ownership 边界
- 还需要一个面向生产工作负载的对象集合容器边界
- 该边界应成为后续 quota / accounting / monitoring / access / bandwidth policy 的重要落点
- 在未显式配置 QoS/policy override 时，不同 tenant 的默认治理优先级应相同
- 因此在驱逐、带宽竞争等资源竞争场景中，跨 tenant 默认按平等策略处理，而不是隐式推导主次优先级
- 在未显式提供 `tenant/domain/object_set` 时，系统应自动补齐到预定义的 default 容器，以保证兼容性与治理闭环

### 4.7.7 推荐架构图

下面给出一个推荐的分层架构图，用于说明 `object_set` 在 Mooncake 中的定位：

```text
Serving Framework / Gateway / Adapter
    |
    |  (deployment binding / request metadata / cache_salt / workspace)
    v
+-------------------------------------------------------------+
| Mooncake Control Plane / Integration Layer                  |
|-------------------------------------------------------------|
| - tenant / domain binding                                   |
| - object_set mapping                                        |
| - policy profile / tagging                                  |
| - access / namespace / endpoint policy                      |
+-------------------------------------------------------------+
    |
    v
+-------------------------------------------------------------+
| Mooncake Store Metadata Plane                               |
|-------------------------------------------------------------|
| tenant                                                      |
|  └─ domain                                                  |
|      └─ object_set                                          |
|          ├─ object                                           |
|          ├─ object                                           |
|          └─ object                                           |
|                                                             |
| object_set is the primary unit for:                         |
| - accounting / quota / monitoring                           |
| - access policy / traffic policy                            |
| - serving integration binding                               |
|                                                             |
| object remains the primary unit for:                        |
| - placement / replica / transfer / lifecycle state          |
+-------------------------------------------------------------+
    |
    v
+-------------------------------------------------------------+
| Storage / Transfer / Background Execution                   |
|-------------------------------------------------------------|
| - placement / replicas                                      |
| - reclaim / offload / prefetch                              |
| - transfer scheduling / future bandwidth policy             |
+-------------------------------------------------------------+
```

如果从对象语义角度进一步压缩，可以把 Mooncake 的核心分层总结为：

```text
tenant
  └─ domain
      └─ object_set
          └─ object

sharing_scope   = reuse boundary
group/generation = lifecycle / batch-management boundary
```

这个结构的关键好处是：

- `tenant/domain` 不会因为承载过多运营语义而变得过粗
- `group/generation` 不会被误用成 quota / policy / access 的主容器
- `object_set` 可以成为生产接入和资源治理之间的桥梁
- 上层框架可以先绑定到 `object_set`，再由 Mooncake 展开更完整的 metadata 和 policy

## 5. 结论：哪些做法更低侵入

从上述调研对象看，生产里真正常见且可行的“低侵入接入”大致分成三类。

### 第一类：实例 / Namespace 级接入

表现形式：

- 实例级 endpoint
- namespace 级 endpoint
- deployment 级默认租户绑定

特点：

- 框架只需要配置连接哪个 backend/namespace
- 最不要求改框架核心逻辑
- 最适合一实例一租户、一模型服务一业务线的场景

缺点：

- 粒度较粗
- 不适合同一实例混跑多个 tenant 且仍希望细粒度治理

### 第二类：request-level isolation metadata 接入

表现形式：

- `cache_salt`
- `sharing_scope`
- request-level cache policy hint

特点：

- 对 vLLM / TensorRT-LLM 风格系统最自然
- 通常只需要请求入口或网关层注入字段
- 可以低侵入解决 reuse isolation

缺点：

- 主要解决“能不能共享”
- 不能独立解决完整治理问题

### 第三类：控制面 / adapter / sidecar 注入

表现形式：

- gateway 根据身份和 deployment config 生成 tenant/domain
- sidecar 或 adapter 补齐 metadata
- controller 下发 policy

特点：

- 上层框架可少改甚至不改
- Mooncake 仍能拿到较丰富的治理语义
- 更适合多框架并存的生产环境

缺点：

- 系统组件更多
- 需要额外设计映射规则与运维方式

## 6. Mooncake 的推荐接入策略

结合调研与 Mooncake 当前设计目标，推荐分三层推进。

需要明确的是：当前 `main` 分支已经落地的是 metadata plane foundations——即 `tenant/domain/object_set/sharing_scope/qos_tier/logical_key/canonical_key` 这类字段在 Store 主路径、snapshot、HA payload、standby bootstrap 中的存储与透传能力。下面这些接入层、控制面、quota/fairness/bandwidth policy 能力仍属于推荐路线，而不是当前代码已默认具备的现成能力。

## 6.1 第一阶段：Namespace / deployment 级接入

目标：

- 先让生产环境能稳定用起来
- 不要求框架立刻理解完整 tenant/domain 语义

推荐做法：

- 后续为 Mooncake 提供 deployment-level 或 namespace-level 绑定能力
- 通过配置将某个 serving deployment 映射到：
  - 默认 `tenant_id`
  - 默认 `domain_id`
  - 默认 `object_set`
  - 默认 `qos_tier`
  - 可选默认 `sharing_scope`

这相当于：

- SGLang / vLLM 只需要“连对地方”
- Mooncake 在 backend / control-plane 层获得最低限度的治理边界
- 如果上层没有显式传入 `tenant/domain/object_set`，也可以先由 deployment binding 或系统 fallback 自动补齐到 default 容器

适用场景：

- 一实例一租户
- 一业务线一 deployment
- 先做生产 rollout

## 6.2 第二阶段：兼容 request-level `cache_salt` / `sharing_scope`

目标：

- 在多租户混跑场景中，优先解决 reuse isolation

推荐做法：

- 后续对外 API 兼容 `cache_salt`
- 后续在接入层或 client side 将其统一映射为 `sharing_scope`
- 再由 Mooncake client / gateway / adapter 负责将该字段写入 object metadata

这样做的好处是：

- vLLM / TensorRT-LLM 风格系统接入成本最低
- 可以快速对齐业界已有生产模式
- 不需要第一天就要求框架传 tenant/domain/qos 全量信息

## 6.3 第三阶段：adapter / sidecar 补齐完整治理语义

目标：

- 在保持框架低侵入的前提下，逐步获得完整治理能力
- 在默认情况下保持跨 tenant 平权，只有显式 policy/QoS 配置才打破默认平等

推荐做法：

- 后续在 Mooncake 前面增加接入层（gateway / sidecar / adapter）
- 由该接入层根据以下来源补齐 metadata：
  - 请求身份
  - 部署配置
  - Namespace
  - Route / model group
  - workspace / object_set 绑定
  - request-level `cache_salt`

可生成或推导：

- `tenant_id`
- `domain_id`
- `object_set`
- `sharing_scope`
- `qos_tier`
- 未来的 traffic class

这样做的价值在于：

- vLLM / SGLang 不必直接理解 Mooncake 全部治理语义
- Mooncake 侧仍能逐步实现 tenant-aware accounting、quota、fairness
- 且这些治理能力在默认情况下应以跨 tenant 平权为基线，再由显式 QoS/policy 做差异化
- 对老客户端或未接入治理语义的流量，也能通过 `default tenant/domain/object_set` fallback 保持行为闭环

## 7. Mooncake 应避免的接入方式

当前阶段不建议直接采取以下做法：

1. 强迫所有上层框架立即理解完整的 `tenant/domain/qos_tier` 语义
2. 一开始就要求 serving 框架深改内部 scheduler / cache manager
3. 一开始就把 `canonical_key` 暴露为上层请求主键
4. 在没有 metadata/control-plane 闭环前，先推动 RDMA bandwidth execution 级接入

原因很简单：

- adoption 成本太高
- rollout 风险大
- 很难在多框架并存的生产环境中统一推进

## 8. 对当前任务拆解的影响

如果将“生产接入”视为整个多租户隔离特性的组成部分，那么它不应只是附属讨论，而应成为独立工作流。

推荐将相关工作拆成以下子任务：

1. 定义 Mooncake 对外最小接入语义：`cache_salt/sharing_scope`
2. 定义 deployment / namespace 级默认 `tenant/domain/object_set` 绑定
3. 定义未传 metadata 时的 `default tenant/domain/object_set` fallback 规则
4. 设计 gateway / adapter 注入完整 metadata 的方式
5. 明确 `object_set` 作为 serving workload container 的映射规则
6. 明确 SGLang L3 backend 接入路径
7. 明确 vLLM request-level metadata 接入路径
8. 为未来的 traffic class / transfer policy 预留接口

## 9. 最终建议

Mooncake 若希望在生产里真正被 SGLang / vLLM / 其他 serving 系统使用，推荐遵循以下原则：

- **先低侵入，再增强**
- **先兼容 `cache_salt`，再补齐 tenant/domain**
- **先 deployment/namespace 级接入，再做 request-level enrich**
- **优先通过控制面和 adapter 承接复杂治理语义，而不是强迫所有框架重写核心逻辑**

更具体地说：

- 对 vLLM / TensorRT-LLM 风格系统，优先走 `cache_salt -> sharing_scope` 路线
- 对 SGLang / HiCache 风格系统，优先走 L3 backend integration + deployment binding 路线
- 对更偏平台化的部署场景，优先走 namespace / instance-group / controller 路线

这条路径既符合业界已验证的生产模式，也更符合 Mooncake 当前“metadata first, behavior second”的演进路线。

## 10. 参考资料

- vLLM Automatic Prefix Caching
- TensorRT-LLM KV Cache 文档
- SGLang HiCache 系统设计文档
- LMCache 架构与生产部署文档
- Alibaba Tair KVCache Manager 架构文章
- Volcengine / ByteDance EIC 公开技术文章
