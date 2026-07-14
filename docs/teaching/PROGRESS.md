# 学习记录

## 总览

- 学习目标：整体通读 InternNav，理解架构、基线、InternVLA-N1 及部署链路，达到能独立阅读和修改代码的深度。
- 总章节数：6
- 已完成：3

---

## 章节进度

| 章节 | 状态 | 完成日期 | 掌握度 | 疑难点 / 待复习 | 问答收获 |
|------|------|----------|--------|------------------|----------|
| 01 全局架构与注册机制 | ✅ 已掌握 | 2026-06-29 | 良好 | 无 | 理解了注册机制与 5 个 Agent 的用途 |
| 02 配置系统与入口脚本 | ✅ 已掌握 | 2026-07-01 | 良好 | 无 | 理解了 Pydantic 配置、三大入口与默认配置填充 |
| 03 基线模型 | ✅ 已掌握 | 2026-07-01 | 良好 | 无 | 理解了 Seq2Seq/CMA/RDP/NavDP 的原理与差异 |
| 04 InternVLA-N1 双系统模型 | ✅ 已掌握 | 2026-07-14 | 良好 | 无 | 掌握双系统架构、S2/S1 协作、训练/评测/部署/数据格式 |
| 05 环境封装与评测链路 | 🟡 进行中 | — | — | — | — |
| 06 训练、部署与工程实践 | ⬜ 未开始 | — | — | — | — |

---

## 疑难点台账

- [x] 无待消化疑难点
- [ ] 待学习过程中补充

---

## 每日学习记录

### 第 1 天（2026-06-29）

- **学习主题**：项目概览与学习计划制定
- **对应计划条目**：第一阶段第 1 天
- **完成情况**：
  - [x] 阅读 `README.md`
  - [x] 阅读 `AGENTS.md` 第 1–5 章
  - [x] 生成教学总览、学习计划、学习记录、章节大纲等文档
- **关键收获**：
  - 了解了 InternNav 支持 VLN-CE、VLN-PE、VN 及 dual-system VLN。
  - 理解了仓库顶层结构：`internnav/` 核心包、`scripts/` 脚本、`tests/` 测试。
  - 明确了四大注册机制：Agent、Env、Evaluator、Policy。
- **遇到的问题 / 疑问**：无
- **下一步计划**：完成环境安装与子模块初始化

---

### 第 2 天（2026-07-01）

- **学习主题**：第 1 章 · 全局架构与注册机制 + 第 2 章 · 配置系统与入口脚本
- **对应计划条目**：第一阶段第 3 天、第 4–5 天
- **完成情况**：
  - [x] 理解 Agent / Env / Evaluator 三类注册表
  - [x] 理解 Policy 工厂 `get_policy()` / `get_config()`
  - [x] 走读 `scripts/eval/eval.py` 完整数据流
  - [x] 掌握 5 个已注册 Agent 的用途与区别
  - [x] 理解 Pydantic 配置系统与三大入口脚本
  - [x] 理解 `vln_default_config.py` 的默认配置填充逻辑
- **关键收获**：
  - Agent、Env、Evaluator 都用类级字典 + 装饰器注册；Policy 用字符串工厂按需导入。
  - `Evaluator.__init__` 默认创建 `AgentClient`，支持远程推理服务。
  - `cma`/`seq2seq`/`rdp` 解决离散 VLN；`internvla_n1` 是双系统导航大模型；`dialog` 用于多轮对话导航。
  - 训练入口用 `tyro` 解析命令行；评测入口动态加载 Python 配置文件。
- **遇到的问题 / 疑问**：无
- **下一步计划**：进入第 3 章：基线模型

---

### 第 3 天（2026-07-01）

- **学习主题**：第 3 章 · 基线模型（开讲）
- **对应计划条目**：第二阶段第 8–11 天
- **完成情况**：
  - [x] 理解 Seq2Seq 的序列到序列设计
  - [x] 理解 CMA 的跨模态注意力
  - [x] 理解 RDP 的连续轨迹 + 离散动作解析
  - [x] 理解 NavDP 的目标条件视觉导航
- **关键收获**：
  - Seq2Seq 和 CMA 解决同一类离散 VLN，区别在是否显式 cross-attention。
  - RDP 把离散动作问题转化为连续轨迹生成，再用后处理映射回离散动作。
  - NavDP 处理无语言指令的视觉导航，可作为 InternVLA-N1 的 System 1。
  - 项目当前 checkpoint：DepthAnythingV2（95MB）+ InternVLA-N1 DualVLN（16GB）；基线 checkpoint 需自行训练。
- **遇到的问题 / 疑问**：
  - 已讨论：为什么有 InternVLA-N1 还需要基线？答：基线用于对照、轻量部署和学习阶梯。
- **下一步计划**：完成第 3 章自测，进入第 4 章

---

### 第 4 天（2026-07-06）

- **学习主题**：第 3 章收尾 + 问答讨论
- **对应计划条目**：第二阶段第 8–11 天
- **完成情况**：
  - [x] 完成第 3 章 4 道自测题
  - [x] 深入理解 Seq2SeqNet 与 CMANet 的 Policy 差异
  - [x] 讨论 GRU 与 Transformer 在导航中的取舍
  - [x] 梳理项目 checkpoint 现状与基线模型价值
- **关键收获**：
  - Policy 是 Agent 内部的神经网络模型，Seq2SeqNet 直接 concat 特征过 GRU，CMANet 用显式 cross-attention。
  - GRU 适合实时逐步控制，Transformer 适合高层理解；InternVLA-N1 的 System 2 用 Transformer，System 1 可用 Transformer-based diffusion。
  - 基线模型是 InternVLA-N1 的对照组、轻量替代方案和学习阶梯。
  - 项目当前 checkpoint 只有 DepthAnythingV2（95MB）和 InternVLA-N1 DualVLN（16GB）；基线需自行训练。
- **遇到的问题 / 疑问**：无
- **下一步计划**：准备进入第 4 章 · InternVLA-N1 双系统模型

---

### 第 5 天（2026-07-08）

- **学习主题**：第 4 章 · InternVLA-N1 双系统模型（4.1–4.6）
- **对应计划条目**：第四阶段第 22 天
- **完成情况**：
  - [x] 理解双系统整体架构（System 2 + System 1）
  - [x] 理解 System 2 的输入输出与 prompt 设计
  - [x] 理解 System 1 的 nextdit / navdp 实现
  - [x] 理解 `InternVLAN1Net` Policy 包装层
  - [x] 理解 `InternVLAN1Agent` 的 `sync` / `partial_async` 运行模式
  - [x] 理解两阶段训练流程（先训 System 2，再冻结 VLM 训 System 1）
  - [x] 通过问答深入理解真实部署与仿真评测的 Agent 差异
- **关键收获**：
  - InternVLA-N1 = Qwen2.5-VL（System 2 规划）+ nextdit/navdp（System 1 控制）。
  - System 2 输出 pixel goal / latent plan / 离散动作；System 1 生成连续轨迹。
  - `s1_step_latent()` 内部生成连续轨迹，但仿真评测路径会解析为离散动作。
  - `partial_async` 把慢速 S2 推理放到后台线程，主线程快速执行 S1。
  - 训练分阶段：先 `train_system2.sh` 训 S2，再 `train_dual_system.sh` 冻结 VLM 训 S1。
  - 仿真评测用 `InternVLAN1Agent` + `InternVLAN1Net`；真实部署用 `InternVLAN1AsyncAgent`。
- **遇到的问题 / 疑问**：无
- **下一步计划**：明天继续第 4 章 4.7 节（评测配置），后续依次讲解 4.8 真实部署、4.9 数据格式

---

### 第 6 天（2026-07-14）

- **学习主题**：第 4 章收尾 · InternVLA-N1 评测/部署/数据格式 + 自测 + 环境链路预习
- **对应计划条目**：第四阶段第 22–24 天、第五阶段第 25 天
- **完成情况**：
  - [x] 完成 4.7 评测配置教学
  - [x] 完成 4.8 真实世界部署教学
  - [x] 完成 4.9 数据格式 v0.5 教学
  - [x] 完成第 4 章 8 道自测题并逐题讲解
  - [x] 通过问答深入理解 `partial_async`、`control_thread`、episode/batch 数据量
  - [x] 开始第 5 章环境封装与评测链路概览（已暂停，待后续继续）
- **关键收获**：
  - 仿真评测默认用 `partial_async`，真实部署 `InternVLAN1AsyncAgent` 也按 `PLAN_STEP_GAP` 异步触发 S2。
  - `control_thread` 从 ROS2 client 启动后一直 10Hz 运行，有目标时才发布 `/cmd_vel_bridge`。
  - `robot_flash=True` 直接设置 world pose；`one_step_stand_still=True` 保证 dual system 图像稳定。
  - 一个 episode 原始数据几十到几百 MB；训练时 `batch_size=2` 指 2 个子轨迹 sample。
  - LeRobot v0.5 用 `task` 替代 `instruction_text`，在 `internvla_n1_lerobot_dataset.py:770` 读取。
  - 第 5 章三种环境（Habitat/InternUtopia/RealWorld）和四种评测器初步了解。
- **遇到的问题 / 疑问**：无
- **下一步计划**：暂停第 5 章，后续按学习节奏继续

---

## 本次会话问答摘要

### 日期：2026-07-01

**问题 1**：`AgentCfg.model_name` 和 `EnvCfg.env_type` 的值从哪里来，分别对应什么？

**回答**：两者都来自用户配置。`model_name` 对应 `Agent.agents` 的 key（如 `'rdp'`、`'internvla_n1'`）；`env_type` 对应 `Env.envs` 的 key（如 `'internutopia'`、`'habitat'`）。

**问题 2**：`scripts/eval/eval.py` 为什么要动态加载 Python 配置文件，而不是 YAML/JSON？

**回答**：因为配置是嵌套的 Pydantic 对象，且需要引用项目内的 Python 常量（如 `rdp_cfg`）。Python 配置比 YAML/JSON 更灵活、类型更安全。

**问题 3**：`vln_default_config.py` 的作用是什么？

**回答**：为 `vln_distributed` 评测自动填充机器人、传感器、控制器、场景、模型默认参数等复杂配置，减少用户重复编写。

**问题 4**：`scripts/eval/start_server.py` 里为什么要 `from internnav.agent import Agent`？

**回答**：为了触发注册副作用，import 该模块会执行所有 `@Agent.register(...)` 装饰器，把 agent 类注册到 `Agent.agents`。删掉会导致 server 找不到 agent。

**问题 5**：`vln_default_config.py` 里有四个模型配置（cma/rdp/seq2seq/internvla_n1），是同时调用四个模型吗？

**回答**：不是，是四选一。`get_config()` 根据 `agent.model_name` 选择对应默认配置模板，一次评测只跑一个模型，加载一个 checkpoint。

**问题 6**：项目中已有 InternVLA-N1 checkpoint，为什么还需要 CMA / Seq2Seq / RDP？

**回答**：基线用于：① 学术研究对照；② 更小更快更易部署；③ 不同环境适配；④ 学习阶梯，帮助理解核心问题。

**额外讨论**：发现 `assets/realworld_sample_data1/` 和 `assets/realworld_sample_data2/` 是生成的调试图片，应在 `.gitignore` 中忽略；已新增 `assets/realworld_sample_data*/` 和 `model/` 规则并提交推送。

**收获**：巩固了注册机制、配置驱动设计、入口脚本与模型选择逻辑，理解了基线与旗舰模型的关系，完成了一次 git 提交/推送实践。

---

### 日期：2026-07-06

**问题 1**：Seq2SeqAgent 和 CmaAgent 内部的 policy 类有哪些差异？这里的 policy 具体是什么？

**回答**：Policy 是 Agent 内部的神经网络模型。Seq2SeqAgent 加载 `Seq2Seq_Policy`（类 `Seq2SeqNet`），CmaAgent 加载 `CMA_Policy`（类 `CMANet`）。两者对外接口相同，但内部融合方式不同：Seq2SeqNet 把语言/RGB/depth 特征直接 concat 后过 GRU；CMANet 保留视觉空间特征，通过显式 cross-attention 让 state→text、text→rgb、text→depth 互相关注，再经过第二 GRU。

**问题 2**：GRU 后续会被 Transformer 替代吗？

**回答**：Transformer 会替代 GRU 做高层理解和规划，但 GRU 在实时逐步控制层仍有优势（固定隐状态、低延迟、天然因果性）。InternVLA-N1 的 System 2 已用 Transformer（Qwen2.5-VL），System 1 用 Transformer-based diffusion。未来可能是 Mamba/Streaming Transformer 这类兼顾两者的架构。

**问题 3**：`vln_default_config.py` 都有哪些评测内容？

**回答**：包含 InternUtopia 物理参数、H1 机器人配置、传感器（pano_camera、topdown_camera、pointcloud）、控制器（move_by_speed、stand_still、move_by_discrete、move_by_flash）、场景缩放（mp3d/grscene/kujiale）、VLNPEMetric 指标、四种模型默认参数、分布式 Ray 配置、递归 None 校验。

**问题 4**：项目中当前有哪些 checkpoint？

**回答**：`checkpoints/depth_anything_v2_metric_hypersim_vits.pth`（95MB，NavDP 用）和 `model/InternVLA-N1-DualVLN/`（16GB，旗舰模型）。CMA/Seq2Seq/RDP 基线 checkpoint 需自行训练。

**问题 5**：为什么有 InternVLA-N1 的 checkpoint 还需要 CMA/Seq2Seq/RDP？

**回答**：基线用于学术研究对照、更小更快更易部署、适配不同环境、作为学习阶梯。InternVLA-N1 是旗舰方案，但不是所有场景都需要 16GB 大模型。

**收获**：深入理解了 Policy 与 Agent 的关系、Seq2SeqNet 与 CMANet 的架构差异、GRU 与 Transformer 的适用边界，以及项目 checkpoint 现状和基线价值。

---

### 日期：2026-07-08

**问题 1**：latent plan 和 pixel goal 是搭配使用的吗？

**回答**：不是必须搭配，而是二选一。pixel goal 是图像坐标，更直观；latent plan 是隐式向量，更抽象。具体用哪个由训练时的 `system1` 参数决定。

**问题 2**：`look_down` 是调用下视相机吗？但我们没有下视相机。

**回答**：`look_down` 不一定是单独相机，而是「相机以俯视角度拍摄」。仿真里通过调整相机俯仰角实现；真实世界里取决于硬件。当前真实部署代码只有一个前视相机，所以 `look_down=False`。

**问题 3**：请讲解 `s1_step_latent()` 的代码。

**回答**：`generate_traj()` 先生成连续轨迹 `dp_actions`，然后根据 `continuous_traj` 选择 `traj_to_actions()` 或 `chunk_token()` 把轨迹解析成离散动作，过滤 stop 后取前 4 个动作返回。这是为了和离散动作评测接口对齐。

**问题 4**：`internvla_n1_agent.py` 的 `def step(self, obs)` 是谁来调用的？

**回答**：`Agent.step(obs)` 是统一接口，由 `Evaluator`（本地/远程评测）或 `AgentServer` 调用。真实部署不用这个 Agent，而是用 `InternVLAN1AsyncAgent`。

**问题 5**：真实部署用的是 `internvla_n1_agent_realworld.py` 吗？

**回答**：是的。`scripts/realworld/http_internvla_server.py` 直接 import 并实例化 `InternVLAN1AsyncAgent`。它未注册到 `Agent.agents`，直接加载 `InternVLAN1ForCausalLM`，接口更适合帧-by-帧的真实部署。

**问题 6**：如果想在 ROS2 中跑起来，是在 ROS2 中搭建 wrapper 然后实例化 `InternVLAN1AsyncAgent` 吗？

**回答**：是的，有两种方案：① 直接在 ROS2 节点中实例化 `InternVLAN1AsyncAgent`，延迟最低但机器人需足够算力；② 沿用现有 HTTP Server + ROS2 Client 架构，模型放独立服务器，机器人端只做控制。

**收获**：理解了 InternVLA-N1 双系统的工作流程、System 1 的连续轨迹到离散动作的转换、仿真与真实部署的 Agent 差异，以及 ROS2 部署的可行架构。

---

### 日期：2026-07-14

**问题 1**：`control_thread` 是什么时候工作的？

**回答**：`control_thread` 在 `http_internvla_client.py` 启动时就开始运行（`__main__` 里 `start()`），是 10Hz 的无限循环。初始阶段没有 MPC/PID 目标时不发布速度；`planning_thread` 拿到 trajectory 后更新 MPC 参考轨迹并切到 `MPC_Mode`，或拿到 discrete_action 后更新 `homo_goal` 并切到 `PID_Mode`，之后 `control_thread` 每 100ms 解算并发布 `/cmd_vel_bridge`。它只在真实部署 ROS2 client 中存在，仿真模式没有。

**问题 2**：现在项目用的是 `partial_async` 吗？

**回答**：是的。仿真评测配置 `h1_internvla_n1_async_cfg.py` 中 `infer_mode='partial_async'`；真实部署 `InternVLAN1AsyncAgent` 也按 `PLAN_STEP_GAP` 间隔触发 System 2，两次 S2 之间复用 latent 给 System 1，本质同样是异步。

**问题 3**：目前一个 episode 数据量多大，一个 batch 多大？

**回答**：一个 episode 是一条导航轨迹，LeRobot 格式下包含 meta、parquet、多张 RGB/深度图，数据量通常几十到几百 MB。训练时 `NavPixelGoalDataset` 把 episode 按 `sample_step` 切成多个子轨迹 sample；`train_system2.sh` 中 `per_device_train_batch_size=2` 指每个 GPU 每步处理 2 个 sample。全局 batch size = `2 × num_gpus × gradient_accumulation_steps`。

**问题 4**：为什么第 5 章先不继续？

**回答**：用户希望先完成学习记录更新，第 5 章内容已做概览介绍，后续按节奏继续。

**收获**：巩固了 InternVLA-N1 的异步推理机制、真实部署控制线程时序、训练数据量级与 batch 概念，并梳理了三种环境/四种评测器的整体关系。

---

> 每章结束后更新本文件，并同步刷新 `docs/teaching/README.md` 顶部的"进度快照"。
