# InternNav 教学总览

> 本目录由「小k带我学习」skill 生成，用于系统学习 InternNav 项目。
> 学习原则：先全局，再细节，最后算法；一次一章，掌握后再推进。

---

## 学习目标

- **整体通读**：理解 InternNav 作为 PyTorch 具身导航工具箱的定位、支持任务与技术栈。
- **架构掌握**：掌握 Agent / Env / Evaluator / Policy 四大注册机制及其协作方式。
- **配置与入口**：能独立修改 Pydantic 配置，跑通训练、评测、服务三大入口。
- **模型理解**：理解 Seq2Seq、CMA、RDP、NavDP 四大基线，以及 InternVLA-N1 双系统设计。
- **环境评测**：理解 Habitat、InternUtopia、RealWorld 三种环境的动作空间与评测链路。
- **工程实践**：了解 DeepSpeed 训练、分布式评测、ROS2 真实部署的关键步骤与常见坑点。

---

## 项目速览

| 项目 | 内容 |
|------|------|
| 项目名称 | InternNav |
| 定位 | 开源 PyTorch 具身导航工具箱 |
| 版本 | 0.3.1 |
| 支持任务 | VLN-CE、VLN-PE、VN、Dual-System VLN、VLLN |
| 旗舰模型 | InternVLA-N1（基于 Qwen2.5-VL） |
| 基线模型 | Seq2Seq、CMA、RDP、NavDP |
| 主要平台 | Habitat、InternUtopia / Isaac Sim、ROS2 真实机器人 |
| 仓库路径 | `/ephstorage/vln_code/InternNav` |

---

## 核心架构

```text
用户命令 / 配置文件
       │
       ▼
┌─────────────────┐
│  scripts/train  │ 训练入口：train.py
│  scripts/eval   │ 评测入口：eval.py / start_server.py
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Registry 层                         │
│  Agent.agents / Env.envs /           │
│  Evaluator.evaluators / get_policy() │
└────────┬────────────────────────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
 Agent      Env     Evaluator    Policy
 (决策)    (环境)    (评测编排)   (模型)
```

### 关键抽象

| 抽象 | 基类 | 注册表 | 核心方法 |
|------|------|--------|---------|
| Agent | `internnav/agent/base.py` | `Agent.agents` | `Agent.init(config).step(obs)` |
| Env | `internnav/env/base.py` | `Env.envs` | `Env.init(env_cfg, task_cfg)` |
| Evaluator | `internnav/evaluator/base.py` | `Evaluator.evaluators` | `Evaluator.init(cfg).eval()` |
| Policy | `internnav/model/__init__.py` | `get_policy(name)` | `from_pretrained(...)` |

### 已注册组件

- **Agents**: `cma`, `seq2seq`, `rdp`, `internvla_n1`, `dialog`
- **Envs**: `habitat`, `internutopia`, `realworld`
- **Evaluators**: `vln_distributed`, `habitat_vln`, `habitat_evaluator`, `habitat_dialog`
- **Policies**: `CMA_Policy`, `CMA_CLIP_Policy`, `Seq2Seq_Policy`, `RDP_Policy`, `NavDP_Policy`, `InternVLAN1_Policy`

---

## 章节索引

| 章节 | 标题 | 目标 |
|------|------|------|
| [第 1 章](01_全局架构与注册机制.md) | 全局架构与注册机制 | 理解四大抽象、注册表实现、项目骨架 |
| [第 2 章](02_配置系统与入口脚本.md) | 配置系统与入口脚本 | 掌握 Pydantic 配置、训练/评测/服务入口 |
| [第 3 章](03_基线模型.md) | 基线模型 | 理解 Seq2Seq、CMA、RDP、NavDP 的原理与差异 |
| [第 4 章](04_InternVLA_N1_双系统模型.md) | InternVLA-N1 双系统模型 | 理解 S2 规划器 + S1 控制器的协同 |
| [第 5 章](05_环境封装与评测链路.md) | 环境封装与评测链路 | 理解 Habitat/InternUtopia/RealWorld 与评测器 |
| [第 6 章](06_训练部署与工程实践.md) | 训练、部署与工程实践 | 掌握 DeepSpeed 训练、分布式评测、真实部署 |

---

## 学习资源

- 项目 README：`/ephstorage/vln_code/InternNav/README.md`
- Agent 指南：`/ephstorage/vln_code/InternNav/AGENTS.md`
- 变更日志：`/ephstorage/vln_code/InternNav/docs/changelog.md`
- 兼容性说明：`/ephstorage/vln_code/InternNav/docs/compatibility.md`
- 学习计划：`/ephstorage/vln_code/InternNav/docs/InternNav学习计划.md`
- 学习记录：`/ephstorage/vln_code/InternNav/docs/InternNav学习记录.md`

---

## 进度快照

- 总章节数：6
- 已完成：4
- 进行中：0
- 状态：✅ 第 4 章已完成，准备进入第 5 章

> 由 `docs/teaching/PROGRESS.md` 持续更新。
