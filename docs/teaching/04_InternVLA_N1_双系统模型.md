# 第 4 章 · InternVLA-N1 双系统模型

## 本章目标

学完本章后，你能：
1. 解释 InternVLA-N1 的 System 2 和 System 1 各自负责什么。
2. 说明 `nextdit` 与 `navdp` 两种 System 1 实现的区别。
3. 理解 `InternVLAN1Net` Policy 包装层与 `InternVLAN1Agent` 的运行模式。
4. 找到训练 System 2 与 Dual System 的脚本，并说明两阶段训练流程。
5. 理解真实部署与仿真评测的 Agent 差异。
6. 理解 LeRobot 数据格式与 v0.5 变更（`task` 替代 `instruction_text`）。

---

## 讲解

### 4.1 整体架构

InternVLA-N1 是基于 **Qwen2.5-VL** 的导航基础模型，采用双系统设计：

```text
System 2（高层规划器） ──► System 1（低层控制器） ──► 动作/轨迹
```

- **System 2**：多模态大模型，输入图像、指令、历史，输出 pixel goal / discrete action / latent plan。
- **System 1**：条件扩散模型（nextdit 或 navdp），把高层计划转换为连续轨迹。

### 4.2 System 2：Qwen2.5-VL 规划器

输入：当前 RGB、深度、位姿、指令、历史图像。  
输出：pixel goal（图像坐标）、discrete action（离散动作）、或 latent plan（隐式计划）。

核心 prompt（`internvla_n1_policy.py:64`）：

```python
"You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task."
```

`look_down` 用于辅助判断，俯视图像不加入历史序列。

### 4.3 System 1：nextdit / navdp 控制器

| 实现 | 文件 | 输入条件 | 特点 |
|------|------|---------|------|
| **nextdit** | `internnav/model/basemodel/internvla_n1/nextdit_*.py` | pixel goal / latent | RGB-only，流匹配扩散 Transformer |
| **navdp** | `internnav/model/basemodel/internvla_n1/navdp.py` | pixel goal / goal image | RGB-D，NavDP Transformer decoder |

输出为连续轨迹 `[B, T, 3]`，每步 `(dx, dy, dyaw)`。仿真评测路径下会通过 `traj_to_actions()` 或 `chunk_token()` 解析为离散动作。

### 4.4 `InternVLAN1Net` Policy 包装层

文件：`internnav/model/basemodel/internvla_n1/internvla_n1_policy.py:26`

- 加载 `InternVLAN1ForCausalLM`、tokenizer、processor。
- 提供 `s2_step()` 和 `s1_step_latent()` 等高层调用接口。
- 仿真评测用；真实部署直接实例化 `InternVLAN1AsyncAgent`，跳过此包装。

### 4.5 Agent 运行模式

`internnav/agent/internvla_n1_agent.py:36`：

```python
self.mode = getattr(self._model_settings, 'infer_mode', 'sync')
```

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `sync` | S2 和 S1 串行执行 | 简单、可复现 |
| `partial_async` | S2 后台线程推理，S1 主线程快速响应 | **推荐**，延迟更低 |

### 4.6 两阶段训练

**阶段 1 — 训练 System 2：**

```bash
bash scripts/train/qwenvl_train/train_system2.sh
```

- 基模型：`Qwen2.5-VL-7B-Instruct`
- `system1=none`
- 微调 vision tower、MLP merger、LLM

**阶段 2 — 训练 Dual System：**

```bash
bash scripts/train/qwenvl_train/train_dual_system.sh
```

- 从 System 2 checkpoint 开始
- `system1=nextdit_async` 或 `navdp_async`
- 冻结 VLM，只训练 System 1 + latent queries

DeepSpeed 配置：`zero2.json` / `zero3.json` / `zero3_offload.json`。

### 4.7 评测配置

> 待明天讲解。  
> 核心文件：`scripts/eval/configs/h1_internvla_n1_async_cfg.py`。

### 4.8 真实世界部署

> 待明天讲解。  
> 核心文件：`scripts/realworld/http_internvla_server.py`。

### 4.9 数据格式 v0.5

> 待明天讲解。  
> 核心文件：`internnav/dataset/internvla_n1_lerobot_dataset.py`。

---

## 关键代码走读

### `internnav/model/basemodel/internvla_n1/internvla_n1.py`

`InternVLAN1ForCausalLM` 继承 `Qwen2_5_VLForConditionalGeneration`，实现 `generate_latents` 与 `generate_traj`。

### `internnav/model/basemodel/internvla_n1/internvla_n1_arch.py`

`latent_queries`、`traj_dit`、NavDP builder 的组装位置。

### `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`

`InternVLAN1Net`：加载模型、tokenizer、processor，封装 `s2_step()` 和 `s1_step_latent()`。

### `internnav/agent/internvla_n1_agent.py`

`InternVLAN1Agent`：仿真评测用，支持 `sync` / `partial_async`。

### `internnav/agent/internvla_n1_agent_realworld.py`

`InternVLAN1AsyncAgent`：真实部署用，直接加载 `InternVLAN1ForCausalLM`。

### `internnav/trainer/internvla_n1_trainer.py`

训练循环入口，以及 trainable-parameter masking 逻辑。

### `scripts/train/qwenvl_train/zero2.json`

DeepSpeed ZeRO-2 配置。

---

## 小结

- InternVLA-N1 = Qwen2.5-VL（System 2）+ nextdit/navdp（System 1）。
- `InternVLAN1Net` 是仿真评测用的 Policy 包装层。
- `InternVLAN1Agent` 支持 `sync` / `partial_async` 两种运行模式。
- 训练分两步：先训 S2，再冻结 S2 训 S1。

---

## 掌握自测

1. System 2 可以输出哪三种形式的高层计划？
2. `partial_async` 模式与 `sync` 模式在 S2/S1 执行方式上有什么区别？
3. Dual System 训练时为什么要冻结 VLM？
4. 仿真评测和真实部署分别用哪个 Agent？
