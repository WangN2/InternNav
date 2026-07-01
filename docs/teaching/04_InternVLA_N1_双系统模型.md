# 第 4 章 · InternVLA-N1 双系统模型

## 本章目标

学完本章后，你能：
1. 解释 InternVLA-N1 的 System 2 和 System 1 各自负责什么。
2. 说明 `nextdit` 与 `navdp` 两种 System 1 实现的区别。
3. 找到训练 System 2 与 Dual System 的脚本，并说明两阶段训练流程。
4. 理解 LeRobot 数据格式与 v0.5 变更（`task` 替代 `instruction_text`）。

---

## 讲解

### 4.1 模型定位

InternVLA-N1 是基于 **Qwen2.5-VL** 的导航基础模型。它把导航问题拆成两层：

- **System 2（高层规划器）**：多模态大模型，看图像、读指令、看历史，输出高层计划（pixel goal / discrete action / latent tokens）。
- **System 1（低层控制器）**：把 S2 的计划转换为连续轨迹或离散动作。

### 4.2 System 1 的两种实现

| 实现 | 文件 | 说明 |
|------|------|------|
| nextdit | `internnav/model/basemodel/internvla_n1/nextdit_*.py` | 流匹配扩散 Transformer，生成 3-DoF 相对轨迹 |
| navdp | `internnav/model/basemodel/internvla_n1/navdp.py` | NavDP Transformer decoder，RGB-D 条件 |

### 4.3 关键文件

- 主模型：`internnav/model/basemodel/internvla_n1/internvla_n1.py`
- 架构定义：`internnav/model/basemodel/internvla_n1/internvla_n1_arch.py`
- Policy 包装：`internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`
- Agent：`internnav/agent/internvla_n1_agent.py`
- 真实部署 Agent：`internnav/agent/internvla_n1_agent_realworld.py`

### 4.4 两阶段训练

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

### 4.5 数据格式

训练数据使用 **LeRobot** 格式。v0.5 关键变更：

- `instruction_text` → `task`
- 不再兼容 v0.1 的转换逻辑

数据集类：`internnav/dataset/internvla_n1_lerobot_dataset.py`

---

## 关键代码走读

### `internnav/model/basemodel/internvla_n1/internvla_n1.py`

`InternVLAN1ForCausalLM` 继承 `Qwen2_5_VLForConditionalGeneration`，实现 `generate_latents` 与 `generate_traj`。

### `internnav/model/basemodel/internvla_n1/internvla_n1_arch.py`

`latent_queries`、`traj_dit`、NavDP builder 的组装位置。

### `internnav/trainer/internvla_n1_trainer.py`

训练循环入口，以及 trainable-parameter masking 逻辑。

### `scripts/train/qwenvl_train/zero2.json`

DeepSpeed ZeRO-2 配置。

---

## 小结

- InternVLA-N1 用 Qwen2.5-VL 做高层理解，用 nextdit/navdp 做低层轨迹生成。
- 训练分两步：先训 S2，再冻结 S2 训 S1。
- 数据格式升级后使用 `task` 字段，与旧版不兼容。

---

## 掌握自测

1. System 2 可以输出哪三种形式的高层计划？
2. `partial_async` 模式与 `sync` 模式在 S2/S1 执行方式上有什么区别？
3. Dual System 训练时为什么要冻结 VLM？
4. v0.5 数据格式相比 v0.1 的核心变化是什么？在哪里读取这个字段？
