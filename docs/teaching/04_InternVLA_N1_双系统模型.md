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

文件：`scripts/eval/configs/h1_internvla_n1_async_cfg.py`

#### 4.7.1 整体结构

```text
EvalCfg
├── agent      # 模型与推理参数
├── env        # 仿真器参数
├── task       # 任务与机器人参数
├── dataset    # 评测数据集
└── eval_type / eval_settings  # 评测模式
```

#### 4.7.2 Agent 关键参数

```python
agent=AgentCfg(
    server_port=8023,
    model_name='internvla_n1',
    ckpt_path='',
    model_settings={
        'env_num': 1,
        'sim_num': 1,
        'model_path': "checkpoints/InternVLA-N1-DualVLN",
        'camera_intrinsic': [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
        'width': 640, 'height': 480, 'hfov': 79,
        'resize_w': 384, 'resize_h': 384,
        'max_new_tokens': 1024,
        'num_frames': 32,
        'num_history': 8,
        'num_future_steps': 4,
        'device': 'cuda:0',
        'predict_step_nums': 32,
        'continuous_traj': True,
        'infer_mode': 'partial_async',
        'vis_debug': True,
    },
)
```

#### 4.7.3 机器人与任务参数

```python
task=TaskCfg(
    task_name='test_n1',
    task_settings={
        'env_num': 1,
        'use_distributed': False,
        'proc_num': 1,
        'max_step': 1000,      # flash 模式默认 1000；物理模式设 50000
    },
    scene=SceneCfg(
        scene_type='mp3d',
        scene_data_dir='data/scene_data/mp3d_pe',
    ),
    robot_name='h1',
    robot_flash=True,         # True=flash 模式；False=物理模式
    flash_collision=False,
    robot_usd_path='data/Embodiments/vln-pe/h1/h1_internvla.usd',
    camera_resolution=[640, 480],
    camera_prim_path='torso_link/h1_1_25_down_30',
    one_step_stand_still=True,  # dual system 必须 True
)
```

重点：

- `robot_flash=True`：直接设置 world pose（瞬移），评测速度快、稳定。
- `one_step_stand_still=True`：每步之间先 stand still，确保 System 2 看到稳定图像。
- `max_step=1000`：flash 模式；物理模式需设 `50000`。

#### 4.7.4 数据集与运行命令

```python
dataset=EvalDatasetCfg(
    dataset_type="mp3d",
    dataset_settings={
        'base_data_dir': 'data/vln_pe/raw_data/r2r',
        'split_data_types': ['val_unseen'],
        'filter_stairs': True,   # 论文 True；IROS challenge False
    },
)

eval_type='vln_distributed'
eval_settings={
    'save_to_json': True,
    'vis_output': True,
    'use_agent_server': False,
}
```

运行：

```bash
python scripts/eval/eval.py --config scripts/eval/configs/h1_internvla_n1_async_cfg.py
```

### 4.8 真实世界部署

#### 4.8.1 核心文件

| 文件 | 作用 |
|------|------|
| `internnav/agent/internvla_n1_agent_realworld.py` | 真实部署 Agent |
| `scripts/realworld/http_internvla_server.py` | Flask 推理服务 |
| `scripts/realworld/http_internvla_client.py` | ROS2 客户端 |
| `scripts/realworld/controllers.py` | MPC / PID 控制器 |
| `scripts/realworld/thread_utils.py` | 读写锁 |

#### 4.8.2 `InternVLAN1AsyncAgent`

和仿真评测不同，真实部署直接加载大模型：

```python
self.model = InternVLAN1ForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": self.device},
)
```

运行节奏：

```python
def step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
    # 每隔 PLAN_STEP_GAP 触发一次 System 2
    if (self.episode_idx - self.last_s2_idx > self.PLAN_STEP_GAP) or ...:
        action, latent, pixel = self.step_s2(...)

    # 两次 S2 之间复用 latent，交给 S1 生成轨迹
    if latent is not None:
        trajectories = self.step_s1(latent, rgbs, depths)
```

#### 4.8.3 HTTP Server

```python
@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    image = request.files['image']      # JPEG
    depth = request.files['depth']      # PNG uint16
    data = json.loads(request.form['json'])  # {"reset": bool, "idx": int}

    depth = depth.astype(np.float32) / 10000.0   # 转成米

    dual_sys_output = agent.step(image, depth, camera_pose, instruction, ...)
    ...
    return jsonify(json_output)
```

默认端口 `5801`，绑定 `0.0.0.0`。不要直接暴露到公网。

#### 4.8.4 ROS2 Client

`http_internvla_client.py`：

- 订阅 RGB、depth、`/odom_bridge`。
- `planning_thread`：发 HTTP 请求，拿到 `trajectory` 或 `discrete_action`。
- `control_thread`：10Hz 循环，用 MPC/PID 解算速度，发布 `/cmd_vel_bridge`。

部署架构：

```text
机器人端（ROS2）                    模型端（GPU 工作站）
├─ 订阅 RGB-D、里程计              ├─ Flask Server :5801
├─ planning_thread ──HTTP POST──►  ├─ InternVLAN1AsyncAgent
├─ control_thread                  └─ InternVLA-N1 16GB
└─ 发布 /cmd_vel_bridge
```

### 4.9 数据格式 v0.5

训练数据使用 **LeRobot** 格式：

```text
dataset/
├── meta/
│   └── episodes.jsonl
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
└── videos/
    └── chunk-000/
        └── episode_000000.mp4
```

v0.5 关键变更：

```text
v0.1: instruction_text
v0.5: task
```

读取位置：

```python
# internnav/dataset/internvla_n1_lerobot_dataset.py:770
ep_instructions = ep["tasks"][0].split("<INSTRUCTION_SEP>")
```

一个 episode 可以包含多条指令，用 `<INSTRUCTION_SEP>` 分隔。

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

- InternVLA-N1 = Qwen2.5-VL（System 2 规划）+ nextdit/navdp（System 1 控制）。
- `InternVLAN1Net` 是仿真评测用的 Policy 包装层。
- `InternVLAN1Agent` 支持 `sync` / `partial_async` 两种运行模式；当前项目默认用 `partial_async`。
- 训练分两步：先 `train_system2.sh` 训 S2，再 `train_dual_system.sh` 冻结 VLM 训 S1。
- 仿真评测用 `InternVLAN1Agent` + `InternVLAN1Net`；真实部署用 `InternVLAN1AsyncAgent`。
- 真实部署通过 HTTP Server 接收 RGB/depth/json，depth 按 `/10000` 转成米。
- 数据格式 v0.5 用 `task` 字段替代 `instruction_text`，在 `internvla_n1_lerobot_dataset.py:770` 读取。

---

## 掌握自测

1. System 2 可以输出哪三种形式的高层计划？
2. `partial_async` 模式与 `sync` 模式在 S2/S1 执行方式上有什么区别？
3. Dual System 训练时为什么要冻结 VLM？
4. 仿真评测和真实部署分别用哪个 Agent/Policy 包装？
5. `robot_flash=True` 和 `False` 的区别是什么？
6. `one_step_stand_still=True` 对 dual system 有什么意义？
7. 真实部署 HTTP Server 接收哪些字段？depth 如何转成米？
8. v0.5 数据格式相比 v0.1 的核心变化是什么？代码里在哪里读取？
