"""
Demo script: InternVLA-N1 推理测试
基于 scripts/notebooks/inference_only_demo.ipynb 的逻辑
用法（在 InternNav 目录下）:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True conda run -n internvla_n1 python scripts/notebooks/run_demo.py
"""
import sys
import os
import glob
import warnings
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image
import torch

# ============================================================
# 0. Monkey-patch: flash_attn 未安装，用 eager attention 代替
# ============================================================
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
_orig = InternVLAN1ForCausalLM.from_pretrained

@classmethod
def _patched(cls, *args, **kwargs):
    kwargs["attn_implementation"] = "eager"
    return _orig.__func__(cls, *args, **kwargs)

InternVLAN1ForCausalLM.from_pretrained = _patched

warnings.filterwarnings("ignore")

# ============================================================
# 1. 导入 Agent
# ============================================================
from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

# ============================================================
# 2. 参数配置
# ============================================================
class Args:
    device = "cuda:0"
    model_path = str(project_root / "model" / "InternVLA-N1-DualVLN")
    resize_w = 384
    resize_h = 384
    num_history = 4          # MIG 20GB 显存限制，不能用 8
    plan_step_gap = 4
    camera_intrinsic = np.array([
        [386.5, 0.0, 328.9, 0.0],
        [0.0, 386.5, 244.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

# ============================================================
# 3. 加载模型 & 数据
# ============================================================
args = Args()
scene_dir = project_root / "assets" / "realworld_sample_data1"
instruction = (scene_dir / "instruction.txt").read_text().strip()
rgb_paths = sorted(glob.glob(str(scene_dir / "debug_raw_*.jpg")))

print(f"Device: {args.device}")
print(f"Instruction: {instruction}")
print(f"Images: {len(rgb_paths)}")

agent = InternVLAN1AsyncAgent(args)

# Warm up
agent.reset()
agent.step(
    np.zeros((480, 640, 3), dtype=np.uint8),
    np.zeros((480, 640), dtype=np.float32),
    np.eye(4),
    "hello",
    intrinsic=args.camera_intrinsic,
)

# ============================================================
# 4. 逐帧推理
# ============================================================
agent.reset()
N = 6

for i, rgb_path in enumerate(rgb_paths[:N]):
    look_down = "look_down" in rgb_path
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = 10.0 * np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
    pose = np.eye(4)

    with torch.no_grad():
        output = agent.step(rgb, depth, pose, instruction,
                            intrinsic=args.camera_intrinsic, look_down=look_down)

    basename = os.path.basename(rgb_path)
    if output.output_action:
        print(f"  [{i}] {basename} → action: {output.output_action}")
    elif output.output_pixel is not None:
        print(f"  [{i}] {basename} → pixel=({output.output_pixel[0]},{output.output_pixel[1]}) "
              f"traj_len={len(output.output_trajectory)}")
    else:
        print(f"  [{i}] {basename} → no output (waiting for S2)")

    torch.cuda.empty_cache()

print("Done.")
