# InternNav — Agent Guide

> This file is written for AI coding agents. It assumes no prior knowledge of the project. All statements below are derived from the actual repository contents (`README.md`, `setup.py`, `pyproject.toml`, source tree, CI workflow, tests, etc.).

---

## 1. Project Overview

**InternNav** is an open-source PyTorch toolbox for embodied navigation. It supports:

- Vision-and-Language Navigation with discrete actions (VLN-CE).
- Vision-and-Language Navigation in Physical Environments (VLN-PE).
- Visual Navigation (VN) given point / image / trajectory goals.
- The full dual-system VLN pipeline with continuous trajectory outputs.
- Multi-turn dialog / instance-goal navigation (VL-LN / IIGN).

The flagship model is **InternVLA-N1**, a dual-system navigation foundation model built on Qwen2.5-VL. The codebase also implements several baselines: Seq2Seq, CMA, RDP, and NavDP.

| Item | Value |
|------|-------|
| Package name | `internnav` |
| Current version | `0.3.1` (from `setup.py` and `docs/changelog.md`) |
| License | `LICENSE` and `README.md` state MIT; `setup.py` states `Apache 2.0`. Note the discrepancy. |
| Supported Python | 3.8, 3.9, 3.10, 3.11, 3.12 (enforced in `setup.py`) |
| Main platforms | Habitat, Isaac Sim via InternUtopia, real-world ROS2 deployment |

### Compatibility note (v0.3.1)

The InternData-N1 VLN-PE trajectory training dataset was upgraded from `v0.1` to `v0.5`. The conversion logic and training pipeline now use the key `task` instead of `instruction_text`. This is **not backward-compatible** with the `v0.1` dataset. See `docs/compatibility.md`.

---

## 2. Technology Stack

### Core runtime

- **PyTorch** — deep-learning runtime.
- **Habitat / Habitat-Lab** — classic discrete-action VLN simulation.
- **InternUtopia** (formerly GRUtopia) / Isaac Sim — physics-based continuous simulation.
- **ROS2** — real-world robot deployment (`scripts/realworld/`).
- **FastAPI + uvicorn** — model inference server.
- **Ray** — distributed evaluation.
- **LeRobot** — dataset format conventions.

### Key Python libraries

- `transformers==4.51.0`, `diffusers`, `accelerate`, `flash_attn` — for InternVLA-N1 and diffusion baselines.
- `pydantic>=2.11.0,<2.12` — configuration schemas.
- `tyro>=0.9.26,<0.10` — CLI parsing for training.
- `gym>=0.22.0,<=0.26.2`, `gymnasium==0.29.1` — environment interfaces.
- `opencv-python-headless`, `open3d`, `numpy>=1.26,<2.0` — perception / geometry.
- `pytest`, `pytest-timeout`, `pytest-cov` — testing.

Dependencies are split across five requirement files. The exact mapping in `setup.py` is:

- `requirements/core_requirements.txt` — always installed.
- `requirements/habitat_requirements.txt` — `pip install -e ".[habitat]"`.
- `requirements/isaac_requirements.txt` — `pip install -e ".[isaac]"`.
- `requirements/model_requirements.txt` — `pip install -e ".[model]"` or `"[baseline]"`.
- `requirements/internvla_n1.txt` — `pip install -e ".[internvla_n1]"`.

Verify the exact contents of each file before assuming a dependency belongs to a specific backend; the requirement lists overlap (e.g., `transformers`, `diffusers`, `flash_attn` appear in both `[habitat]` and `[internvla_n1]` extras).

---

## 3. Repository Structure

```text
.
├── internnav/                    # Main Python package (~377 Python files, ~149 dirs)
│   ├── agent/                    # Decision agents (registry pattern)
│   ├── configs/                  # Pydantic configuration schemas
│   │   ├── agent/
│   │   ├── evaluator/
│   │   ├── model/
│   │   └── trainer/
│   ├── dataset/                  # Iterable datasets (LeRobot / LMDB / torch Dataset)
│   ├── env/                      # Simulator wrappers
│   │   ├── habitat_env.py
│   │   ├── internutopia_env.py
│   │   └── realworld_agilex_env.py
│   ├── evaluator/                # Evaluation orchestration
│   ├── habitat_extensions/       # Habitat task/measure extensions
│   │   ├── vln/                  # Standard VLN evaluator
│   │   └── vlln/                 # Dialog / multi-turn VLLN
│   ├── model/                    # Policies and encoders
│   │   ├── basemodel/
│   │   │   ├── cma/
│   │   │   ├── seq2seq/
│   │   │   ├── rdp/
│   │   │   ├── navdp/
│   │   │   ├── internvla_n1/
│   │   │   ├── diffusion_policy_modified/
│   │   │   └── LongCLIP/         # Git submodule
│   │   ├── encoder/
│   │   └── utils/
│   ├── trainer/                  # Policy-specific trainers
│   ├── utils/                    # Shared utilities
│   │   └── comm_utils/           # AgentClient / AgentServer
│   └── __init__.py
├── scripts/
│   ├── eval/                     # Evaluation launcher, server, bash helpers, configs
│   ├── train/base_train/         # Unified baseline training entry point and configs
│   ├── train/qwenvl_train/     # InternVLA-N1 DeepSpeed / SLURM training scripts
│   ├── realworld/                # ROS2 / HTTP real-world deployment
│   ├── iros_challenge/           # IROS 2025 challenge scaffolding
│   ├── notebooks/                # Inference demo notebook and run_demo.py
│   └── dataset_converters/       # LeRobot conversion tools
├── tests/                        # pytest test suite
├── third_party/
│   └── diffusion-policy/         # Git submodule
├── requirements/                 # Split dependency lists
├── docs/                         # changelog.md, compatibility.md
├── pyproject.toml                # black, isort, pytest
├── setup.cfg                     # flake8, yapf, isort, codespell
├── setup.py                      # Package metadata
└── .pre-commit-config.yaml       # Pre-commit hooks
```

---

## 4. Build and Installation

### Clone

```bash
git clone git@github.com:InternRobotics/InternNav.git --recursive
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

Required submodules (from `.gitmodules`):

- `internnav/model/basemodel/LongCLIP` → `https://github.com/beichenzbc/Long-CLIP`
- `third_party/diffusion-policy` → `https://github.com/real-stanford/diffusion_policy.git`

### Install

```bash
# Core only
pip install -e .

# With specific backends / models
pip install -e ".[habitat]"
pip install -e ".[isaac]"
pip install -e ".[model]"      # baseline model dependencies
pip install -e ".[internvla_n1]"

# All extras
pip install -e ".[habitat,isaac,model,internvla_n1]"
```

### Important path note

The `diffusion-policy` submodule lives at `third_party/diffusion-policy`. However, the codebase is not consistent about where it adds this path:

- Uses `third_party/diffusion-policy`: `scripts/eval/eval.py`, `scripts/eval/start_server.py`.
- Uses `src/diffusion-policy` (legacy): `scripts/train/base_train/train.py`.

If you hit import errors during training, verify the path matches your checkout.

---

## 5. Runtime Architecture

### Registry pattern

The framework uses simple class registries for its core concepts.

| Concept | Base class | Registry | Key method |
|---------|-----------|----------|------------|
| Agent | `Agent` (`internnav/agent/base.py`) | `Agent.agents` | `Agent.init(config).step(obs)` |
| Environment | `Env` (`internnav/env/base.py`) | `Env.envs` | `Env.init(env_cfg, task_cfg)` |
| Evaluator | `Evaluator` (`internnav/evaluator/base.py`) | `Evaluator.evaluators` | `Evaluator.init(cfg).eval()` |
| Policy | factory in `internnav/model/__init__.py` | `get_policy(name)`, `get_config(name)` | `from_pretrained(...)` |

Registered agents include:

- `cma` → `CmaAgent`
- `rdp` → `RdpAgent`
- `seq2seq` → `Seq2SeqAgent`
- `internvla_n1` → `InternVLAN1Agent`
- `dialog` → `DialogAgent`

Registered environments include:

- `habitat` → `HabitatEnv`
- `internutopia` → `InternutopiaEnv`
- `realworld` → `RealWorldEnv` (registered but not exported in `internnav/env/__init__.py`)

Registered evaluators include:

- `vln_distributed` → `VLNDistributedEvaluator`
- `habitat_vln` → `HabitatVLNEvaluator`
- `habitat_evaluator` → `HabitatDefaultEvaluator`
- `habitat_dialog` → `HabitatDialogEvaluator`

Supported policy names (for `get_policy` / `get_config`):

- `CMA_Policy`
- `CMA_CLIP_Policy`
- `Seq2Seq_Policy`
- `RDP_Policy`
- `NavDP_Policy`
- `InternVLAN1_Policy`

### Client-server evaluation

The default evaluator creates an `AgentClient` that talks to a remote/local `AgentServer` over HTTP:

- Server: `scripts/eval/start_server.py` (FastAPI, default port `8087`).
- Server routes: `POST /agent/init`, `POST /agent/{name}/step`, `POST /agent/{name}/reset`.
- Observations are `pickle` + `base64` encoded over JSON. See `internnav/utils/comm_utils/server.py`.
- Client: `internnav/utils/comm_utils/client.py` (`AgentClient`).

This decouples the model inference process from the environment rollout process, which is useful for multi-GPU or distributed evaluation.

### Training

- Baseline training entry point: `scripts/train/base_train/train.py`.
- It uses `tyro.cli(TrainCfg)` to parse `--name` and `--model_name`.
- Supported `model_name` values: `cma`, `cma_plus`, `seq2seq`, `seq2seq_plus`, `rdp`, `navdp`.
- It loads the matching `ExpCfg` from `scripts/train/base_train/configs/` and calls the corresponding trainer (`CMATrainer`, `RDPTrainer`, `NavDPTrainer`).
- InternVLA-N1 training uses `scripts/train/qwenvl_train/` with DeepSpeed configs (`zero2.json`, `zero3.json`, `zero3_offload.json`) and SLURM shell scripts (`train_system2.sh`, `train_dual_system.sh`, `train_system2_vlln.sh`).

### Evaluation

- Main launcher: `scripts/eval/eval.py`.
- It loads a Python config file (`--config`) that exposes an `eval_cfg` variable.
- For `eval_type == 'vln_distributed'`, defaults are filled from `internnav/configs/evaluator/vln_default_config.py`.
- Helper scripts live in `scripts/eval/bash/` (SLURM, torchrun, distributed, dual-system, etc.).

---

## 6. Configuration System

Configs are **Pydantic `BaseModel`** objects, mostly with `extra='allow'` for forward compatibility.

Key config modules:

- `internnav/configs/agent/__init__.py` — `AgentCfg`, `InitRequest`, `StepRequest`, `ResetRequest`.
- `internnav/configs/evaluator/__init__.py` — `EnvCfg`, `TaskCfg`, `EvalCfg`, `EvalDatasetCfg`, `RobotCfg`, `SceneCfg`, `MetricCfg`, `SensorCfg`, `ControllerCfg`.
- `internnav/configs/evaluator/vln_default_config.py` — default VLN-PE evaluation settings and `get_config(evaluator_cfg)`.
- `internnav/configs/model/{base_encoders,cma,rdp,seq2seq,navdp,internvla_n1}.py` — per-model schemas.
- `internnav/configs/trainer/{exp,il,eval,task}.py` — experiment / training configs.

Training configs are plain Python objects in `scripts/train/base_train/configs/`. Evaluation configs are also Python files exposing `eval_cfg` in `scripts/eval/configs/`.

---

## 7. Code Style Guidelines

The project uses `pre-commit` with several formatters/linters.

### Pre-commit hooks (`.pre-commit-config.yaml`)

```bash
pre-commit run --all-files
# Or on a PR diff only
pre-commit run --from-ref origin/<base> --to-ref HEAD
```

Hooks run in this order:

1. `autoflake` — remove unused imports (excludes `__init__.py`).
2. `flake8`
3. `isort`
4. `black`
5. `codespell` — ignores the word `ro`, excludes `*.ipynb`.
6. `pre-commit-hooks` — trailing whitespace, YAML check, EOF fixer, merge-conflict check, encoding pragma removal, LF line endings.

Excludes: `^internnav/model/basemodel/LongCLIP/|^assets/`

### Tool configurations

- **Black** (`pyproject.toml`): line length `120`, `skip-string-normalization = true`, force-exclude `lcmtypes`.
- **isort** (`pyproject.toml`): profile `black`, skip `**/lcmtypes/**`.
- **flake8** (`setup.cfg`): max line length `120`, max complexity `30`, ignores `E402,E501,W503,E203,D401,R504,R505,SIM102,SIM117,E711,E226`, `per-file-ignores = */__init__.py:F401`.
- **yapf** (`setup.cfg`): configured but Black is the active formatter.
- **codespell** (`setup.cfg`): quiet level `3`, ignores a project-specific word list.

Style rule of thumb: run `pre-commit run --all-files` before committing.

---

## 8. Testing Instructions

Test runner: `pytest` (configured in `pyproject.toml`).

```bash
# Run all tests
pytest

# Skip GPU / slow tests
pytest -m "not gpu and not slow"

# Run Ray-based evaluator test only
pytest -m ray

# CI-style run (from .github/workflows/ci.yml)
python -m pytest -q -W ignore --timeout=900 --timeout-method=signal
```

### Test markers

Declared in `pyproject.toml`:

- `slow` — slow tests.
- `gpu` — requires CUDA.

Custom runtime skipping in `tests/conftest.py`:

- `gpu`-marked tests are skipped if `torch.cuda.is_available()` is false or torch is missing.
- `ray`-marked tests are skipped if Ray cannot initialize.

Note: `ray` and `cpu` are used in `tests/function_test/e2e_test.py` but are not registered in `pyproject.toml`, so pytest may emit `PytestUnknownMarkWarning`.

### Test layout

- `tests/unit_test/test_basic.py` — sanity checks, slow and GPU examples.
- `tests/function_test/test_server.py` — starts the FastAPI agent server as a subprocess.
- `tests/function_test/test_evaluator.py` — constructs an InternUtopia evaluator and runs rollout.
- `tests/function_test/e2e_test.py` — wraps server and evaluator tests via subprocess; writes `test_result.json` and `../total_result.jsonl`. Uses `subprocess.Popen(..., shell=True)`.

Note: the functional tests require a working InternUtopia / Habitat installation and scene data, which are not bundled in the repository.

---

## 9. CI / CD

File: `.github/workflows/ci.yml`.

- Triggers on pull requests to `main`, `master`, `develop` (opened, synchronize, reopened, ready_for_review); ignores `**.md` and `docs/**`.
- Runs on a `self-hosted` runner.
- Matrix: Python 3.10 on Linux.
- Concurrency: one job per PR, cancels in-progress runs.
- Skips draft PRs.
- Steps:
  1. Checkout with `submodules: recursive` and `fetch-depth: 0`.
  2. Set up Python 3.10.
  3. Activate the `internutopia` conda environment and run `pre-commit` on the PR diff.
  4. Create symlinks for data:
     - `data/vln_pe` → `/cpfs/user/wangyukai/mp3d_data/vln_pe`
     - `data/Embodiments` → `/cpfs/user/wangyukai/mp3d_data/Embodiments`
     - `data/scene_data` → `/cpfs/user/wangyukai/mp3d_data/scene_data`
     - `checkpoints` → `/cpfs/user/wangyukai/checkpoints`
  5. Run pytest with a 900-second timeout.

Do not assume these absolute data paths exist in your environment; they are specific to the project's self-hosted CI machine.

---

## 10. Real-World Deployment

The real-world stack is in `scripts/realworld/`:

- `http_internvla_server.py` — Flask server (port `5801`) that loads `InternVLAN1AsyncAgent` and serves `POST /eval_dual`.
- `http_internvla_client.py` — ROS2 node (`rclpy`) that subscribes to RGB-D + odometry topics and publishes `Twist` to `/cmd_vel_bridge`.
- `controllers.py` — MPC (CasADi/IPOPT) and PID controllers.
- `thread_utils.py` — `ReadWriteLock`.

The server expects multipart form fields: `image` (JPEG), `depth` (PNG), and `json` (`{"reset": bool, "idx": int}`).

The IROS 2025 on-site SDK is located at `scripts/iros_challenge/onsite_competition/sdk/` and contains camera, control, and real-world environment wrappers (`cam.py`, `control.py`, `main.py`, `real_world_env.py`, `stream.py`, `test_agent.py`, `test_robot.py`, `save_obs.py`).

---

## 11. Security Considerations

- The real-world Flask server (`scripts/realworld/http_internvla_server.py`) binds to `0.0.0.0:5801` with no authentication. If you expose this outside localhost, add authentication/TLS.
- The FastAPI `AgentServer` (`scripts/eval/start_server.py` and `internnav/utils/comm_utils/server.py`) pickles and unpickles observations over HTTP (`pickle.loads` in `step_agent`). Do not expose it to untrusted networks; unpickling untrusted payloads is unsafe.
- The CI workflow symlinks absolute internal data paths. Do not copy those paths into public documentation or other environments.
- Several scripts use `subprocess.Popen(..., shell=True)` (e.g., `tests/function_test/e2e_test.py`). Avoid passing user-controlled strings into these commands.
- The IROS on-site SDK stream server (`scripts/iros_challenge/onsite_competition/sdk/stream.py`) also binds to `0.0.0.0:8080` without authentication.

---

## 12. Development Conventions and Tips

### Adding a new baseline

1. Implement the policy in `internnav/model/basemodel/<name>/`.
2. Register it in `internnav/model/__init__.py` via `get_policy(name)` and `get_config(name)`.
3. Implement an `Agent` subclass in `internnav/agent/<name>_agent.py` and register it with `@Agent.register("<name>")`.
4. Add a `Trainer` subclass in `internnav/trainer/<name>_trainer.py` if needed.
5. Add training config in `scripts/train/base_train/configs/<name>.py`.
6. Add evaluation config in `scripts/eval/configs/`.

### Adding a new environment

1. Subclass `Env` in `internnav/env/<name>_env.py`.
2. Register it with `@Env.register("<name>")`.
3. Add any Habitat-specific evaluator extensions under `internnav/habitat_extensions/` if applicable.

### Data format compatibility

As of v0.3.1, the InternData-N1 VLN-PE trajectory dataset was upgraded from `v0.1` to `v0.5`. The training pipeline now uses the key `task` instead of `instruction_text`. The updated conversion logic is **not backward-compatible** with `v0.1`.

### Useful commands

```bash
# Editable install with all backends
pip install -e ".[habitat,isaac,model,internvla_n1]"

# Run pre-commit
pre-commit run --all-files

# Train a baseline
python scripts/train/base_train/train.py --name my_rdp --model rdp

# Evaluate with a config
python scripts/eval/eval.py --config scripts/eval/configs/h1_rdp_cfg.py

# Start the agent server
python scripts/eval/start_server.py --host localhost --port 8087
```

### When editing this file

If you change build steps, test commands, style tools, submodules, dependency extras, or deployment paths, update this `AGENTS.md` to keep it accurate.
