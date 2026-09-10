# 显存与训练吞吐报告（Phase 1 · S1）

实测日期：2026-09-10
硬件：NVIDIA GeForce RTX 4060 Laptop GPU（8188 MiB），Windows 原生
环境：`torch 2.9.1+cu128` / `mani_skill 3.0.1` / `sapien 3.0.3`（详见 `docs/env_decision.md`）

> 本文件所有数字均为本机实测，不含任何推测值。采样方式：`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2`。

## 1. 显存基线

| 项 | 实测 |
|---|---|
| Windows 桌面占用 | 1786–1824 MiB（90 次采样） |
| PyTorch 报告的可用显存 | **7106 MiB free / 8188 MiB total** |

注：Windows 的 WDDM 会为桌面预留显存，因此「标称 8 GB ≠ 可用 8 GB」。所有训练判据按 **< 7.0 GB** 执行。

## 2. ACT 训练 smoke test

**配置**

| 项 | 值 |
|---|---|
| env | `PickCube-v1`，`obs_mode=rgb`，`control_mode=pd_joint_delta_pos`，`sim_backend=physx_cpu` |
| 数据 | `trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5`，前 20 条 demo → **1495 个训练样本** |
| 模型 | ACT（resnet18 backbone，hidden_dim=256，2 enc + 4 dec，num_queries=30），**不含 depth** |
| 训练 | batch_size=32，lr=1e-4，total_iters=60，EMA |
| 入口 | `scripts/train_act_windows.py`（注入 Windows 必需 env 配置） |

**结果**

| 指标 | 实测值 | 证据 |
|---|---|---|
| **训练峰值显存** | **3310 MiB**（含桌面基线） | `nvidia-smi` 每 2 s 采样 |
| 训练增量 | ≈ **1.5 GB** | 3310 − 1790 |
| 剩余可用显存 | ≈ **4.9 GB** | 8188 − 3310 |
| **训练速度** | **≈ 0.12 s/iter** | tensorboard `time/update`：60 iters 累计 7.05 s |
| 30000 iters 预计 | **≈ 1 小时** | 30000 × 0.12 s |
| loss 曲线 | **98.20 → 10.11 → 5.66 → 3.62 → 2.78 → 2.88** | 训练日志 `Iteration 0/10/20/30/40/50` |
| tensorboard tags | `losses/total_loss`, `charts/learning_rate`, `time/update`, `time/eval` | `runs/act-smoke-rgb3/events.*` |

**结论**：RTX 4060 8GB **完全能训 ACT（rgb，batch 32）**，显存与速度都有充足余量；正式训练（30000 iters）约 1 小时，不需要云 GPU。

## 3. rollout 评测吞吐

| 项 | 实测 |
|---|---|
| 顺序单环境（`render_backend=cpu`） | **≈ 210 step/s**（`steps_mean=50`、`duration_s_mean=0.239`） |
| 100 条 rollout × 150 步 预计 | 约 1–2 分钟 |

## 4. 尚未测量（留待后续 Phase）

- [ ] Diffusion Policy 的显存与速度（项目 A 可选对照）
- [ ] 更大 batch（64 / 128）的显存上限
- [ ] SmolVLA 训练上限（Phase 3 的 S2 smoke test，以实测为准，不预设可训）
- [ ] 多进程并行 rollout 吞吐（当前用顺序单环境，已满足需求）

## 5. 过程中必须记住的三个坑（详见 `docs/env_decision.md` §8）

1. `render_backend` 必须显式设为 `"cpu"`，否则进程**原生崩溃**（exit 5、无 traceback）。
2. `sim_backend` 必须是 `"physx_cpu"`（Windows 无 GPU 仿真）；且官方 `act/make_env.py` 只在该取值下走 CPU 路径。
3. ACT 自带评测（`act/evaluate.py`）依赖 GPU 向量化环境的 `info["final_info"]`，在 CPU 路径下会 `KeyError` → 本项目统一改用 `embodied_core.rollout.runner` 评测（保证项目 A/B 结果可对照）。
