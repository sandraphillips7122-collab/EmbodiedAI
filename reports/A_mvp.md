# 项目 A · MVP 报告：PickCube 模仿学习闭环（数据 → 训练 → 评测 → 失败归因）

> 本文所有数字均来自本仓库可复现的实验，原始记录在 `experiments/<exp_id>/`
> （`episodes.jsonl` 逐 episode 记录、`meta.json` 配置快照、`summary.json` 汇总）。
> **仿真环境验证，未做真机（Sim2Real）迁移。**

运行环境：Windows 原生 / RTX 4060 Laptop 8 GB / `mani_skill 3.0.1` + `sapien 3.0.3` / `torch 2.9.1+cu128`
（环境决策与三个 Windows 硬约束见 `docs/env_decision.md`）

## 1. 实验设置

| 项 | 值 |
|---|---|
| 任务 | `PickCube-v1`：抓取红色方块并移动到目标位置（阈值 0.025 m） |
| 观测 | `rgb`：1 个相机（`base_camera`，128×128，训练时 resize 到 224）+ **29 维 state**（`qpos(9) + qvel(9) + is_grasped(1) + tcp_pose(7) + goal_pos(3)`） |
| 控制 | **关节空间** `pd_joint_delta_pos`，动作 8 维（7 关节增量 + 夹爪） |
| 仿真 | `sim_backend=physx_cpu`，`render_backend=cpu`（Windows 约束） |
| 示教 | 官方 `motionplanning` 1000 条 → 全量 RGB replay 成功（418 MB）；训练只用**前 100 条**（约 7700 个训练样本） |
| 模型 | ACT：resnet18 backbone + DETRVAE（hidden 256，2 enc + 4 dec，`num_queries=30`），不含 depth |
| 训练 | batch 32，lr 1e-4，**30000 iters**，EMA；loss 98.9 → 0.0090 |
| 评测 | **3 个 seed 块 × 50 episodes = 150 次 rollout**，每 episode 上限 150 步，`ignore_terminations=True` |

## 2. 主结果（MVP）

| 指标 | 值 |
|---|---|
| **成功率** | **59 / 150 = 0.3933** |
| 标准误 | ±0.0399 |
| 95% 置信区间（Wilson） | **[0.319, 0.473]** |
| 失败模式分布 | `success: 59`，`timeout: 91` |
| 平均 episode 耗时 | 5.60 s |

原始记录：`experiments/A_mvp_pickcube_act/`（含 3 个 MP4：`videos/0.mp4` 等）

## 3. 泛化实验

每个变体 **30 episodes**（seeds 10–12），其余设置同 §1：

| 变体 | 成功率 | ±SE | 相对 baseline | 说明 |
|---|---|---|---|---|
| baseline | 5/30 = **0.167** | 0.068 | — | 与 MVP 用不同 seed 块 |
| 初始位姿范围加倍 | 5/30 = **0.167** | 0.068 | **无变化** | `cube_spawn_half_size` 0.05 → 0.10 m |
| 光照变暗 | 1/30 = **0.033** | 0.033 | **−80%** | ambient 0.3 → 0.03 |
| 光照变亮 | 1/30 = **0.033** | 0.033 | **−80%** | ambient 0.3 → 1.0 |

**结论**：
1. **光照是当前策略最脆弱的泛化维度**（两个方向都掉 80%）——说明视觉特征对整体亮度高度敏感，这是后续最值得做的数据增强方向（randomize lighting / 颜色抖动）。
2. 初始位姿范围加倍**没有影响**，说明位姿维度的覆盖是够的。
3. baseline（seeds 10–12）= 0.167 vs MVP（seeds 0–2）= 0.393：**同一 checkpoint 在不同 seed 块上差异近 2.4 倍**，说明 30–50 次 rollout 的抽样波动不可忽视 —— 这也是本文坚持报标准误与置信区间、而不是只报一个点的原因。

## 4. 过程中发现并修复的问题（都直接影响结果可信度）

| 问题 | 现象 | 根因与修复 |
|---|---|---|
| **`max_episode_steps` 没传给 env** | 成功率 0.1，`steps_mean=50` | 只设了 rollout 循环上限，env 内部仍是 PickCube 默认的 **50 步**，而示教平均 77 步 → 策略没做完就被截断。改为 `EnvSpec(max_episode_steps=150)`（即传给 `gym.make`）后成功率 **0.1 → 0.6**（20 episodes 探路） |
| **`norm_stats` 是 `None`** | 加载 checkpoint 后无法归一化 | delta 控制模式下官方**不计算** norm stats（`train_rgbd.py:233`）→ 按官方 `evaluate.py` 的 `delta_control = not stats` 语义**跳过** state 标准化与 action 反归一化 |
| **state_dict 键前缀** | `Missing keys: "model.pos_table"` | checkpoint 存的是 `Agent.state_dict()`（带 `model.` 前缀），加载时 `removeprefix("model.")` |
| **`CPUGymWrapper` 剥掉 batch 维并转 numpy** | `ndarray has no attribute permute` / `input.dim()=3` | 观测树里的数值字段需统一补 batch 维（图像 `(H,W,C)`→`(B,H,W,C)`，向量 `(N,)`→`(B,N)`）后才与官方 `FlattenRGBDObservationWrapper` 的假设一致 |

## 5. 局限（明确声明，不作过度声称）

- **纯仿真，未做真机迁移**；不声称任何 Sim2Real 效果。
- **控制空间是关节空间**：Windows 上没有 pinocchio（IK 不可用），无法使用官方 baseline 默认的 `pd_ee_delta_pos`。策略必须自行学出逆运动学，**难度高于官方设置**，因此 0.39 这个数字不能与官方 ee 控制的 baseline 直接对比。
- 渲染走 CPU（Windows 约束），训练不受影响。
- 仅单任务 `PickCube-v1`；`StackCube-v1` 与 DP 对照未做（见计划中的完整版）。
- 失败模式目前只区分 `success` / `timeout`；更细的「未接触 / 滑落 / 碰撞」分类需要从轨迹里提取几何信号，属于后续工作。

## 6. 复现

见 `project-a-il/README.md`（含数据准备、训练、评测、泛化实验的完整命令）。
