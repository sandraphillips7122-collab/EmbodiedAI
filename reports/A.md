# 项目 A · 报告：ManiSkill 模仿学习闭环（数据 → 训练 → 评测 → 失败归因）

> 本文所有数字均来自本仓库可复现的实验，原始记录在 `experiments/<exp_id>/`
> （`episodes.jsonl` 逐 episode 记录、`meta.json` 配置快照、`summary.json` 汇总）。
> **仿真环境验证，未做真机（Sim2Real）迁移。**

运行环境：Windows 原生 / RTX 4060 Laptop 8 GB / `mani_skill 3.0.1` + `sapien 3.0.3` / `torch 2.9.1+cu128`
（环境决策与三个 Windows 硬约束见 `docs/env_decision.md`）

## 1. 实验设置

| 项 | 值 |
|---|---|
| 任务 | `PickCube-v1`：抓取红色方块并移动到目标位置（阈值 0.025 m）；扩展任务 `StackCube-v1` |
| 观测 | `rgb`：1 个相机（`base_camera`，128×128，训练时 resize 到 224）+ **29 维 state**（`qpos(9) + qvel(9) + is_grasped(1) + tcp_pose(7) + goal_pos(3)`） |
| 控制 | **关节空间** `pd_joint_delta_pos`，动作 8 维（7 关节增量 + 夹爪） |
| 仿真 | `sim_backend=physx_cpu`，`render_backend=cpu`（Windows 约束） |
| 示教 | 官方 `motionplanning` 1000 条 → 全量 RGB replay 成功（418 MB）；主实验用**前 100 条**（约 7700 个训练样本） |
| 模型 | ACT：resnet18 backbone + DETRVAE（hidden 256，2 enc + 4 dec，`num_queries=30`），不含 depth |
| 训练 | batch 32，lr 1e-4，**30000 iters**，EMA；loss 98.9 → 0.0090 |
| 资源占用（实测） | 训练峰值显存 **3310 MiB**（含 Windows 桌面基线约 1.8 GB），约 **0.12 s/iter**（30000 iters ≈ 1 小时）；详见 [`docs/vram_report.md`](../docs/vram_report.md) |
| 评测 | **3 个 seed 块 × 50 episodes = 150 次 rollout**，每 episode 上限 150 步，`ignore_terminations=True` |

## 2. 主结果（PickCube）

| 指标 | 值 |
|---|---|
| **成功率** | **59 / 150 = 0.3933** |
| 标准误 | ±0.0399 |
| 95% 置信区间（Wilson） | **[0.319, 0.473]** |
| 失败模式分布 | `success: 59`，`timeout: 91` |
| 平均 episode 耗时 | 5.60 s |

原始记录：`experiments/A_mvp_pickcube_act/`（含 3 个 MP4：`videos/0.mp4` 等）

## 3. 第二任务：StackCube-v1（验证管道可移植性）

**动机**：证明 `embodied_core` 不是"只为 PickCube 写死"的脚本，而是换任务即可复用的基础设施。

改动量：`EnvSpec(task="StackCube-v1", max_episode_steps=200)` —— 仅此而已（数据卡 / 训练 / 评测 / 统计全链路复用）。

| 指标 | 值 |
|---|---|
| 成功率 | **13 / 30 = 0.4333** |
| 标准误 | ±0.0905 |
| 95% CI（Wilson） | **[0.274, 0.608]** |
| 失败模式 | `success: 13`，`timeout: 17` |

原始记录：`experiments/A_stackcube_act/`

**结论**：换任务的边际成本仅是一个 EnvSpec 字段；任务难度相近时成功率也在同一量级（0.3933 vs 0.4333）。

## 4. 数据量曲线（非线性的数据规模效应）

同口径对照（同一份 1 相机 replay 数据，同一批超参，**均取第 5000 步 checkpoint**，各 30 episodes）：

| 示教条数 | 成功率 | ±SE | 相对 50 条 |
|---|---|---|---|
| **50** | 3/30 = **0.1000** | 0.0548 | — |
| **100** | 4/30 = **0.1333** | 0.0621 | +33% |
| **200** | 21/30 = **0.7000** | 0.0837 | **+600%** |

> 说明：主实验的 0.3933 用的是 100 demos × **30000** iters，与上表不是同一批 checkpoint，**不可直接比较**；
> 上表是严格同口径的一组（同 iters、同 seed 块）。

**结论**：观察到**明显的非线性数据规模效应** —— 50→100 条几乎无变化（0.10 → 0.13），而 100→200 条跃升到 0.70。
本实验只能说明「在这个区间内数据规模的变化不是等比例地反映到成功率上」，**不足以断言数据量是唯一或主导因素**（未做控制其他变量的进一步实验）。

原始记录：`experiments/A_demos_{50,100_15k,200}/`

## 5. 泛化实验

每个变体 **30 episodes**（seeds 10–12），其余设置同 §1：

| 变体 | 成功率 | ±SE | 相对 baseline | 说明 |
|---|---|---|---|---|
| baseline | 5/30 = **0.167** | 0.068 | — | 与主结果用不同 seed 块 |
| 初始位姿范围加倍 | 5/30 = **0.167** | 0.068 | **无变化** | `cube_spawn_half_size` 0.05 → 0.10 m |
| 光照变暗 | 1/30 = **0.033** | 0.033 | **−80%** | ambient 0.3 → 0.03 |
| 光照变亮 | 1/30 = **0.033** | 0.033 | **−80%** | ambient 0.3 → 1.0 |

**结论**：
1. **光照是当前策略最脆弱的泛化维度**（两个方向都掉 80%）——说明视觉特征对整体亮度高度敏感，这是后续最值得做的数据增强方向（randomize lighting / 颜色抖动）。
2. 初始位姿范围加倍**没有影响**，说明位姿维度的覆盖是够的。
3. baseline（seeds 10–12）= 0.167 vs 主结果（seeds 0–2）= 0.393：**同一 checkpoint 在不同 seed 块上差异近 2.4 倍**，说明 30–50 次 rollout 的抽样波动不可忽视 —— 这也是本文坚持报标准误与置信区间、而不是只报一个点的原因。

## 6. 细粒度失败归因

把粗粒度的 `timeout` 拆成可归因类别（判据实现在 `embodied_core/metrics/failure_modes.py`，阈值集中在文件头部常量）：

| 标签 | 判据 |
|---|---|
| `no_contact` | 全程未夹住目标、且目标几乎未被移动 |
| `slip` | 曾经夹住（`is_grasped` 为真）但最终未成功 → 抓取后滑落/丢失 |
| `not_reached` | 夹住过且被送近目标，但仍差一点（`final_goal_distance < 0.05`） |
| `collision` | 从未夹住却把目标推动了 → 碰撞/推飞 |

**实测**（20 episodes，checkpoint = `act-pickcube-rgb-100demos/25000.pt`）：

| 标签 | 次数 | 占比 |
|---|---|---|
| success | 3 | 15% |
| **slip** | **12** | **60%** |
| not_reached | 4 | 20% |
| collision | 1 | 5% |

原始记录：`experiments/A_failure_modes.json`

**结论**：**主导失败模式是 `slip`（60%）**——策略已经能定位并夹住方块，问题出在**抓取后的保持与运输**阶段。
这把优化方向从"提高视觉感知"改成了"提升抓取稳定性"（更长的抓取奖励/更强的时间集成/夹爪力控制），
同时也解释了为什么单纯加数据（§4）到 200 条时提升最大：更多数据主要是在教会它"抓紧"。

## 7. 过程中发现并修复的问题（都直接影响结果可信度）

| 问题 | 现象 | 根因与修复 |
|---|---|---|
| **`max_episode_steps` 没传给 env** | 成功率 0.1，`steps_mean=50` | 只设了 rollout 循环上限，env 内部仍是 PickCube 默认的 **50 步**，而示教平均 77 步 → 策略没做完就被截断。改为 `EnvSpec(max_episode_steps=150)`（即传给 `gym.make`）后成功率 **0.1 → 0.6**（20 episodes 探路） |
| **`norm_stats` 是 `None`** | 加载 checkpoint 后无法归一化 | delta 控制模式下官方**不计算** norm stats（`train_rgbd.py:233`）→ 按官方 `evaluate.py` 的 `delta_control = not stats` 语义**跳过** state 标准化与 action 反归一化 |
| **state_dict 键前缀** | `Missing keys: "model.pos_table"` | checkpoint 存的是 `Agent.state_dict()`（带 `model.` 前缀），加载时 `removeprefix("model.")` |
| **`CPUGymWrapper` 剥掉 batch 维并转 numpy** | `ndarray has no attribute permute` / `input.dim()=3` | 观测树里的数值字段需统一补 batch 维（图像 `(H,W,C)`→`(B,H,W,C)`，向量 `(N,)`→`(B,N)`）后才与官方 `FlattenRGBDObservationWrapper` 的假设一致 |
| **长时渲染触发 Windows TDR** | 评测跑到一半 `vk::ErrorDeviceLost` | 见 `reports/B.md` §6（同一条 GPU 上的问题）；评测循环统一加 `sync_and_free` / `rebuild_env_every` |

## 8. Diffusion Policy（可扩展性 smoke test）

**口径声明**：本项目**没有**把 DP 训练成一个可用 baseline，只做了 smoke test。理由：DP 完整训练需要额外依赖（`hydra` / `omegaconf` / `zarr`，当前环境缺失）与显著更长的训练时间，而本仓库的重点是"评测口径与数据工程"，不靠堆 baseline 充数。

**实测**（ManiSkill 自带 DP 的 `ConditionalUnet1D`）：

```
python: 3.12.7   torch: 2.9.1+cu128   cuda: True
dep OK:   diffusers / einops / wandb
dep MISS: hydra / omegaconf / zarr
DP ConditionalUnet1D OK  params=1090952  in=(2, 16, 8) -> out=(2, 16, 8)
DP on cuda OK            out=(2, 16, 8)
```

**结论**：DP 的核心去噪网络在本机 CPU 与 CUDA 上均可正常前向（1.09M 参数），
说明"再上一个 baseline"在工程上是可行的，只是本次不投入。

## 9. 局限（明确声明，不作过度声称）

- **纯仿真，未做真机迁移**；不声称任何 Sim2Real 效果。
- **控制空间是关节空间**：Windows 上没有 pinocchio（IK 不可用），无法使用官方 baseline 默认的 `pd_ee_delta_pos`。策略必须自行学出逆运动学，**难度高于官方设置**，因此 0.39 这个数字不能与官方 ee 控制的 baseline 直接对比。
- 渲染走 CPU（Windows 约束），训练不受影响。
- **两个任务均为短程桌面操作**，不涉及长时程规划和复杂多阶段依赖（那需要更长 episode 与分层策略，属于后续工作）。
- **失败模式分类是规则启发式**，不是人工标注：方块位移用末端位置近似（ManiSkill 未直接暴露方块位姿），阈值见模块常量；数量级可靠，绝对边界不必过度解读。
- **每个配置的 episode 数有限**（30–150），CI 较宽；§5 已给出同 checkpoint 换 seed 块差异近 2.4 倍的直接证据。

## 10. 复现

见 `project-a-il/README.md`（含数据准备、训练、评测、泛化实验、数据量曲线、失败归因的完整命令）。
