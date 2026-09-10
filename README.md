# EmbodiedAI — 具身智能求职作品集（ManiSkill3 / LeRobot）

面向 **VLA / 机器人操作算法** 与 **仿真·数据·评测** 方向的作品集仓库。
两个互补项目共享同一套基础设施（`embodied_core/`），保证评测口径一致、结果可对照。

> **诚实性约定**：所有指标均来自本仓库可复现的实验，原始记录在 `experiments/`；
> 一律表述为「仿真环境验证」，**未做真机（Sim2Real）迁移**。

## 当前状态

| 项目 | 状态 | 主结果 |
|---|---|---|
| **A. 机器人操作模仿学习闭环**（ManiSkill3 + ACT） | ✅ **MVP 完成** | 成功率 **0.3933**（59/150，95% CI [0.319, 0.473]） |
| **B. 语言条件 VLA 操作**（SmolVLA + LeRobot） | 📋 方案就绪，待实施 | — |

### 项目 A 已完成的内容

- **数据工程**：官方 `motionplanning` 示教 1000 条 → 全量 RGB replay（418 MB）+ 可复核数据卡
- **训练**：ACT（resnet18 + DETRVAE）在 RTX 4060 8 GB 上训 30000 iters；峰值显存 3310 MiB、≈0.12 s/iter
- **严谨评测**：3 seed 块 × 50 = **150 次 rollout**，成功率 ± 标准误 + Wilson 95% 区间，逐 episode JSONL
- **泛化实验**：初始位姿范围加倍 / 光照变暗 / 光照变亮（各 30 episodes）
- **失败归因**：定位并修复 4 个直接影响结果可信度的问题（报告 §4）

📄 结果与归因 → [`reports/A_mvp.md`](reports/A_mvp.md)　🚀 一键复现 → [`project-a-il/README.md`](project-a-il/README.md)

### 关键发现（面试可讲）

| 发现 | 数据 |
|---|---|
| **光照是最脆弱的泛化维度** | baseline 0.167 → 变暗/变亮均 **0.033**（−80%） |
| 初始位姿覆盖已足够 | 位姿范围加倍：0.167 → 0.167（无变化） |
| 小样本波动不可忽视 | 同一 checkpoint：seeds 0–2 得 0.393，seeds 10–12 得 0.167 |
| **评测口径能直接改变结论** | `max_episode_steps` 未传给 env 时成功率被压到 0.1（修复后 0.6） |

## 平台约束（Windows 原生，均有实测证据）

见 [`docs/env_decision.md`](docs/env_decision.md) §8，三条必须遵守：

1. `render_backend` 必须显式设 `"cpu"` —— 默认 `"gpu"` 会让进程**原生崩溃**（exit 5、无 traceback）
2. `sim_backend` 必须 `"physx_cpu"`（Windows 无 GPU 仿真）
3. **无 pinocchio → 无 IK** → 只能用关节空间控制（`pd_joint_delta_pos`）；官方 baseline 默认的 `pd_ee_delta_pos` 不可用

## 仓库结构

```
embodied_core/          两个项目共享的基础设施（唯一允许共享的地方）
  envs/adapter.py       EnvSpec（含 Windows 硬约束校验）+ make_env
  rollout/runner.py     评测器：固定 seed、成功率±SE + Wilson CI、MP4、JSONL
  rollout/act_policy.py ActPolicy：官方 ACT checkpoint → 统一 Policy 接口（零侵入）
  metrics/recorder.py   实验记录与统计（纯标准库，可离线复核）
configs/                实验配置
docs/                   env_decision.md（环境决策）· vram_report.md（显存实测）· data_card_*.md（数据卡）
experiments/            实验产物（JSONL / MP4 / summary.json）——不入库
reports/                项目技术报告
scripts/                环境安装、数据统计、训练包装、动作探针
project-a-il/           项目 A 说明与复现
project-b-vla/          项目 B（规划中）
```

## 环境（已实测）

- **执行侧：Windows 原生**（WSL 无 Vulkan 渲染设备，不可用）
- Python 3.12.7 venv；`torch 2.9.1+cu128`；`mani_skill 3.0.1` + `sapien 3.0.3`
- 依赖源：清华 PyPI（纯 Python 包）／阿里云 `pytorch-wheels`（CUDA torch）／`gh-proxy.com`（GitHub 源码）／`hf-mirror.com`（模型与数据）

## 项目 B 的下一步（方案已定稿）

复用项目 A 的 env / 数据 / 评测框架，引入 **SmolVLA（450M）+ LoRA**：

1. `lerobot` 安装 + `lerobot/smolvla_base` 权重（约 1.4 GB 下载）
2. **S2 显存 smoke test**（不预设可训）：`freeze_vision_encoder` + 只训 action expert → LoRA → 逐级降级
3. 指令泛化（模板 / 同义 / 组合 / 干扰四档）+ 场景泛化（复用项目 A 的光照维度）
4. 推理延迟与动作分块（延迟 ↔ 成功率权衡）
5. Gradio 交互 Demo（自然语言指令 + 实时状态反馈）
