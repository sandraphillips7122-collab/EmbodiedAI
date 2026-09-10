# EmbodiedAI — 具身智能实习项目（ManiSkill3 / LeRobot）

面向 **VLA / 机器人操作算法** 与 **仿真·数据·评测** 方向实习的作品集仓库。
两个互补项目共享同一套底层基础设施（`embodied_core/`）。

> 状态：Phase 1（环境决策 + 地基）进行中。所有指标均来自真实实验，**未做真机（Sim2Real）验证**。

## 项目

| 项目 | 目录 | 对应岗位能力 |
|---|---|---|
| **A. 机器人操作模仿学习闭环**：示教数据工程、策略对比与泛化评测 | `project-a-il/` | 模仿学习、数据工程、严谨评测、失败归因 |
| **B. 语言条件 VLA 操作**：SmolVLA 高效微调、指令泛化与实时推理闭环 | `project-b-vla/` | VLA、多模态指令理解、具身闭环、推理效率 |

## 仓库结构

```
embodied_core/     两个项目共享的基础设施（唯一允许共享的地方）
  envs/            EnvAdapter：任务注册、相机配置、观测 schema、泛化扰动接口、成功判定
  data/            DatasetPipe：ManiSkill .h5 → LeRobotDataset v3、key/维度对齐、归一化统计、数据卡与校验
  rollout/         RolloutRunner：多进程 CPU env、固定 seed、MP4 录制
  metrics/         成功率 ± 标准误、失败模式标签、JSONL 落盘、图表
  config/          Hydra/OmegaConf schema 与实验 ID 规范
configs/           实验配置
docs/              env_decision.md（环境决策）、vram_report.md（显存实测）、其余技术文档
experiments/       实验产物（JSONL / 检查点 / 视频）——不入库
reports/           项目技术报告
scripts/           环境安装与可复现脚本
project-a-il/      项目 A 专属代码
project-b-vla/     项目 B 专属代码
```

## 环境（已实测确认，详见 `docs/env_decision.md`）

- **执行侧：Windows 原生**（WSL 无 Vulkan 渲染设备，不可用）。
- Python 环境：`.venv`（Python 3.12.7）。
- 仿真：ManiSkill3，**CPU sim + 多进程并行**（Windows 无 GPU sim）。
- 训练：Windows PyTorch CUDA；训练只消费离线数据，不依赖仿真器。

## 快速开始

```bash
# 环境准备（Windows PowerShell / cmd）
scripts\setup_env.bat

# 端到端最小闭环（Phase 1 目标）
.venv\Scripts\python.exe -m embodied_core.rollout --config configs/pickcube_act.yaml
```

## 诚实性约定

- 所有数字必须来自本仓库可复现的实验，落在 `experiments/`；不使用他人结果冒充。
- 一律表述为「仿真环境（ManiSkill3）验证，未做真机迁移」，不出现 Sim2Real 字样。
