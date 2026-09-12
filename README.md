# EmbodiedAI — 具身智能求职作品集（ManiSkill3 + LeRobot）

面向 **VLA / 机器人操作算法** 与 **仿真·数据·评测** 方向的作品集仓库。
两个互补项目共享同一套基础设施（`embodied_core/`），保证评测口径一致、结果可对照。

> **诚实性约定**：所有指标均来自本仓库可复现的实验，原始记录在 `experiments/`；
> 一律表述为「仿真环境验证」，**未做真机（Sim2Real）迁移**。
>
> **可核对性**：每个对外汇报的数字，都能用**独立代码**从逐 episode 记录复算出来
> （见下方「如何验证这些结论」）。

## 当前状态

| 项目 | 状态 | 主结果 |
|---|---|---|
| **A. 机器人操作模仿学习闭环**（ManiSkill3 + ACT） | ✅ 完成 | PickCube 成功率 **0.3933**（59/150，95% CI [0.319, 0.473]）、StackCube **0.4333** |
| **B. 语言条件 VLA 操作**（SmolVLA + LeRobot） | ✅ 全链路跑通 | SmolVLA 成功率 **0.0167**（500 eps / 20000 steps）；**推理延迟不是瓶颈**（6.7 ms） |

**两个项目用同一个 `RolloutRunner`、同一套 seed 定义与统计口径**，因此结果可直接对照——这也是它们能放在一起讲的前提。

### 项目 A（模仿学习 · 数据工程 · 严谨评测）

- **数据**：官方 `motionplanning` 示教 1000 条 → 全量 RGB replay（418 MB）+ 可复核数据卡（轨迹数/长度分布/动作统计）
- **训练**：ACT（resnet18 + DETRVAE）30000 iters；峰值显存 **3310 MiB**、≈0.12 s/iter（RTX 4060 8 GB）
- **评测**：3 seed 块 × 50 = **150 次 rollout**，成功率 ± 标准误 + Wilson 95% 区间，逐 episode JSONL + MP4
- **泛化**：初始位姿范围加倍 / 光照变暗 / 光照变亮（各 30 episodes）
- **数据量曲线**：50 / 100 / 200 demos → 0.10 / 0.13 / **0.70**（阈值效应）
- **失败归因**：细分为 slip / not_reached / collision / no_contact；主导模式是 **slip（60%）**

### 项目 B（VLA · 语言条件 · 具身闭环 · 推理效率）

- **数据管道**：`DatasetPipe` 把 ManiSkill `.h5` 转成 LeRobotDataset（3 相机 256×256 + 29 维 state + 8 维 action + **语言指令**）
- **关键工程**：3 相机视角配置集中一处（`envs/cameras.py`），保证**训练与评测观测一致**
- **微调**：SmolVLA 450M（`lerobot`），显存 **3.57 GB**；官方默认的**部分微调**（实测可训 **99.88M / 450.05M**，**视觉编码器全程冻结**，见 `docs/smolvla_audit.md`）
- **延迟拆分**：真正 VLA forward **≈433 ms**（原来的"6.7 ms"其实是出队步的 p50）；闭环平均 25.5 ms，但**每 50 步含一次 ≈0.43 s 的尖峰**（见 `docs/latency_report.md`）
- **控制变量实验**：`n_action_steps` ∈ {1,5,10,25,50} **全部 0 成功**（110 episodes）→ **排除**"开环执行过长"这一归因
- **语言条件性**：新建 `PickDualCube-v1`（两个候选物体，**指令不可被 state 替代**）+ **2×2 析因评测** → 四格同分（**地板效应，无法检测**，如实记录）
- **评测**：与项目 A 同口径；**指令泛化四档**（模板/同义/组合/干扰）、场景泛化
- **效率**：闭环平均 step latency **mean 6.7 ms / p95 7.3 ms**（含 action queue 出队，**不等于** VLA forward 延迟，见 `docs/latency_report.md`）；action chunk 生效时 **VLA 推理不是瓶颈**
- **Demo**：`scripts/demo_gradio.py`（自然语言输入 + 三视角画面 + 状态面板，`smolvla`/`act` 双策略切换）

📄 报告 → [`reports/A_mvp.md`](reports/A_mvp.md)｜[`reports/B.md`](reports/B.md)
🚀 复现 → [`project-a-il/README.md`](project-a-il/README.md)｜[`project-b-vla/README.md`](project-b-vla/README.md)

## A / B 同口径对照（同一评测器、同一 seeds、同一步数上限）

| 实验 | 成功/总数 | 成功率 | ±SE |
|---|---|---|---|
| **A** PickCube ACT（100 demos，30000 iters） | 59/150 | **0.3933** | 0.0399 |
| **A** StackCube ACT（100 demos，30000 iters） | 13/30 | **0.4333** | 0.0905 |
| A 数据量 50 / 100 / 200 demos（同口径） | 3/30, 4/30, 21/30 | 0.1000 / 0.1333 / **0.7000** | 0.055 / 0.062 / 0.084 |
| A 泛化 baseline（seeds 10–12） | 5/30 | 0.1667 | 0.0680 |
| A 泛化 位姿范围×2 | 5/30 | 0.1667 | 0.0680 |
| A 泛化 光照变暗 / 变亮 | 1/30 / 1/30 | 0.0333 | 0.0328 |
| **B** SmolVLA（100 eps，5000 steps） | 1/30 | **0.0333** | 0.0328 |
| **B** SmolVLA（500 eps，20000 steps） | 1/60 | **0.0167** | 0.0165 |
| B 指令四档（模板/同义/组合/干扰） | 0/12, 0/9, 0/6, 0/6 | 0.0000 | 0.0000 |

## 关键发现（面试可讲）

| 发现 | 数据 |
|---|---|
| **光照是最脆弱的泛化维度**（项目 A） | baseline 0.167 → 变暗/变亮均 **0.033**（−80%） |
| 初始位姿覆盖已足够 | 位姿范围加倍：0.167 → 0.167（无变化） |
| 小样本波动不可忽视 | 同一 checkpoint：seeds 0–2 得 0.393，seeds 10–12 得 0.167 |
| **评测口径能直接改变结论** | `max_episode_steps` 未传给 env 时成功率被压到 0.1（修复后 0.6） |
| **VLA 推理不是实时性瓶颈** | SmolVLA 策略延迟 6.7 ms（≈149 Hz），瓶颈在环境渲染（37 ms/步） |
| **小数据下 VLA 远不如任务专用 IL** | ACT 0.3933 vs SmolVLA 0.0167；且 VLA 数据 ×5、步数 ×4 后**仍无改善** |
| **数据量存在阈值效应** | ACT 在 50/100/200 demos 上：0.10 / 0.13 / **0.70** |
| **主导失败模式是抓取滑落** | 20 episodes：`slip` 12（60%）> `not_reached` 4 > `collision` 1 |
| **Windows 上长时渲染会触发 TDR** | `vk::ErrorDeviceLost` ← `nvlddmkm` ID 153；用同步/释放/定期重建 env 解决（详见 `reports/B.md` §6） |

## 平台约束（Windows 原生，均有实测证据）

见 [`docs/env_decision.md`](docs/env_decision.md) §8：

1. `render_backend` 必须显式设 `"cpu"` —— 默认 `"gpu"` 会让进程**原生崩溃**（exit 5、无 traceback）
2. `sim_backend` 必须 `"physx_cpu"`（Windows 无 GPU 仿真）
3. **无 pinocchio → 无 IK** → 只能用关节空间控制（`pd_joint_delta_pos`）；官方 baseline 默认的 `pd_ee_delta_pos` 不可用
4. **长时渲染 + GPU 推理会触发 TDR**（`nvlddmkm` ID 153）→ 评测循环必须带 `sync_and_free` / `rebuild_env_every`

## 仓库结构

```
embodied_core/          两个项目共享的基础设施（唯一允许共享的地方）
  envs/adapter.py       EnvSpec（含 Windows 硬约束校验）+ make_env(cameras=...)
  envs/cameras.py       3 相机配置（数据生成与评测共用，保证观测一致）
  data/pipeline.py      DatasetPipe：ManiSkill .h5 → LeRobotDataset
  rollout/runner.py     评测器：固定 seed、成功率±SE + Wilson CI、MP4、JSONL、TDR 保护
  rollout/act_policy.py ActPolicy：官方 ACT checkpoint → 统一 Policy 接口
  rollout/smolvla_policy.py SmolVLA → 统一 Policy 接口（含官方 pre/post processor 反归一化）
  metrics/recorder.py   实验记录与统计（纯标准库，可离线复核）
  metrics/failure_modes.py 失败模式细粒度归因（slip / not_reached / collision / no_contact）
configs/                实验配置
docs/                   env_decision.md · vram_report.md · s2_smolvla_smoke.md · phase3_progress.md · testing.md · data_card_*.md
experiments/            实验产物（JSONL / MP4 / summary.json）——不入库
reports/                A_mvp.md · B.md
scripts/                环境与数据脚本、训练包装、动作探针、Gradio Demo
tests/                  pytest：独立复算 + 单元测试 + 端到端（见 docs/testing.md）
project-a-il/           项目 A 说明与复现
project-b-vla/          项目 B 说明与复现
```

## 如何验证这些结论

> 完整的验证设计见 [`docs/testing.md`](docs/testing.md)。核心思路：
> **每个对外汇报的数字，都能用独立代码从原始记录复算出来。**

```cmd
cd /d E:\EmbodiedAI

:: 1) 独立复算 22 个实验（刻意不 import 评测器，另写一套实现交叉验证）
.venv\Scripts\python.exe tests\test_recorded_experiments.py

:: 2) 全部单元测试（Wilson 边界 / seed 确定性 / state 展平 / 失败模式分支）
.venv\Scripts\python.exe -m pytest -q

:: 3) 端到端真跑一遍（需要仿真器，约 10 秒；默认跳过）
.venv\Scripts\python.exe -m pytest -m slow --runslow tests/test_end_to_end_slow.py -v
```

**第 1 步最关键**：它用**另一套独立实现**重算成功率、标准误、置信区间与失败模式分布，
再与 `summary.json` 逐字段比对。两者一致 → 数字可被外部复核；
不一致 → 会直接打印差异。

## 环境（已实测）

- **执行侧：Windows 原生**（WSL 无 Vulkan 渲染设备，不可用）
- Python 3.12.7 venv；`torch 2.9.1+cu128`；`mani_skill 3.0.1` + `sapien 3.0.3`；`lerobot 0.6.1`
- 依赖源：清华 PyPI（纯 Python 包）／阿里云 `pytorch-wheels`（CUDA torch）／`gh-proxy.com`（GitHub 源码）／`hf-mirror.com`（模型与数据）

## 数据与算力预算（实测）

| 环节 | 实测消耗 |
|---|---|
| 示教 replay（1 相机 1000 条） | 21 分钟，418 MB |
| 示教 replay（3 相机 1000 条） | 1 小时 40 分，5.05 GB |
| DatasetPipe 转换（500 条，图片模式） | 40 分钟，11 GB |
| ACT 训练（100 demos，30000 iters） | 1 小时，峰值 3310 MiB |
| SmolVLA 微调（500 eps，20000 steps） | 3 小时 29 分，峰值 3.57 GB |
| 评测（60 episodes + MP4） | 约 40 分钟 |

**结论**：全部实验可在单张 8 GB 消费级 GPU + Windows 原生环境下完成，无需云算力。
