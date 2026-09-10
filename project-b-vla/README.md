# 项目 B · 语言条件 VLA 操作（SmolVLA + LeRobot）

**目标**：证明「VLA 微调 → 多模态指令理解 → 具身闭环 → 推理效率」能力。
复用项目 A 的 env / 数据 / 评测框架（`embodied_core/`），保证两个项目结果可直接对照。

## 已完成

| 项 | 状态 |
|---|---|
| `lerobot 0.6.1` + `lerobot[smolvla]` 安装 | ✅ torch 未被改动（仍 `2.9.1+cu128`，CUDA 可用） |
| `lerobot/smolvla_base` 权重下载 | ✅ HF cache（走 `hf-mirror.com`） |
| **S2 显存 smoke test** | ✅ **通过**：450M 参数、加载 883 MiB、batch=8 峰值 **3654 MiB** → **8GB 可直接训练，无需 LoRA/冻结降级**（详见 [`docs/s2_smolvla_smoke.md`](../docs/s2_smolvla_smoke.md)） |

## 待实施（按依赖顺序）

### 1. 解决「相机数量不匹配」（当前最大技术障碍）

SmolVLA 的 `image_features` 固定为 **`camera1` / `camera2` / `camera3`** 三个视角，而
`PickCube-v1` 默认只暴露 1 个相机（`base_camera`，128×128）。三条可选路线：

| 方案 | 做法 | 评价 |
|---|---|---|
| **A（推荐）** | 给任务配置 3 个不同视角相机（外部固定 / 腕部 / 侧视），**重新 replay 100 条**（约 6–10 分钟） | 数据更有价值，也顺带支持「视角泛化」实验 |
| B | 复制同一相机 3 次 | 快，但模型看到 3 个相同视图，学不到多视角信息 |
| C | 改 SmolVLA config 的 `image_features` 为 1 个相机 | 需确认 `from_pretrained` 后能否安全改配置，且偏离官方预设 |

ManiSkill 侧通过 env 的 `sensor_configs` 增加相机；replay 时需确认 `replay_trajectory.py`
是否支持相机相关参数（否则在 `trajectory.json` 的 `env_info.env_kwargs` 里补 `sensor_configs`）。

### 2. `DatasetPipe`：ManiSkill `.h5` → LeRobotDataset

在 `embodied_core/data/pipeline.py` 实现（两个项目共享的基础设施）：

- 读 replay 产物 `.h5`：每 episode 含 `obs.sensor_data.{camera}.rgb`、`actions`、`env_states`
- 映射到 LeRobot 约定：
  - `observation.images.camera1/2/3`（`(3, 256, 256)`，SmolVLA 期望的分辨率）
  - `observation.state`（SmolVLA `max_state_dim=32`，我们的 29 维需补齐/记录）
  - `action`（`chunk_size=50` × `max_action_dim=32`）
  - `task`：**语言指令字符串**（每个 episode 绑定一条，这是「语言条件」的关键字段）
- 用 `LeRobotDataset.create(...)` 落盘，并**校验 `meta.json` 的 features 与 SmolVLA 期望一致**（写断言）

### 3. 微调

```cmd
lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=<local_dataset> ^
  --batch_size=8 --steps=10000 --save_freq=2000
```

显存已实测够用（batch 8 → 3654 MiB）；若加到 3 个相机 256×256，**需要重跑一次显存扫描**
（本仓库的 S2 smoke 脚本可直接复用）。

### 4. 指令泛化（四档）

模板 / 同义改写 / 组合指令（"把红色方块放进左边"）/ 干扰指令（描述无关物体）——
复用 `embodied_core.rollout.runner`，在 `Policy.predict(obs, instruction)` 里把指令送入
SmolVLA 的 tokenizer（注意官方要求指令**以换行符结尾**）。

### 5. 推理效率

动作分块（`n_action_steps=50`）、KV-cache、延迟 ↔ 成功率曲线；
与项目 A 的 ACT 基线对照，给出「什么条件下值得上 VLA」的结论。

### 6. Gradio 交互 Demo

自然语言输入 + **实时状态反馈**（当前子任务 / 夹爪开合 / 成功判定 / 单步延迟 / 累计成功率）。

## 关键参考

- [`docs/s2_smolvla_smoke.md`](../docs/s2_smolvla_smoke.md) — SmolVLA 显存实测与规格
- [`docs/env_decision.md`](../docs/env_decision.md) — Windows 平台约束与依赖镜像源
- [`reports/A_mvp.md`](../reports/A_mvp.md) — 共享评测框架（`RolloutRunner`）的用法与口径
- `embodied_core/rollout/act_policy.py` — 项目 A 的策略适配器，可作为 `SmolVLAPolicy` 适配器的模板
  （同样需要：obs 转换、维度对齐、动作后处理）
