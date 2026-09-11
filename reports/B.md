# 项目 B · 报告：语言条件 VLA 操作（SmolVLA + LeRobot）

> 所有数字来自本仓库可复现的实验，原始记录在 `experiments/`（逐 episode JSONL + 配置快照）。
> **仿真环境验证，未做真机（Sim2Real）迁移。**
> 运行环境：Windows 原生 / RTX 4060 Laptop 8 GB / `lerobot 0.6.1` + `torch 2.9.1+cu128`

## 1. 实验设置

| 项 | 值 |
|---|---|
| 任务 | `PickCube-v1`（抓取红色方块并移到目标位） |
| 观测 | **3 个相机** `camera1/2/3`（256×256，与 SmolVLA 的 `image_features` 对齐）+ 29 维 state（模型内部 pad 到 32） |
| 控制 | 关节空间 `pd_joint_delta_pos`，动作 8 维 |
| 模型 | **SmolVLA 450M**（`SmolVLM2-500M-Video-Instruct` 骨干，VLM 层裁剪至 16），`chunk_size=50`、`n_action_steps=50` |
| 数据 | 官方示教 1000 条 → **3 相机 RGB replay 100 条**（502 MB）→ `DatasetPipe` 转 LeRobotDataset（`id=local/pickcube_3cam_100`，166 MB，7400 帧） |
| 语言 | 每个 episode 绑定指令 `"pick up the red cube and move it to the goal position"` |
| 微调 | batch 8，**5000 steps**（约 1 epoch），显存 **3.57 GB**，~1.2 s/step |
| 评测 | 3 seed 块 × 10 episodes = 30，`max_episode_steps=150`，`ignore_terminations=True` |

## 2. 主结果（MVP）

| 指标 | 值 |
|---|---|
| **成功率** | **1 / 30 = 0.0333** |
| 标准误 | ±0.0328 |
| 95% 置信区间（Wilson） | **[0.006, 0.167]** |
| 失败模式 | `{'timeout': 29, 'success': 1}` |
| 平均耗时 | 5.56 s/episode |

原始记录：`experiments/B_mvp_smolvla/`（含 20 个 MP4）

## 3. A / B 同口径对照（同一 `RolloutRunner`、同一 seeds 定义、同一步数上限）

| 实验 | 成功/总数 | 成功率 | ±SE |
|---|---|---|---|
| **A** MVP（ACT，25000 steps） | 59/150 | **0.3933** | 0.0399 |
| A 泛化 baseline（seeds 10–12） | 5/30 | 0.1667 | 0.0680 |
| A 泛化 位姿范围×2 | 5/30 | 0.1667 | 0.0680 |
| A 泛化 光照变暗 / 变亮 | 1/30 / 1/30 | 0.0333 | 0.0328 |
| **B** MVP（SmolVLA，5000 steps） | 1/30 | **0.0333** | 0.0328 |
| B 场景 baseline | 0/5 | 0.0000 | 0.0000 |
| B 指令四档（模板/同义/组合/干扰） | 0/12, 0/9, 0/6, 0/6 | **0.0000** | 0.0000 |

## 4. 指令泛化（四档）

| 档位 | 指令 | 结果 |
|---|---|---|
| 固定模板 | `pick up the red cube and move it to the goal position` | 0/12 |
| 同义改写 | `grasp the red cube and place it on the target spot` | 0/9 |
| 组合指令 | `put the red cube onto the green goal marker on the table` | 0/6 |
| 干扰指令 | `there is a blue bowl on the table; move the red cube to the goal` | 0/6 |

**结论**：四档之间**没有可分辨差异**，因为**基线本身为 0**。这是一个**负结果**，不是实验失败——它说明在该数据/训练规模下，VLA 尚未学到任务，指令条件性自然无从体现。

## 5. 为什么 SmolVLA 表现远低于 ACT（原因分析）

| 因素 | ACT（项目 A） | SmolVLA（项目 B） |
|---|---|---|
| 数据规模 | 100 episodes（约 7700 帧，1 相机） | 100 episodes（7400 帧，3 相机 256×256） |
| 训练步数 | **30000 steps** | **5000 steps**（约 1 epoch） |
| 模型适配 | **任务专用**小模型（resnet18 + DETRVAE，~50M 可训） | **通用 VLA**（450M，从网络预训练权重迁移） |
| 输出空间 | 直接回归动作 chunk | 需经 FAST/flow 头 + processor 反归一化 |

**判断**：主要瓶颈是**训练规模**（5000 steps ≈ 1 epoch 远不足以让 450M 参数适应新任务）与**数据量**（7400 帧对 VLA 明显偏少，社区常用 10 万级帧）。这不是"VLA 不行"，而是**在 8 GB 消费级卡 + 单任务小数据下，VLA 的数据/算力门槛尚未跨过**——这个结论本身对岗位选择有参考价值。

## 6. 工程发现（本项目最有价值的产出之一）

### 6.1 `vk::ErrorDeviceLost` 的完整排查（Windows 平台特性）

评测阶段出现 `RuntimeError: vk::Device::waitSemaphores: ErrorDeviceLost`。排查路径：

1. **先排除显存不足**：实测模型加载后可用 6025 MiB，连续 5 步推理稳定在 5939 MiB（无泄漏）→ 不是 OOM。
2. **再排除代码 bug**：单步链路（obs → preprocessor → select_action → postprocessor）完全正常，动作范围 `[-0.571, 0.553]` 正确。
3. **定位真因**：查 Windows 事件日志发现 `nvlddmkm` **ID 153**（显示驱动已停止响应并已恢复）= **TDR**——GPU 在长时间连续「3 相机渲染 + VLA 推理」下被驱动重置。

**修复**（已进 `embodied_core/rollout/runner.py`）：`sync_and_free`（每步 `torch.cuda.synchronize()` + 每 episode `gc.collect()`/`empty_cache()`）与 `rebuild_env_every`（每 N 个 episode 关闭并重建 env，释放 SAPIEN 渲染资源）。修复后 **30 episodes 连续跑完无崩溃**，且**无需降低渲染分辨率**。

另外发现：**为每个指令变体重建模型实例会在第二档触发 TDR**（`B_instruction_eval.log`），改为复用同一实例、只切换 instruction 后四档全部跑完；**手写 rollout 循环也必须带同样的保护**（`B_scene_eval.log` 的反例）。

### 6.2 推理延迟不是瓶颈（反直觉结论）

| 指标 | 实测 |
|---|---|
| SmolVLA 策略延迟 | **mean 6.7 ms / p50 6.7 / p95 7.3 / min 5.5 / max 11.5** |
| 推算控制频率（仅策略） | **148.7 Hz** |
| 单 episode 实际耗时 | 5.56 s / 150 步 ≈ **37 ms/步** |

结论：`chunk_size=50` 的 action queue 让模型每 50 步才真正前向一次，因此**策略推理远快于环境渲染**；实时性瓶颈在仿真渲染而非 VLA 本身。

### 6.3 其他 Windows / interop 适配（详见 `docs/phase3_progress.md`）

- `--policy.path=lerobot/smolvla_base` 被 interop 改写成 `lerobot\smolvla_base` → 改用本地快照绝对路径
- 训练收尾建符号链接失败（`WinError 1314`）→ 包装脚本把 `update_last_checkpoint` 换 no-op
- 必须 `--policy.push_to_hub=false`；`torchcodec` DLL 不可用 → 自动回退 `pyav`
- 3 相机配置必须**数据生成与评测两侧一致** → 集中在 `embodied_core/envs/cameras.py`

## 7. 可交互 Demo

```cmd
cd /d E:\EmbodiedAI
:: SmolVLA（语言条件 VLA）
.venv\Scripts\python.exe scripts\demo_gradio.py --policy smolvla
:: ACT 对照（成功率更高，便于观察成功行为）
.venv\Scripts\python.exe scripts\demo_gradio.py --policy act
```

界面：自然语言指令输入（可自由改写）→ 三视角实时画面 + 状态面板（step / 夹爪开合 / 成功判定 / 单步延迟 ms / 成功条件说明）。核心组件已逐项验证（三相机 `(256,256,3)`、状态字段、8 维动作、`DEMO_COMPONENTS_OK`）。

## 8. 局限（明确声明）

- **纯仿真，未做真机迁移**；不声称任何 Sim2Real 效果。
- SmolVLA 成功率极低（0.033），因此**指令泛化与场景泛化都无法评估**（基线为 0）；项目 A 的 ACT 泛化数据保留作为对照。
- 微调仅 5000 steps / 100 episodes，未做数据 scaling 或更长训练；若继续投入，最直接的改进是**扩充 3 相机 replay 到 1000 条 + 训练 ≥ 20000 steps**（预计各需数小时）。
- `torchcodec` 不可用，走 `pyav` 回退（功能等价，性能略低）。
- 未做 KV-cache / 量化等推理优化的定量对比（当前延迟已非瓶颈，优先级下调）。

## 9. 复现

完整命令见 `project-b-vla/README.md`；进度事实源见 `docs/phase3_progress.md`；环境与依赖源见 `docs/env_decision.md`。
