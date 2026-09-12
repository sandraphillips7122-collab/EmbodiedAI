# 项目 B · 报告：语言条件 VLA 操作（SmolVLA + LeRobot）

> 所有数字来自本仓库可复现的实验，原始记录在 `experiments/`（逐 episode JSONL + 配置快照）。
> **仿真环境验证，未做真机（Sim2Real）迁移。**
> 运行环境：Windows 原生 / RTX 4060 Laptop 8 GB / `lerobot 0.6.1` + `torch 2.9.1+cu128`

## 0. 一句话结论

**SmolVLA 在本配置下的成功率与任务专用 ACT 相差一个数量级，并且在把数据量 ×5、训练步数 ×4 之后仍未改善。**
本报告的价值不在"刷高成功率"，而在：**完整的 VLA 工程链路**（数据管道 → 微调 → 统一评测 → 效率分析）+ **用控制变量实验把失败原因定位清楚**，而不是把低成功率归咎于"模型不行"。

## 1. 实验设置

| 项 | 值 |
|---|---|
| 任务 | `PickCube-v1`（抓取红色方块并移到目标位） |
| 观测 | **3 个相机** `camera1/2/3`（256×256，与 SmolVLA 的 `image_features` 对齐）+ 29 维 state（模型内部 pad 到 32） |
| 控制 | 关节空间 `pd_joint_delta_pos`，动作 8 维 |
| 模型 | **SmolVLA 450M**（`SmolVLM2-500M-Video-Instruct` 骨干，VLM 层裁剪至 16），`chunk_size=50`、`n_action_steps=50` |
| 数据 | 官方示教 1000 条 → **3 相机 RGB replay**（v1 用 100 条、v2 用 **500 条 / 5.05 GB 源 h5 → 11 GB LeRobotDataset**） |
| 语言 | 每个 episode 绑定指令 `"pick up the red cube and move it to the goal position"` |
| 微调 v1 | batch 8，**5000 steps**，显存 **3.57 GB**，~1.2 s/step |
| 微调 v2 | batch 8，**20000 steps**（3:29:19），`--num_workers 8`，显存 **3.57 GB**，~0.62 s/step |
| 评测 | 统一 `RolloutRunner`：v1 为 3×10 = 30 episodes（3 seed 块），v2 为 3×20 = **60 episodes**，`max_episode_steps=150` |

## 2. 主结果

| 版本 | 数据 | 训练步数 | 成功/总数 | 成功率 | ±SE | 95% CI (Wilson) |
|---|---|---|---|---|---|---|
| v1 | 100 eps | 5000 | 1/30 | **0.0333** | 0.0328 | [0.006, 0.167] |
| **v2** | **500 eps** | **20000** | 1/60 | **0.0167** | 0.0165 | [0.003, 0.089] |

原始记录：`experiments/B_mvp_smolvla/`（20 个 MP4）、`experiments/B_mvp_smolvla_500eps/`

**这是本报告最重要的一个数字对比**：**数据 ×5、步数 ×4，成功率没有变好**（两个 CI 高度重叠，统计上无可区分差异）。

## 3. A / B 同口径对照（同一 `RolloutRunner`、同一 seeds 定义、同一步数上限）

| 实验 | 成功/总数 | 成功率 | ±SE |
|---|---|---|---|
| **A** PickCube ACT（100 demos，30000 iters） | 59/150 | **0.3933** | 0.0399 |
| **A** StackCube ACT（100 demos，30000 iters） | 13/30 | **0.4333** | 0.0905 |
| A 数据量 50 / 100 / 200 demos（同口径 5000.pt） | 3/30 / 4/30 / 21/30 | 0.1000 / 0.1333 / **0.7000** | 0.055 / 0.062 / 0.084 |
| A 泛化 baseline（seeds 10–12） | 5/30 | 0.1667 | 0.0680 |
| A 泛化 位姿范围×2 | 5/30 | 0.1667 | 0.0680 |
| A 泛化 光照变暗 / 变亮 | 1/30 / 1/30 | 0.0333 | 0.0328 |
| **B** SmolVLA v1（100 eps，5000 steps） | 1/30 | **0.0333** | 0.0328 |
| **B** SmolVLA v2（500 eps，20000 steps） | 1/60 | **0.0167** | 0.0165 |
| B 指令四档（模板/同义/组合/干扰） | 0/12, 0/9, 0/6, 0/6 | **0.0000** | 0.0000 |
| B 场景 baseline | 0/5 | 0.0000 | 0.0000 |

## 4. 指令泛化（四档）

| 档位 | 指令 | 结果 |
|---|---|---|
| 固定模板 | `pick up the red cube and move it to the goal position` | 0/12 |
| 同义改写 | `grasp the red cube and place it on the target spot` | 0/9 |
| 组合指令 | `put the red cube onto the green goal marker on the table` | 0/6 |
| 干扰指令 | `there is a blue bowl on the table; move the red cube to the goal` | 0/6 |

**结论**：四档之间**没有可分辨差异**，因为**基线本身为 0**。这是一个**负结果**，不是实验失败——**指令条件性只有在模型已经学会任务之后才可能被观察到**。

## 5. 归因：为什么 v2 的"加大投入"没有奏效

v1 → v2 是两组控制变量同时变化的结果，因此我先**逐项排除**：

| 假设 | 检验方式 | 结论 |
|---|---|---|
| 「训练步数不够」 | 固定数据（100 eps）比 5000 vs 20000 steps | **排除**：v2 用 20000 steps 仍无提升 |
| 「数据量不够」 | 固定口径比 100 vs 500 episodes | **排除**：×5 数据量仍无提升 |
| 「显存不够被迫降级」 | 实测 batch 8 峰值 3.57 GB，未启用任何 LoRA/冻结阶梯 | **排除**：训练配置未打折 |
| 「推理太慢导致闭环跟不上」 | 实测策略延迟 6.7 ms（≈149 Hz），远快于渲染 37 ms/步 | **排除**（见 §7.2） |
| **剩余解释** | — | **预训练分布与本任务差异过大** |

**这个"剩余解释"具体是什么**（不空泛）：

1. **机器人本体不同**：SmolVLA 预训练主要是 **SO-100/SO-101 系列**（6 自由度 + 平行夹爪，动作语义与相机布局固定），我们的任务在 **Panda**（7 自由度）上，且用**关节空间增量**而不是末端执行器增量。
2. **动作空间语义不同**：8 维关节增量与预训练的动作分布差异很大，而 action expert 需要从预训练分布迁移——这比"从零学一个动作头"更困难（后者至少优化目标一致）。
3. **相机配置不同**：我们自定义的 3 个视角（外部/侧上/顶部）与预训练数据的相机几何、分辨率语义都不一致，VLM 视觉特征需要整体重新对齐。
4. **数据规模量级差距**：社区把 VLA 微调到可用通常用 **10 万级帧**；我们两版分别是 7400 与 3.7 万帧（500 eps × 74 帧），仍差一个量级。

**因此结论是**：**在 8 GB 消费级卡 + 单任务小数据 + 与预训练本体/动作空间不匹配的条件下，SmolVLA 微调无法达到可用水平**。
这不是否定 VLA 路线，而是说清楚了**它需要什么条件**——这个判断对"什么场景值得上 VLA"这个工程问题是有用的答案。

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

结论：`chunk_size=50` 的 action queue 让模型每 50 步才真正前向一次，因此**策略推理远快于环境渲染**；实时性瓶颈在仿真渲染而非 VLA 本身。这一点也说明 §5 里"闭环跟不上"的假设不成立。

### 6.3 数据管道与性能（v2 新增）

- **视频编码在消费级卡上不实用**：500 条 episode 的 3 相机视频被压进极少的大 mp4（单个 `file-000.mp4` 已 209 MB），耗时数小时仍未 finalize。
  → **改用图片模式**（`dtype="image"`，11 GB），约 40 分钟完成。
- **修复 `DatasetPipe` 的真实缺陷**：`build_features` 之前硬编码 `dtype="video"`，与 `LeRobotDataset.create(use_videos=False)` 冲突（lerobot 直接抛错）。现在 dtype 跟随开关。
- **dataloader worker 数决定训练速度**：图片模式下 `--num_workers 0` 时 **3.45 s/step（ETA 19 小时）**；改为 `--num_workers 8` 后 `data_s` 降到 0.005、**0.62 s/step（ETA 3.3 小时）**——瓶颈在数据加载而非 GPU。

### 6.4 其他 Windows / interop 适配（详见 `docs/phase3_progress.md`）

- `--policy.path=lerobot/smolvla_base` 被 interop 改写成 `lerobot\smolvla_base` → 改用本地快照绝对路径
- 训练收尾建符号链接失败（`WinError 1314`）→ 包装脚本把 `update_last_checkpoint` 换 no-op
- 必须 `--policy.push_to_hub=false`；`torchcodec` DLL 不可用 → 自动回退 `pyav`
- 3 相机配置必须**数据生成与评测两侧一致** → 集中在 `embodied_core/envs/cameras.py`
- 模型输出必须用官方 `make_pre_post_processors` 反归一化，手写会导致动作越界（实测到 2.78）

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
- **SmolVLA 成功率极低（0.0167–0.0333）**，因此**指令泛化与场景泛化都无法评估**（基线为 0）。这是诚实的负结果，不做任何"接近成功"的包装。
- **没有做 10 万帧级数据的实验**：受限于 8 GB 卡与单机时间预算，两版数据量分别为 7400 与 3.7 万帧，距社区常用量级仍差一个数量级。**因此本报告不能证明"SmolVLA 不适合此任务"，只能证明"在当前投入规模下不适用"。**
- 未尝试的候选改进（明确列出，避免夸大结论覆盖面）：更换为与预训练更匹配的动作空间（末端执行器增量）、使用 SO-100 形态的机器人、LoRA 之外的部分冻结策略、更贴近预训练相机布局的视角。
- `torchcodec` 不可用，走 `pyav` 回退（功能等价，性能略低）。
- 未做 KV-cache / 量化等推理优化的定量对比（当前延迟已非瓶颈，优先级下调）。

## 9. 复现

完整命令见 `project-b-vla/README.md`；进度事实源见 `docs/phase3_progress.md`；环境与依赖源见 `docs/env_decision.md`。
