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
| 「训练步数不够」 | 固定数据（100 eps）比 5000 vs 20000 steps | **在当前测试范围内未观察到改善**（20000 steps 仍无提升） |
| 「数据量不够」 | 固定口径比 100 vs 500 episodes | **在当前测试范围内未观察到改善**（×5 数据量仍无提升） |
| 「显存不够被迫降级」 | 实测 batch 8 峰值 3.57 GB，未**额外**启用 LoRA/NF4 等降级手段 | **已排除**（显存维度）—— 但见 §5.1 的审计更正 |
| 「**开环执行过长**」：`n_action_steps=50` 让 49/50 步按过时观测执行 | **控制变量实验**（§5.2）：同一 checkpoint、同一 seeds，只改 `n_action_steps ∈ {1,5,10,25,50}`，共 110 episodes | **已排除**：五个配置**全部 0 成功**；`n=1`（每步重规划）0/30 且单集耗时 77.8 s（比 n=50 慢 **21.7 倍**） |
| 「推理太慢导致闭环跟不上」 | 延迟拆分实测（§6.2）：forward ≈433 ms 尖峰 | **部分修正**：平均不慢，但每 50 步有 ≈0.43 s 尖峰；**但 §5.2 已证明即使把它消掉也无效** |
| **剩余解释** | — | **「预训练分布不匹配」+「视觉编码器全程冻结」**（§5.1）——⚠️ **尚未经控制变量实验证明，是假设而非结论** |

**这个"剩余解释"具体是什么**（不空泛）：

1. **机器人本体不同**：SmolVLA 预训练主要是 **SO-100/SO-101 系列**（6 自由度 + 平行夹爪，动作语义与相机布局固定），我们的任务在 **Panda**（7 自由度）上，且用**关节空间增量**而不是末端执行器增量。
2. **动作空间语义不同**：8 维关节增量与预训练的动作分布差异很大，而 action expert 需要从预训练分布迁移——这比"从零学一个动作头"更困难（后者至少优化目标一致）。
3. **相机配置不同**：我们自定义的 3 个视角（外部/侧上/顶部）与预训练数据的相机几何、分辨率语义都不一致，VLM 视觉特征需要整体重新对齐。
4. **数据规模量级差距**：社区把 VLA 微调到可用通常用 **10 万级帧**；我们两版分别是 7400 与 3.7 万帧（500 eps × 74 帧），仍差一个量级。

**因此结论是**：**在 8 GB 消费级卡 + 单任务小数据 + 与预训练本体/动作空间不匹配的条件下，SmolVLA 微调无法达到可用水平**。
这不是否定 VLA 路线，而是说清楚了**它需要什么条件**——这个判断对"什么场景值得上 VLA"这个工程问题是有用的答案。

### 5.1 审计更正：本次微调是"部分微调"，且视觉编码器全程冻结

> 详细审计见 [`docs/smolvla_audit.md`](../docs/smolvla_audit.md)（实测，非推断）。

本文早期版本写过"无需 LoRA/冻结降级""未启用任何冻结阶梯"。**这个表述不准确，现更正**：

| 项 | 实测（`scripts/audit_smolvla_config.py`） |
|---|---|
| 总参数 | **450,046,176** |
| 可训练参数 | **99,880,992（22.19%）** |
| 冻结参数 | 350,165,184 |
| **冻结的模块** | **`model.vlm_with_expert.vlm`（整个 VLM 主干，含视觉编码器）** |
| **可训练的模块** | `model.vlm_with_expert.lm_expert`（action expert，98.25M）+ `state_proj` + `action_in/out_proj` + `action_time_mlp_in/out` |
| `use_peft` | `false`（确实**没有**用 LoRA） |

准确的说法是：**本次微调采用了 SmolVLA 官方默认的 `freeze_vision_encoder=true` + `train_expert_only=true`，
属于"冻结 VLM + 只训 action expert"的部分微调**；我们**没有额外**启用 LoRA / NF4 等进一步降级。
v1 与 v2 两版的可训练集合**完全相同**（冻结策略由 config 决定，与数据量、步数无关）。

**这对归因的意义（把定性假设具体化）**：

第 3 条"相机配置不同"不再只是一个泛泛的说法 —— 它有**具体机制**：
**视觉编码器在训练中权重一个都没动**，也就是说它**自身无法更新视觉表征**；
但**下游 action expert 仍可能在一定程度上适配这些冻结特征**，因此**不能声称模型完全没有跨域适配机制**，
只能说「视觉表征本身未被更新」。
它只能沿用 SmolVLM 在 SO-100 相机布置下学到的视觉特征，而这些特征与我们的视角几何并不匹配。

**但必须严格区分"机制解释"与"已证明的原因"**：

- ✅ 可以说的：审计表明视觉编码器全程冻结，模型无机制适配自定义相机 —— 这是分布不匹配假设的**具体机制**
- ❌ **不能**说的："已证明成功率低是因为视觉编码器被冻结" —— 本审计**没有**做"解冻视觉编码器后重训"的对照实验

**可检验该机制的最小实验 —— 已尝试执行，结论：全量 VLM 解冻成本不可接受**

1. **配置陷阱**：只设 `--policy.freeze_vision_encoder=false` **不生效** ——
   因为 `train_expert_only=True` 会强制把整个 VLM 的 `requires_grad` 置为 False
   （`lerobot/policies/smolvla/smolvlm_with_expert.py:157`），**优先级更高**。
   证据：该次训练与 v2 的 **loss 与 grdn 在 24/24 个数据点上逐位相同**。
2. **全量解冻的算力成本不可接受**：`freeze_vision_encoder=False` + `train_expert_only=False` 时，
   单步耗时 **47.72 s/step**（vs 冻结版 1.40 s/step，**慢 34 倍**）→ 20000 步需 **约 265 小时**，
   训练在 step 100 后终止。
3. **措辞边界（重要）**：以上证明的是**「全量 VLM 解冻在当前硬件上不可行」**。
   **不能**据此说「单独解冻 vision encoder 不可行」—— 当前 LeRobot 配置**没有**该选项，
   **那个条件从未被测试到**（详见 `docs/smolvla_audit.md` §3.3 的措辞更正框）。
4. **因此**"视觉编码器冻结导致低成功率"**仍未经验证** —— 属**算力受限而未完成的验证**，
   而非"已证明无关"。后续用 LoRA 尝试参数高效的视觉适配（见 §5.3）。

详细记录见 [`docs/smolvla_audit.md`](../docs/smolvla_audit.md) §3.3
（含无效对照的存放位置 `experiments/B_unfreeze_finetune_invalid/`）。

### 5.2 控制变量实验：开环执行过长**不是**主因（已排除）

**动机**：延迟拆分（§6.2）显示 `n_action_steps=50` 意味着 50 步里只有第 1 步基于当前观测，
其余 49 步执行的是"按已过时观测规划的动作"，其间还有 ≈0.43 s 的 forward 尖峰。
一个自然假设是：**低成功率部分来自这种长时间开环执行**。

**设计**（除被测变量外全部固定）：同一 checkpoint（v2 `020000`）、同一任务（`PickCube-v1`）、
同一 seeds、同一评测器；**只**改 `policy.config.n_action_steps`。
`chunk_size` **保持不变**（模型仍一次预测 50 个动作，只改实际执行/重规划多少个），
所以这不是"换模型"，而是纯粹的**闭环执行策略**变量。

**结果**（新 exp_id：`experiments/B_replan_n{N}/`，不改动任何历史记录）

| `n_action_steps` | 成功/总数 | 成功率 | 95% CI 上限 | 平均单集耗时 |
|---|---|---|---|---|
| **1**（每步都重规划） | 0/30 | 0.0000 | 0.114 | **77.83 s** |
| 5 | 0/15 | 0.0000 | 0.204 | 17.05 s |
| 10 | 0/15 | 0.0000 | 0.204 | 9.76 s |
| 25 | 0/20 | 0.0000 | 0.161 | 4.97 s |
| **50**（原配置） | 0/30 | 0.0000 | 0.114 | **3.59 s** |

**结论**：

1. **五个配置全部 0 成功**（合计 110 episodes）。把重规划频率从 50 提到 **1（每步重规划）**
   **没有任何改善** → 「低成功率源于长时间开环执行」这一假设**不成立，已排除**。
2. 而代价极大：`n=1` 单集 77.83 s，比 `n=50` 慢 **21.7 倍**（每步都要做一次 VLA forward）。
3. 该排除**有实测支撑**，故 §5 排除表把它记为「已排除」。

**诚实标注（样本量限制）**：n=5 / n=10 各只有 15 episodes（Wilson CI 上限 ≈0.204），
n=1 / n=25 / n=50 为 20–30。严格说，本实验证明的是
**"在这个样本量下没有观察到任何改善"**，而非"两者精确相等"。
但 0/30 与 0/15 的 CI 与基线（0/30，CI 上限 0.114）**高度重合**、方向一致（全为 0），
作为排除依据是合理的。

**工程副产物**：批量连续跑 5 个配置会在第 4 个触发 **TDR**
（`vk::Device::waitSemaphores: ErrorDeviceLost`，与 §6.1 同一问题）→ 改为**单配置独立进程**运行，
该经验已补进 §6.1。

### 5.3 LoRA：视觉适配路线的尝试与边界（**未获得结果**）

**动机**：§5.1 指出"全量 VLM 解冻"成本不可接受，因此改用 LoRA 做**参数高效的视觉适配**。

**(a) 官方默认 LoRA 不含视觉模块**（读源码，非猜测）

`modeling_smolvla.py:420` `_get_default_peft_targets()` 的 target 只有：

- `model.vlm_with_expert.lm_expert.*.{q,v}_proj`（action expert 注意力）
- `model.{state_proj, action_in_proj, action_out_proj, action_time_mlp_in, action_time_mlp_out}`

**`vlm`（视觉编码器 + LLM 主干）完全不在其中** → **默认 LoRA 无法验证"视觉域适配不足"**，
它在结构上等价于"继续训 action expert，只是参数更少"。

**(b) 视觉 LoRA 在配置层面可行**（实测）

| 项 | 实测 |
|---|---|
| 插入模块 | **49 个**：vision `self_attn.{q,k,v,out}_proj` × 48 + `connector.modality_projection.proj` × 1（`lm_expert=0`） |
| 可训练参数 | **1,391,616 / 450,046,176 = 0.309%** |
| **adapter 是否真的更新** | **YES**（5 步后 32/98 个 LoRA 张量变化，最大 0.004979） |
| 峰值显存 | **908 MiB** |

**(c) 但训练吞吐极低 → 未获得结果**（原始证据，逐字）

```
Training:   0%|          | 1/400 [02:15<14:57:50, 135.01s/step]
Training:   0%|          | 2/400 [06:16<21:49:40, 197.44s/step]
Training:   1%|          | 3/400 [10:34<24:51:20, 225.39s/step]
```

| 配置 | step time |
|---|---|
| v2（冻结 VLM + 训 expert） | **1.40 s/step** |
| 全量 VLM 解冻 | 47.72 s/step |
| **视觉 LoRA（本配置）** | **135 → 225 s/step**（400 步约 **25 小时**） |

> ⚠️ **精确措辞（不做过度推断）**：**在当前 Windows + RTX 4060 8GB + LeRobot/PEFT 实现和视觉
> target 配置下，显存可接受，但训练计算效率极低，无法在合理时间预算内完成有效微调；
> 具体性能瓶颈原因未定位，因此不做进一步因果推断。**

**因此视觉 LoRA 路线未获得结果（not evaluated）** —— **不是**"被证明无效"。
已排除的简单解释：显存不足（峰值仅 908 MiB）。

**(d) 归因状态因此未变**

| 假设 | 状态 |
|---|---|
| 训练步数不足 / 数据量不足 | ⚪ **在当前测试范围内未观察到改善** |
| 显存降级 / 开环执行过长 | ✅ **已排除**（显存非瓶颈；`n_action_steps` 1→50 全 0 成功） |
| **视觉域适配不足 / 预训练分布不匹配** | ⚠️ **仍是假设** —— 两条验证路径（全量解冻、视觉 LoRA）**均未完成** |

**新增的方法论结论**：**"改配置字段"不等于"行为已改变"** —— 本项目两次踩到
（`freeze_vision_encoder=false` 被 `train_expert_only` 覆盖，loss/grdn **24/24 逐位相同**）。
任何配置变更都必须用**行为指标**（loss / 梯度 / 速度 / adapter 权重变化）验证。

详细记录与全部原始证据见 [`docs/lora_experiment.md`](../docs/lora_experiment.md)。

## 6. 工程发现（本项目最有价值的产出之一）

### 6.1 `vk::ErrorDeviceLost` 的完整排查（Windows 平台特性）

评测阶段出现 `RuntimeError: vk::Device::waitSemaphores: ErrorDeviceLost`。排查路径：

1. **先排除显存不足**：实测模型加载后可用 6025 MiB，连续 5 步推理稳定在 5939 MiB（无泄漏）→ 不是 OOM。
2. **再排除代码 bug**：单步链路（obs → preprocessor → select_action → postprocessor）完全正常，动作范围 `[-0.571, 0.553]` 正确。
3. **定位真因**：查 Windows 事件日志发现 `nvlddmkm` **ID 153**（显示驱动已停止响应并已恢复）= **TDR**——GPU 在长时间连续「3 相机渲染 + VLA 推理」下被驱动重置。

**修复**（已进 `embodied_core/rollout/runner.py`）：`sync_and_free`（每步 `torch.cuda.synchronize()` + 每 episode `gc.collect()`/`empty_cache()`）与 `rebuild_env_every`（每 N 个 episode 关闭并重建 env，释放 SAPIEN 渲染资源）。修复后 **30 episodes 连续跑完无崩溃**，且**无需降低渲染分辨率**。

另外发现：**为每个指令变体重建模型实例会在第二档触发 TDR**（`B_instruction_eval.log`），改为复用同一实例、只切换 instruction 后四档全部跑完；**手写 rollout 循环也必须带同样的保护**（`B_scene_eval.log` 的反例）。

**第三条经验**（§5.2 新增）：即使带了上述保护，**在同一个进程里连续跑多个配置**
（如 5 个 `n_action_steps` 各 15 episodes）**仍会在第 4 个触发 TDR** —— **累计渲染时长本身就是风险因素**。
**做法**：把"一次跑多个配置"改成**单配置独立进程**（每个配置启动一次 Python 进程）。
这条在 §5.2 的实验里实测踩到过一次（`replan.log` 的 `exit status 9`）。

### 6.2 延迟拆分：平均不慢，但每 50 步有一次 ≈0.43 s 的 forward 尖峰

> ⚠️ **本节已更正**。早期版本写的是"策略延迟 mean 6.7 ms ≈ 149 Hz → VLA 推理不是瓶颈"，
> 那是一个**只看中位数的错误结论**。完整拆分见 [`docs/latency_report.md`](../docs/latency_report.md)（实测）。

| 指标 | 实测（样本数） |
|---|---|
| **真正 VLA forward**（queue 空，含 10 步 flow-matching 去噪） | **≈ 433 ms**（p50 393 / p95 620，n=12） |
| 纯 action queue 出队 | **≈ 2.42 ms**（n=588） |
| `preprocessor` / `postprocessor` | 1.56 ms / 0.83 ms |
| `env.step`（仿真 + CPU 渲染） | 6.65 ms |
| **闭环平均 step latency** | **≈ 25.5 ms（≈39 Hz）** |
| 其中 policy 调用（p50 / max） | 6.54 ms / **525 ms** |
| forward 触发频率 | **每 50 步 1 次** |
| 环境占闭环比例 | 26.1% |

**原来的 6.7 ms 是什么**：它是 `policy call` 的 **p50**（实测 6.54 ms），而 p50 落在**出队步**上
（出队 2.42 ms + pre/post ≈ 6.5 ms）。**中位数把每 50 步一次、长达 0.38–0.86 s 的 forward 尖峰完全掩盖了。**

**修正后的结论（必须分两层陈述）**：

1. **平均而言**：闭环 ≈25.5 ms/step，环境占 26.1% —— 单看平均值，闭环并不慢。
2. **但每 50 步有一次 ≈0.43 s 的 forward 尖峰**（p95 0.62 s），那一步的
   **瞬时控制频率降到约 1.6–2.6 Hz**。对接触丰富的操作任务，0.4–0.6 秒的卡顿意味着
   机器人按**已经过时的观测**算出的动作序列继续盲走。

**因此**：早期"VLA 推理不是瓶颈"**只在平均意义上成立，作为工程结论是不完整的**。
准确表述是：**平均闭环 25.5 ms/step，但每 50 步含一次 ≈0.43 s 的 forward 尖峰；
是否构成瓶颈取决于任务对瞬时控制频率的容忍度，不能只用平均值下结论。**

因此它**不再**被用来排除"闭环跟不上"这一假设（§5 的排除表已同步更正）；
反过来，它给出一条**不需要重训就能检验**的方向：缩短 `n_action_steps`，
让 forward 更频繁但每次开环更短，观察成功率如何变化。

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

## 7. 语言条件性实验：一个真正需要读指令的最小任务

> 本节回应一个关键质疑：**原来的"指令四档"（§4）本质上是对同一个目标的不同说法** ——
> 场景里只有一个正确动作，模型即使完全忽略语言也可能做对，因此**不能证明 language-conditioned manipulation**。
> 本节设计并执行了一个**真正需要读指令**的最小任务。

### 7.1 两个实测约束（推翻了最初的设计）

最初的设想是"两个固定 goal 位置，指令说去左边还是右边"。**实测把这个方案否掉了**：

| 约束 | 实测证据 |
|---|---|
| **① goal 在相机里不可见** | `goal_site` 被注册进 `_hidden_objects`，ManiSkill 每次取 sensor 观测前调 `hide_visual()`（`mani_skill/envs/sapien_env.py:601`）。实测相距 **0.3 m** 的两个 goal 在三个相机上都是 **0 像素差异** |
| **② 即使可见，也不构成语言条件** | `goal_pos` 就在 `obs.extra` 里 —— 模型只要看 state 就能做对，**不需要读指令** |

**所以唯一满足定义的方案是"多个候选物体"**：场景里放两个同形状不同颜色的方块，
指令指定抓哪一个，而 **`obs.extra` 里不含"哪个是目标"**。
"红蓝方块"不是硬凑的例子，而是从"指令不可被 state 替代"这个要求**推导出来的必要条件**。

### 7.2 任务形式与"零额外示教成本"的技巧

新环境 `PickDualCube-v1`（`embodied_core/envs/dual_cube.py`，继承 `PickCubeEnv`）。
关键在于**不改颜色、只交换两个方块的初始位置**：

| 变体 | 红块 | 蓝块 | 指令 | 示教动作 |
|---|---|---|---|---|
| `red`  | **槽位 A** | 槽位 B | `pick up the red cube ...`  | 抓 A ✓ |
| `blue` | 槽位 B | **槽位 A** | `pick up the blue cube ...` | 抓 A ✓ |

现有 PickCube 示教的动作轨迹**总是抓槽位 A**，所以同一批示教在两种布局下**都成立** ——
不需要重新做 motion planning，只是**标签（语言指令）不同**。

**已验证**：两变体的三张相机图像差异显著（各相机最大像素差 **231–255**），即**视觉上可分辨**。

### 7.3 数据与训练

| 项 | 值 |
|---|---|
| 数据 | `trajectory.dual{red,blue}.rgb...h5`，各 **200/200 = 100%** replay 成功 |
| 数据集 | `local/pickcube_dualcolor_200`，**200 episodes red/blue 交替写入** |
| 为什么必须交替 | 若红蓝各一个数据集分别训练，模型只要认出"我在哪套数据里"就够了 —— **那是任务泄漏，不是语言条件性**。只有交错出现，指令才是唯一可区分信息 |
| 微调 | SmolVLA 从 `smolvla_base` 微调 **6000 steps**（1:02:59），新的 exp_id 目录 `B_langcond_finetune/` |

### 7.4 结果（2×2 析因，同 seeds、同 checkpoint）

| 场景 | 传给模型的指令 | 是否正确 | 成功/总数 | 成功率 | 失败模式 |
|---|---|---|---|---|---|
| red  | `pick up the red cube ...`  | ✅ 正确 | 1/30 | 0.0333 | `{timeout: 29, success: 1}` |
| red  | `pick up the blue cube ...` | ❌ 错误 | 1/30 | 0.0333 | `{timeout: 29, success: 1}` |
| blue | `pick up the blue cube ...` | ✅ 正确 | 1/30 | 0.0333 | `{timeout: 29, success: 1}` |
| blue | `pick up the red cube ...`  | ❌ 错误 | 1/30 | 0.0333 | `{timeout: 29, success: 1}` |

**正确指令 − 错误指令 = 0.0000（两个场景都是）。**

### 7.5 严格判读：**不能**说"已证明模型忽略语言"

这是一个**地板效应**：四格的基线都只有 **0.0333**，正确与错误指令**都接近 0**。
在这种情况下，"没有差异"既可能意味着**模型忽略语言**，也可能只是意味着
**模型还没学会这个任务，因而谈不上有条件性** —— 两者无法区分。

**因此准确的表述是**：

> 在本模型当前的能力水平（≈0.03）下，**无法检测语言条件性**。
> 本实验的价值在于**给出了判据干净的实验设计**（2×2 析因 + 明确的零假设 + 指令是唯一可区分信息），
> 而不是给出了关于语言条件性的结论。

**若要让它有判定力**，前提是先把这个任务的成功率抬到可用区间（例如通过
§9 列出的解冻视觉编码器、增加数据、或换成与预训练匹配的动作空间），
否则任何"指令差异"都会被地板效应吞掉。

### 7.6 顺带修掉的 4 个真实 bug（供复现者避坑）

| # | 现象 | 根因 |
|---|---|---|
| 1 | `TypeError: unexpected keyword argument 'target_color'` | replay 用的是 `env_info["env_id"]`（`replay_trajectory.py:485`），不是 `env_class`；改错字段会继续用 `PickCube-v1` 建环境 |
| 2 | 环境报 `BaseEnv.__init__` 不认参数 | 同上（env_id 没改成功） |
| 3 | 环境"未注册" | 必须 `import embodied_core.envs.dual_cube` 触发 `@register_env` |
| 4 | `KeyError: 'cube_b'` | replay 用示教 `env_states` 递归覆盖新环境状态（`replay_trajectory.py:199`），而 **`env_states` 存在 h5 里、不在 json 里**（json 的 `episodes[i]` 只有 `episode_id/seed/...`）→ 必须在 **h5 层**注入 `cube_b` |

另两条工程约束：**同一进程内第二次 `runpy` 会 `context has already been set`**（需分进程）；
**连续跑多个配置会触发 TDR**（见 §6.1 第三条经验）。

## 8. 可交互 Demo

```cmd
cd /d E:\EmbodiedAI
:: SmolVLA（语言条件 VLA）
.venv\Scripts\python.exe scripts\demo_gradio.py --policy smolvla
:: ACT 对照（成功率更高，便于观察成功行为）
.venv\Scripts\python.exe scripts\demo_gradio.py --policy act
```

界面：自然语言指令输入（可自由改写）→ 三视角实时画面 + 状态面板（step / 夹爪开合 / 成功判定 / 单步延迟 ms / 成功条件说明）。核心组件已逐项验证（三相机 `(256,256,3)`、状态字段、8 维动作、`DEMO_COMPONENTS_OK`）。

## 9. 局限（明确声明）

- **纯仿真，未做真机迁移**；不声称任何 Sim2Real 效果。
- **本次微调是"部分微调"**：视觉编码器全程冻结（实测可训 99.88M / 450.05M，见 `docs/smolvla_audit.md`）。因此**视觉编码器自身无法更新视觉表征**；但**下游 action expert 仍可能适配冻结特征**，所以**不能声称完全没有跨域适配机制**。这是分布不匹配假设的一个可能机制（**尚未经对照实验证明**）。
- **SmolVLA 成功率极低（0.0167–0.0333）**，因此**指令泛化、场景泛化、以及 §7 的语言条件性都无法评估**（基线接近 0）。这是诚实的负结果，不做任何"接近成功"的包装。
- **没有做 10 万帧级数据的实验**：受限于 8 GB 卡与单机时间预算，各版数据量在 7400–3.7 万帧，距社区常用量级仍差一个数量级。**因此本报告不能证明"SmolVLA 不适合此任务"，只能证明"在当前投入规模下不适用"。**
- **"视觉域适配不足"仍是假设，未获因果验证** —— 两条验证路径均未完成：全量 VLM 解冻算力不可接受（47.72 s/step）；视觉 LoRA 配置可行、显存充足（908 MiB、0.309% 参数、adapter 确认在更新），但在当前实现下吞吐极低（135–225 s/step），无法在合理预算内完成训练，**瓶颈原因未定位**（见 §5.3 与 [`docs/lora_experiment.md`](../docs/lora_experiment.md)）。
- **官方默认 SmolVLA LoRA 的 target 不含视觉模块** —— 因此若只用默认 LoRA，**不能**声称验证过"视觉域适配不足"。
- 未尝试的候选改进（明确列出，避免夸大结论覆盖面）：更换为与预训练更匹配的动作空间（末端执行器增量）、使用 SO-100 形态的机器人、更贴近预训练相机布局的视角、换更大显存以完成视觉 LoRA 训练。
- `torchcodec` 不可用，走 `pyav` 回退（功能等价，性能略低）。
- 未做 KV-cache / 量化等推理优化的定量对比（当前延迟已非瓶颈，优先级下调）。

## 10. 复现

完整命令见 `project-b-vla/README.md`；进度事实源见 `docs/phase3_progress.md`；环境与依赖源见 `docs/env_decision.md`。
训练配置审计见 `docs/smolvla_audit.md`；延迟拆分明细见 `docs/latency_report.md`；LoRA 实验见 `docs/lora_experiment.md`。
