# SmolVLA 训练配置审计（项目 B）

> **方法声明**：本文所有数字来自**实测** —— 加载真实 checkpoint 后遍历 `named_parameters()`
> 统计 `requires_grad`，而不是读文档或按默认配置推断。
> 脚本：`scripts/audit_smolvla_config.py`；原始输出：`experiments/B_smolvla_audit/audit.{txt,json}`
> （该目录不入库，可用脚本重跑）。

## 1. 为什么要做这次审计

项目 B 早期文档里有两句话：

- "450M 总参数（其中可训练 99.9M）"
- "**无需 LoRA/冻结降级**""计划中的降级阶梯**暂不启用**"

这两句话放在一起容易读成「**模型做了全参数微调**」。但直接读 checkpoint 的 `config.json`
会发现 `freeze_vision_encoder: true` 与 `train_expert_only: true` 同时存在 ——
**"我没有主动加降级阶梯"和"模型真的全参数微调了"是两回事**。

本审计用实测把这件事说清楚，避免把一个**部分微调**误当成全参数微调来陈述。

## 2. 实测结果

### 2.1 参数量总览（v1 与 v2 **数字完全相同**）

| 指标 | v1（100 eps / 5000 steps） | v2（500 eps / 20000 steps） |
|---|---|---|
| 总参数 | **450,046,176** | **450,046,176** |
| 可训练参数 | **99,880,992** | **99,880,992** |
| 冻结参数 | 350,165,184 | 350,165,184 |
| 可训占比 | **22.19%** | **22.19%** |

> 两次训练的**可训练集合完全一致** —— 因为冻结策略由 config 决定，与数据量/步数无关。

### 2.2 冻结与可训练的归属（实测）

| 模块 | 总参数 | 可训练 | 状态 |
|---|---|---|---|
| `model.vlm_with_expert` | 448,411,024 | 98,245,840 | PARTIAL |
| ├─ `model.vlm_with_expert.vlm` | — | **0** | **FROZEN（含视觉编码器）** |
| └─ `model.vlm_with_expert.lm_expert` | — | 98,245,840 | **TRAINABLE** |
| `model.state_proj` | 31,680 | 31,680 | TRAINABLE |
| `model.action_in_proj` | 23,760 | 23,760 | TRAINABLE |
| `model.action_out_proj` | 23,072 | 23,072 | TRAINABLE |
| `model.action_time_mlp_in` | 1,037,520 | 1,037,520 | TRAINABLE |
| `model.action_time_mlp_out` | 519,120 | 519,120 | TRAINABLE |

**可训练参数名前缀**（实测列出的全部）：
```
+ model.action_in_proj.{weight,bias}
+ model.action_out_proj.{weight,bias}
+ model.action_time_mlp_in.{weight,bias}
+ model.action_time_mlp_out.{weight,bias}
+ model.state_proj.{weight,bias}
+ model.vlm_with_expert.lm_expert.*        ← action expert 主体
```

**冻结参数名前缀**：
```
- model.vlm_with_expert.vlm.*             ← 整个 VLM 主干（视觉编码器 + 语言模型）
```

### 2.3 config 关键字段（直接读自 checkpoint）

| 字段 | v1 / v2 | 含义 |
|---|---|---|
| `freeze_vision_encoder` | `true` | **视觉编码器冻结** |
| `train_expert_only` | `true` | **只训 action expert** |
| `train_state_proj` | `true` | state 投影层可训 |
| `use_peft` | `false` | 确实**没有**用 LoRA |
| `chunk_size` / `n_action_steps` | `50` / `50` | 预测 50 步、开环执行 50 步 |
| `num_steps` | `10` | flow matching 去噪步数 |
| `max_state_dim` / `max_action_dim` | `32` / `32` | state/action 的 padding 目标维度 |
| `resize_imgs_with_padding` | `(512, 512)` | 输入图会被放大到 512×512 |
| `num_vlm_layers` | `16` | VLM 层数（从 500M 骨干裁剪） |
| `vlm_model_name` | `SmolVLM2-500M-Video-Instruct` | 骨干 |
| `use_cache` | `true` | KV cache 开启 |
| `attention_mode` | `cross_attn` | action expert 与 VLM 的交互方式 |

## 3. 这次审计改变了什么

### 3.1 需要修正的表述

| 原表述 | 实际问题 | 修正后 |
|---|---|---|
| "无需 LoRA/**冻结**降级" | 确有冻结，且是**官方默认**行为 | "采用 SmolVLA 官方默认的 `freeze_vision_encoder` + `train_expert_only`（**部分微调**）；未额外启用 LoRA/NF4" |
| "降级阶梯暂不启用" | 严格说只对"**额外**降级"成立 | 同上；并**补上实测数字** 99.88M / 450.05M |

> **关键**：这不是"偷懒降级"，而是 **SmolVLA 官方建议的微调方式**（`smolvla_base` 的 config 本身
> 就带这两个 `true`）。但把它说成"未启用冻结"是不准确的，必须改。

### 3.2 对"为什么成功率低"的归因意义（重要）

审计给出了一条**有机制解释的线索**：

> **视觉编码器被冻结** → 模型看到的图像特征**永远是 SmolVLM 预训练时学到的特征**，
> 而预训练数据是 **SO-100** 的相机布置与视角。
> 我们的 **3 个自定义相机**（外部/侧上/顶部，256×256）与它**几何与语义都不一致**，
> 却**无法由视觉编码器自身更新视觉表征** —— 因为 `vlm`（含 vision encoder）的权重在训练中一个都没动。
>
> ⚠️ 但要注意边界：**下游 action expert 仍可能在一定程度上适配这些冻结特征**，
> 因此**不能声称模型完全没有跨域适配机制**，只能说「视觉表征本身未被更新」。

这**具体化了**之前只有定性描述的"预训练分布不匹配"假设，并且给出一个**可直接检验的方向**：

```
对照实验：freeze_vision_encoder=False（解冻视觉编码器）重训一次，
         其余（数据、步数、seed）保持一致，比较成功率。
```

### 3.3 该对照实验已尝试执行 —— 结果：**全量 VLM 解冻在 8 GB 卡上成本不可接受**

我实际跑了这个对照，过程中暴露两个必须记录的事实。

> ⚠️ **措辞更正（重要，避免过度声称）**
>
> 本实验证明的是：**`train_expert_only=False` 会把整个 VLM 主干（448M 参数）解冻**，
> 单步耗时升到约 **47.72 s**，因此
> **「全量 VLM 解冻在当前硬件上计算成本不可接受」**。
>
> 它**不能**被表述为「已证明单独解冻 vision encoder 不可行」——
> 因为**当前 LeRobot 的 SmolVLA 配置不提供"只解冻 vision tower"的选项**
> （`train_expert_only` 是整体开关，`freeze_vision_encoder` 又被它覆盖），
> **该条件从未被测试到**。两者是不同强度的结论。

**(a) 配置陷阱：改字段 ≠ 改行为**

只设 `--policy.freeze_vision_encoder=false` 时，checkpoint 的 `config.json` 里**确实**写着
`freeze_vision_encoder=False`，看起来"生效了" —— 但训练行为**完全没变**。

根因（`lerobot/policies/smolvla/smolvlm_with_expert.py:157`）：

```python
if self.train_expert_only:
    self.vlm.eval()
    for params in self.vlm.parameters():
        params.requires_grad = False     # ← 强制冻结整个 VLM，优先级更高
```

**铁证**：该次训练与 v2（冻结）的 **loss 与 grdn 在 24/24 个数据点上逐位相同**，
包括末点 `step 20000: loss 0.0460 / grdn 0.959`。

→ **要点**：想解冻视觉编码器，必须**同时**关闭 `train_expert_only`。

**(b) 真正解冻后：算力不可行**

`--policy.freeze_vision_encoder=false --policy.train_expert_only=false`（配置已由日志确认）：

| 配置 | 单步耗时 | 20000 步预估 |
|---|---|---|
| frozen（v2，`expert_only=True`） | **1.40 s/step** | 约 3.5 小时 |
| 无效解冻（`freeze=False, expert_only=True`） | 1.31 s/step | 约 3.5 小时 |
| **真正解冻（`freeze=False, expert_only=False`）** | **47.72 s/step** | **约 265 小时** ❌ |

**慢 34 倍** —— 因为 `train_expert_only=False` 解冻的是**整个 VLM 主干（448M 参数）**，
反向传播要穿过 16 层 VLM 与 3×512×512 视觉编码器的全部激活。RTX 4060 Laptop 8 GB 扛不住，
训练在 step 100 后被终止（无效对照的 checkpoint 保留在
`experiments/B_unfreeze_finetune_invalid/`，作为该配置陷阱的证据）。

**lerobot 的 API 没有"只解冻视觉编码器、不解冻 LLM"的选项**（`train_expert_only` 是整体开关）。
若要走这条路，需要改库代码，或改用 LoRA / 更大显存。

**(c) 因此该因素的最终状态（诚实标注）**

| 说法 | 是否成立 |
|---|---|
| ✅ "审计表明视觉编码器全程冻结，模型无机制适配自定义相机" | 成立（实测参数量） |
| ✅ "**全量 VLM 解冻**在 8 GB 消费级卡上**计算成本不可接受**（47.72 s/step）" | 成立（实测） |
| ⚠️ "单独解冻 **vision tower** 不可行" | **未测试** —— 当前 LeRobot 配置**不提供**该选项 |
| ❌ "已证明成功率低是因为视觉编码器被冻结" | **仍未验证** |

注意第 2 行与第 3 行的区别：我们测的是**整体解冻**（`train_expert_only=False`），
而**"只解冻 vision tower、冻结 LLM"这个条件从未被执行** ——
不能把前者的实测结果当成对后者的结论。

**可行的替代路径**（`LoRA` 一条会在下文 `docs/lora_experiment.md` 中实测）：
用 LoRA 只训低秩适配器、换更大显存的卡、或修改 lerobot 以单独解冻 vision tower。


## 4. 附带排除的一个疑似问题

审计中发现两个 config 都写着：

```json
"observation.state": { "type": "STATE", "shape": [6] }
```

而我们的 state 实际是 **29 维**。**这看似是"state 被截断成 6 维"的严重 bug** —— 若成立，
则 `goal_pos`（目标位置）、`is_grasped`（是否已抓取）等信息根本没进模型，成功率必然极低。

**查证结论：不是 bug，已排除。** 依据 `lerobot/policies/smolvla/modeling_smolvla.py:409`：

```python
def prepare_state(self, batch):
    state = batch[STATE]                                   # 实际传入的 29 维
    state = pad_vector(state, self.config.max_state_dim)   # 只 pad 到 32，不截断
```

- 前向用的是 `batch[STATE]`（运行时实际张量，29 维），**不看 config 里声明的 shape**；
- `pad_vector` 是 **padding**，不做截断；
- 训练与推理走的是**同一个函数**，因此不存在"训练 6 维 / 推理 29 维"这类不一致。

所以 `shape: [6]` 只是**从 `smolvla_base` 继承下来、未被更新的元数据**（该预训练模型是 SO-100 的
6 维 state）。它会让 `config.json` 看起来可疑，但不影响计算。**记录在此，避免下次重复排查。**

## 5. 复现

```cmd
cd /d <你的仓库根目录>
:: 默认审计两个 checkpoint（v1 与 v2）
.venv\Scripts\python.exe scripts\audit_smolvla_config.py

:: 指定 checkpoint + 落盘 json
.venv\Scripts\python.exe scripts\audit_smolvla_config.py --ckpt <dir> --json out.json
```

## 6. 相关

- `docs/s2_smolvla_smoke.md` — 显存实测（450M 参数、batch 8 峰值 3654 MiB）
- `reports/B.md` — 项目 B 完整报告（§5 归因需结合本文第 3.2 节阅读）
- `docs/latency_report.md` — 延迟拆分实测（修正"6.7 ms ≈ 149 Hz"的表述）
