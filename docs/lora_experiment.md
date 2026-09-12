# LoRA 实验（项目 B · 最后一轮算法实验）

> 目的：检验剩余核心假设「**视觉域适配不足**」。
> 本文档为**最终版**；所有数字来自实测，**负结果与原始证据全部保留**。

## 0. 为什么要做这一轮

项目 B 的前几轮诊断（配置审计、延迟拆分、`n_action_steps` 控制变量、全量解冻）**全部是排除性结果**，
原因被收敛到：

- **「预训练分布不匹配」+「视觉编码器全程冻结」**

而"全量 VLM 解冻"这条验证路径**在当前硬件上成本不可接受**（实测 47.72 s/step，见
[`smolvla_audit.md`](smolvla_audit.md) §3.3）。

⚠️ **措辞边界（重要）**：那次实验证明的是 **`train_expert_only=False` 导致整个 VLM 主干被解冻**
（448M 参数），因此「**全量 VLM 解冻在 8 GB 卡上计算成本不可接受**」。
它**不能**被表述为"已证明单独解冻 vision encoder 不可行" —— 当前 LeRobot 配置**没有**
"只解冻 vision tower"这一选项，**该条件从未被测试到**。

**LoRA 因此被选为检验该假设的路径**：只需训练低秩适配器，显存开销小，
且可通过 `target_modules` **精确指定**适配哪些模块 —— 包括**视觉相关模块**。

## 1. 官方默认 LoRA 的 target 是什么（读源码，非猜测）

`lerobot/policies/smolvla/modeling_smolvla.py:420` `_get_default_peft_targets()`：

```python
common_projections = (
    "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
)
target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
```

**官方默认只给这些模块挂 LoRA**：

| 类别 | 模块 |
|---|---|
| action expert 的注意力 | `model.vlm_with_expert.lm_expert.*.{q,v}_proj` |
| 投影层 | `model.state_proj` / `action_in_proj` / `action_out_proj` / `action_time_mlp_in` / `action_time_mlp_out` |

### ⭐ 核心结论（本轮最重要的产出）

> **`vlm`（视觉编码器 + LLM 主干）完全不在官方默认 target 里。**

因此**官方默认 LoRA 在结构上根本不会触碰视觉分支** ——
它等价于"继续训 action expert，只是参数更少"。

| | |
|---|---|
| 默认 LoRA **能**验证 | "参数高效地继续训 action expert 是否够用" |
| 默认 LoRA **不能**验证 | **"视觉域适配不足"** —— 它根本不触及视觉模块 |

→ **没有必要为了一个不能验证核心视觉假设的 baseline 再投入训练时间。**

## 2. 真实 module tree（用于确定视觉 target 的模块名）

> ⚠️ 模块名必须从**当前安装版本**的 `named_modules()` 读出，不能靠模型结构名称猜。
> 脚本：`scripts/inspect_smolvla_modules.py`（本仓库实测输出）

**视觉相关结构（实测）**：

```
model.vlm_with_expert.vlm.model.vision_model                  SmolVLMVisionTransformer
  .embeddings.patch_embedding                                 Conv2d     590,592
  .embeddings.position_embedding                              Embedding  786,432
  .encoder.layers.{0..11}.self_attn.{q,k,v,out}_proj          Linear     各 590,592
  .encoder.layers.{0..11}.mlp.{fc1,fc2}                       Linear
model.vlm_with_expert.vlm.model.connector.modality_projection  SmolVLMSimpleMLP（内部有 .proj: Linear）
```

**实测统计**：

| 类型 | 数量 | 总参数 |
|---|---|---|
| vision `self_attn` 的 q/k/v/out_proj | 12 层 × 4 = **48 个 Linear** | **28,348,416** |
| vision `mlp` 的 fc1/fc2 | 12 层 × 2 | 56,669,184 |
| `connector.modality_projection` | 2 | 11,796,480 |
| `patch_embedding` | 1 | 590,592 |

**实测踩到一个真实坑**：`connector.modality_projection` **本身是 `SmolVLMSimpleMLP` 而不是 `nn.Linear`**，
而 peft 只接受 `Linear`/`Conv`/`Embedding`/`MultiheadAttention` 作为 target。写成
`...modality_projection` 会直接报：

```
ValueError: Target module SmolVLMSimpleMLP(...) is not supported.
```

→ 必须写成 **`...modality_projection.proj`**（它内部那个 `Linear`）。
**这个坑不实测发现不了。**

**本实验实际使用的 target 正则**（只针对视觉，不含 LLM 主干）：

```python
r"(model\.vlm_with_expert\.vlm\.model\.vision_model\.encoder\.layers\.\d+\.self_attn\.(q|k|v|out)_proj"
r"|model\.vlm_with_expert\.vlm\.model\.connector\.modality_projection\.proj)"
```

**配置机制**（读源码确认）：

- `--peft.target_modules` 接受**正则字符串**（`lerobot/configs/default.py:137`）
- `lerobot/policies/pretrained.py:506-509`：CLI 取值**完全覆盖**默认值（不是追加）
  → 这正是"精确只给视觉模块加 LoRA"所需要的
- `pretrained.py:415-422`：先 `p.requires_grad_(False)` 全量冻结，**再** `get_peft_model(...)` 注入

## 3. LoRA smoke test（✅ 全部通过）

> 脚本：`scripts/lora_smoke.py`。依赖：`peft 0.20.0`（实验前未安装，已装入 `.venv`）。
> 配置：`LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, target_modules=<§2 的正则>)`。

**实测输出**：

| 检查项 | 实测结果 |
|---|---|
| **LoRA 插入模块数** | **49 个** |
| 分类 | `vision_model=48`、`connector=1`、**`lm_expert=0`**、其他投影=0 |
| 具体位置 | `...vlm.model.vision_model.encoder.layers.{0..11}.self_attn.{q,k,v,out}_proj`（48）+ `...connector.modality_projection.proj`（1） |
| **可训练张量数** | **98** |
| **可训练参数** | **1,391,616 / 450,046,176 = 0.309%** |
| 视觉 LoRA 是否可训 | **YES** |
| **adapter 是否真的更新** | **YES** —— 5 步 optimizer step 后 **32/98** 个 LoRA 张量发生变化，最大变化幅度 **0.004979** |
| 代理 loss | 0.020868 → 0.019747 → 0.018689 → 0.017694 → 0.016760 |
| **峰值显存** | **908 MiB** |

**这一步的意义**：**排除了上一轮踩过的"配置改了但行为没变"风险** ——
不仅确认插入位置与可训练参数，还**实测确认 LoRA 权重确实在更新**。

## 4. 训练测速（❌ 负结果，原始证据保留）

**背景**：smoke test 通过后按止损流程启动 8000 步训练（`experiments/B_lora_vision_finetune/`）。
但长时间无 loss 输出（`log_freq=200`），为区分"慢"与"卡"，改为 **400 步、`log_freq=20`**
的测速运行（`experiments/B_lora_speedtest/`）。

**实测 tqdm 输出（原始证据，逐字）**：

```
Training:   0%|          | 1/400 [02:15<14:57:50, 135.01s/step]
Training:   0%|          | 2/400 [06:16<21:49:40, 197.44s/step]
Training:   1%|          | 3/400 [10:34<24:51:20, 225.39s/step]
```

| 配置 | step time | 400 步预估 |
|---|---|---|
| v2（冻结 VLM + 训 expert） | **1.40 s/step** | ~9 分钟 |
| 全量 VLM 解冻 | 47.72 s/step | ~5.3 小时 |
| **视觉 LoRA（本配置）** | **135 → 225 s/step** | **~25 小时** ❌ |

**现象**：既不是首次预热（第 2、3 步没有变快，反而变慢），也不是显存不足
（smoke test 峰值仅 908 MiB）。测量期间 GPU 利用率持续 100%，说明**计算确实在进行**，
只是**吞吐极低**。

### ⚠️ 精确措辞（不做过度推断）

> **在当前 Windows + RTX 4060 8GB + LeRobot/PEFT 实现和视觉 target 配置下，
> 显存可接受，但训练计算效率极低，无法在合理时间预算内完成有效微调；
> 具体性能瓶颈原因未定位，因此不做进一步因果推断。**

**因此**：本实验**没有**产生可用的视觉-LoRA checkpoint，
**也**没有产生任何成功率数字 —— 视觉 LoRA 路线**未获得结果（not evaluated）**，
而不是"被证明无效"。

**已排除的简单解释**：显存不足（实测 908 MiB 峰值）。
**未定位的可能原因**（仅列出，不给结论）：peft 的 LoRA 包装是否影响 `use_cache` /
融合注意力路径；`train_expert_only=True` 带来的 `vlm.eval()` 与 LoRA 训练的交互；
Windows 平台上 peft 与该结构的其他开销。

## 5. 结论

### 5.1 实测确认的（不依赖训练结果）

1. ✅ **官方默认 SmolVLA LoRA 完全不含视觉模块**（`modeling_smolvla.py:420`）。
   → 若最终只用了默认 LoRA，**不得**声称验证过"视觉域适配不足"。
2. ✅ **视觉相关 LoRA 在配置层面可行**：精确插入 49 个视觉模块
   （48 attention 投影 + 1 connector），仅训 **1,391,616 参数（0.309%）**，
   且**实测确认 adapter 参数确实更新**。
3. ✅ **显存不是障碍**：LoRA 峰值显存 **908 MiB**。
4. ❌ **但在本机训练吞吐极低**（135–225 s/step），**无法在合理时间预算内完成有效微调**；
   瓶颈原因未定位，不做因果推断。

### 5.2 对失败归因的影响

**没有改变**。项目 B 的归因仍停在：

| 假设 | 状态 |
|---|---|
| 训练步数不足 | ✅ 已排除 |
| 数据量不足 | ✅ 已排除 |
| 显存降级 | ✅ 已排除 |
| 开环执行过长 | ✅ 已排除（110 episodes） |
| **视觉域适配不足 / 预训练分布不匹配** | ⚠️ **仍是假设** —— 两条验证路径均未完成（全量解冻算力不可接受；视觉 LoRA 吞吐极低） |

**新增的一条方法论结论**（建议写进报告与面试口径）：

> **"改配置字段"不能证明"行为已改变"** —— 本项目出现两次反例：
> `freeze_vision_encoder=false` 被 `train_expert_only` 覆盖（loss/grdn 24/24 逐位相同）。
> 任何配置变更都必须用**行为指标**（loss、梯度、速度、adapter 权重变化）验证。

### 5.3 项目 B 的最终表述（建议口径）

> 项目 B 构建了完整的 VLA 微调与评测链路（自写数据管道、与项目 A 共用评测器、效率分析），
> 并通过**五轮控制变量诊断**逐一排除常见归因（数据量、训练步数、显存、开环执行长度）。
> 剩余解释是**预训练分布不匹配**，但**该假设未获得因果验证**：
> 全量 VLM 解冻在 8 GB 卡上成本不可接受；视觉 LoRA 在配置层面可行且显存充足
> （0.9 GB、仅 0.309% 参数、adapter 确认在更新），却在当前实现下吞吐极低，
> 无法在合理预算内完成训练。
> **结论限于"在当前投入规模与硬件条件下不适用"，不外推为"VLA 不适用"。**

## 6. 原始证据位置

| 内容 | 位置 |
|---|---|
| module tree 调查脚本 | `scripts/inspect_smolvla_modules.py` |
| smoke test 脚本（插入位置 / 可训练参数 / adapter 更新） | `scripts/lora_smoke.py` |
| 视觉 LoRA 训练（中断，无可用 checkpoint） | `experiments/B_lora_vision_finetune/`、`experiments/B_lora_speedtest/` |
| 全量解冻的无效对照 | `experiments/B_unfreeze_finetune_invalid/` |
| 训练日志（不入库） | `%TEMP%\ea_scripts\` 下的 `lora_train.log`、`lora_speed.log`、`unfreeze_train.log` |

## 7. 相关

- [`smolvla_audit.md`](smolvla_audit.md) — 训练配置审计 + 全量解冻的算力边界（§3.3）
- [`latency_report.md`](latency_report.md) — 延迟拆分
- `reports/B.md` — 项目 B 完整报告（§5 归因、§7 语言条件实验、§9 局限）
