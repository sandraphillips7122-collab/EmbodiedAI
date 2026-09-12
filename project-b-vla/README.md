# 项目 B · 语言条件 VLA 操作（SmolVLA + LeRobot）

**目标**：证明「VLA 微调 → 多模态指令理解 → 具身闭环 → 推理效率」能力。
复用项目 A 的 env / 数据 / 评测框架（`embodied_core/`），因此两个项目结果**可直接对照**。

> 结果与深入分析见 [`reports/B.md`](../reports/B.md)；进度事实源见 [`docs/phase3_progress.md`](../docs/phase3_progress.md)。

## 结果（全部实测）

| 指标 | 值 |
|---|---|
| **MVP 成功率（v2：500 eps / 20000 steps）** | **1/60 = 0.0167**（±0.0165，Wilson 95% CI [0.003, 0.089]） |
| MVP 成功率（v1：100 eps / 5000 steps） | 1/30 = 0.0333 |
| **真正 VLA forward 延迟** | **≈ 433 ms**（p50 393 / p95 620，n=12）—— 含 10 步 flow-matching 去噪 |
| 纯 action queue 出队 | ≈ 2.42 ms |
| **闭环平均 step latency** | **≈ 25.5 ms（≈39 Hz）**；**每 50 步含一次 ≈0.43 s 的 forward 尖峰**（详见 [`docs/latency_report.md`](../docs/latency_report.md)） |
| `n_action_steps` 控制变量（1/5/10/25/50） | **全部 0 成功**（110 episodes）→ "开环执行过长" **已排除**（`reports/B.md` §5.2） |
| **语言条件性 2×2 析因** | 四格均为 **1/30 = 0.0333**（正确 vs 错误指令差 **0.0000**）→ **地板效应，无法检测**（`reports/B.md` §7） |
| **视觉 LoRA**（最后一轮尝试） | **未获得结果**：配置层面可行（精确插入 49 个视觉模块、可训 1.39M = **0.309%**、**实测 adapter 确实更新**、峰值显存 **908 MiB**），但训练吞吐 **135–225 s/step**（v2 为 1.40 s），无法在合理预算内完成 → 见 `reports/B.md` §5.3 |
| **官方默认 LoRA 的 target** | **不含任何视觉模块**（只有 `lm_expert.*.{q,v}_proj` + 5 个投影层）→ 因此**无法**验证"视觉域适配不足" |
| 指令泛化四档（模板/同义/组合/干扰） | 0/12、0/9、0/6、0/6 → **0.0000** |
| 场景泛化 baseline | 0/5 → 0.0000 |
| 训练显存 | **3.57 GB**（batch 8） |
| **训练方式** | **部分微调**：SmolVLA 官方默认的 `freeze_vision_encoder=true` + `train_expert_only=true`，实测可训 **99.88M / 450.05M（22.19%）**，整个 VLM 主干（含视觉编码器）**冻结**；未额外启用 LoRA（详见 [`docs/smolvla_audit.md`](../docs/smolvla_audit.md)） |

**核心诚实结论**：SmolVLA 远低于项目 A 的任务专用 ACT（**0.3933**）。

归因见 `reports/B.md` §5 / §5.1 / §5.2，**已用控制变量实验排除**：

- ✅ 训练步数不足（5000 → 20000 无改善）
- ✅ 数据量不足（100 → 500 episodes 无改善）
- ✅ 显存降级（batch 8 峰值 3.57 GB，未额外降级）
- ✅ **开环执行过长**（`n_action_steps` 1→50 全部 0 成功，110 episodes）

**剩余解释（⚠️ 属假设，未经对照实验证明）**：
**「预训练分布不匹配」+「视觉编码器全程冻结」** ——
后者是前者的具体机制（模型没有机制适配自定义相机）。

**该假设的检验进展**（诚实记录）：

- 尝试过 `train_expert_only=False` 的**全量 VLM 解冻** → 单步耗时 **47.72 s**（慢 34 倍），
  20000 步需约 265 小时 → **在当前硬件上成本不可接受**（`reports/B.md` §5.1、`docs/smolvla_audit.md` §3.3）
- ⚠️ **但"单独解冻 vision tower"这个条件从未被测到** —— 当前 LeRobot 配置不提供该选项
- 因此改用 **LoRA** 做参数高效的视觉适配尝试（见 §5.3），这才是本轮针对该假设的有效检验

## 已完成 ✅

| 环节 | 要点 |
|---|---|
| 依赖 | `lerobot 0.6.1` + `[smolvla]` + `[dataset]`；**torch 未被改动**（仍 `2.9.1+cu128`） |
| 权重 | `lerobot/smolvla_base`（走 `hf-mirror.com`） |
| **S2 显存验证** | 450M 参数、加载 883 MiB、batch=8 训练峰值 **3654 MiB** → 8 GB 够用，**无需额外降级**（[`docs/s2_smolvla_smoke.md`](../docs/s2_smolvla_smoke.md)）；实际训练采用的是官方默认的「冻结 VLM + 只训 expert」部分微调（[`docs/smolvla_audit.md`](../docs/smolvla_audit.md)） |
| **3 相机数据** | `scripts/replay_3cam.py`（monkeypatch `_default_sensor_configs` 这个 **property**）→ **100/100** episodes，502 MB |
| **DatasetPipe** | `embodied_core/data/pipeline.py`：`.h5` → LeRobotDataset（3 相机 + 29 维 state + 8 维 action + **task 指令**） |
| 微调 | `scripts/train_smolvla_windows.py`（绕过 Windows symlink 权限问题）→ 5000 steps |
| 策略适配器 | `embodied_core/rollout/smolvla_policy.py`（**必须用官方 `make_pre_post_processors` 做反归一化**，否则动作越界到 2.78） |
| 评测 | 与项目 A 同口径（`RolloutRunner` + 3 seed 块 × 10） |
| 指令/场景泛化 | 四档指令 + 光照/位姿扰动 |
| 效率 | 延迟分位数（结论：**推理不是瓶颈**，瓶颈在渲染 37 ms/步） |
| **Demo** | `scripts/demo_gradio.py`（语言输入 + 三视角画面 + 状态面板，`smolvla`/`act` 双策略） |

## 一键复现

```cmd
cd /d E:\EmbodiedAI

:: 1) 3 相机 replay（默认 100 条；要 1000 条加 --count 1000）
.venv\Scripts\python.exe scripts\replay_3cam.py --traj-path "%USERPROFILE%\.maniskill\demos\PickCube-v1\motionplanning\trajectory.h5" --use-first-env-state -c pd_joint_delta_pos -o rgb --save-traj -b physx_cpu --num-envs 1 --count 100

:: 2) 转 LeRobotDataset（视频编码较慢：100 条约 40 分钟）
.venv\Scripts\python.exe -m embodied_core.data.pipeline --h5 "%USERPROFILE%\.maniskill\demos\PickCube-v1\motionplanning\trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5" --repo-id local/pickcube_3cam_100

:: 3) 微调（注意：--policy.path 用本地快照绝对路径，避免 interop 把 / 改成 \）
.venv\Scripts\python.exe scripts\train_smolvla_windows.py --policy.path="C:/Users/<user>/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/<hash>" --policy.push_to_hub=false --dataset.repo_id=local/pickcube_3cam_100 --dataset.root="C:/Users/<user>/.cache/huggingface/lerobot/local/pickcube_3cam_100" --batch_size=8 --steps=5000 --save_freq=1000 --log_freq=100 --num_workers=0 --wandb.enable=false --output_dir="E:/EmbodiedAI/experiments/B_smolvla_finetune"

:: 4) 评测（3 seed 块 × 10 + MP4；注意 cameras='three' 与 TDR 保护）
.venv\Scripts\python.exe -c "from embodied_core.envs.adapter import EnvSpec; from embodied_core.rollout.runner import RolloutConfig, run_rollout; from embodied_core.rollout.smolvla_policy import SmolVLAPolicyAdapter as A; p=A('E:/EmbodiedAI/experiments/B_smolvla_finetune/checkpoints/005000/pretrained_model', instruction='pick up the red cube and move it to the goal position', device='cuda'); cfg=RolloutConfig(exp_id='B_mvp_smolvla', env=EnvSpec(max_episode_steps=150), seeds=(0,1,2), episodes_per_seed=10, max_episode_steps=150, policy_name='smolvla', cameras='three', sync_and_free=True, rebuild_env_every=3, record_video=True, num_videos=3); print(run_rollout(cfg, policy=p))"

:: 5) 交互 Demo（占用终端，Ctrl+C 退出）
.venv\Scripts\python.exe scripts\demo_gradio.py --policy smolvla
.venv\Scripts\python.exe scripts\demo_gradio.py --policy act
```

## 关键工程坑（都已解决，勿重复踩）

1. **`--policy.path=lerobot/smolvla_base` 被 interop 改写**成 `lerobot\smolvla_base` → HF repo id 校验失败。**用本地快照绝对路径**。
2. **训练收尾建符号链接失败**（`WinError 1314`）→ 训练跑完却异常退出。`train_smolvla_windows.py` 里把 `update_last_checkpoint` 换 no-op。
3. 必须 `--policy.push_to_hub=false`，否则因缺 `repo_id` 报错。
4. `torchcodec` DLL 加载失败 → 自动回退 `pyav`（可用）。
5. **3 相机必须在数据生成与评测两侧一致** → 配置集中在 `envs/cameras.py`；评测用 `make_env(spec, cameras="three")`。
6. **模型输出需反归一化**（否则动作越界）→ 用官方 `make_pre_post_processors(cfg, pretrained_path=ckpt)`。
7. **长时渲染 + GPU 推理会触发 Windows TDR**（`vk::ErrorDeviceLost` ← `nvlddmkm` ID 153）→ 评测循环必须带 `sync_and_free=True` 与 `rebuild_env_every`；**手写循环同样要带**；为每个指令变体重建模型实例也会触发，应复用同一实例。
8. **`--policy.freeze_vision_encoder=false` 单独设置是无效的** —— 默认的 `train_expert_only=True` 会强制把整个 VLM 的 `requires_grad` 置为 False（`lerobot/policies/smolvla/smolvlm_with_expert.py:157`），**优先级更高**。要真正解冻必须**两个都设 false**；但那样解冻的是**整个 VLM 主干（448M 参数）**，单步耗时从 **1.40 s → 47.72 s（慢 34 倍）**，20000 步需约 **265 小时** → **「全量 VLM 解冻在 8 GB 消费级卡上成本不可接受」**。⚠️ **措辞边界**：当前配置**没有"只解冻 vision tower"的选项**，因此**不能**说"已证明单独解冻 vision encoder 不可行"—— 那个条件**从未被测试**。详见 [`docs/smolvla_audit.md`](../docs/smolvla_audit.md) §3.3。

## 关键参考

- [`reports/B.md`](../reports/B.md) — 结果、负结果归因、TDR 完整排查过程、局限
- [`docs/s2_smolvla_smoke.md`](../docs/s2_smolvla_smoke.md) — SmolVLA 规格与显存实测
- [`docs/phase3_progress.md`](../docs/phase3_progress.md) — 进度事实源与复现入口
- [`docs/env_decision.md`](../docs/env_decision.md) — Windows 平台约束与依赖镜像源
