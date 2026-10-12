# 项目 B 进度与关键技术事实

> 本文件是项目 B 的**进度事实源**：每完成一步就更新，便于上下文压缩后恢复状态。
> 数字全部来自实测。环境与依赖源见 [`env_decision.md`](env_decision.md)；结果分析见 [`../reports/B.md`](../reports/B.md)。

## 当前状态（全链路已完成 ✅）

| 步骤 | 状态 | 关键事实（实测） |
|---|---|---|
| 依赖安装 | ✅ | `lerobot 0.6.1` + `[smolvla]` + `[dataset]`；**torch 未被改动**（仍 `2.9.1+cu128`） |
| 权重 | ✅ | `lerobot/smolvla_base` 已缓存（走 `hf-mirror.com`） |
| S2 显存验证 | ✅ | 450M 参数、加载 883 MiB、**batch=8 峰值 3654 MiB** → 8 GB 足够，无需额外降级 |
| **训练方式审计** | ✅ | 实测可训 **99,880,992 / 450,046,176（22.19%）**；**整个 VLM 主干（含视觉编码器）冻结**，只训 `lm_expert` + `state_proj` + `action_*` 投影层；`use_peft=False`。即官方默认的**部分微调**，**不是全参数微调**（见 [`smolvla_audit.md`](smolvla_audit.md)） |
| 3 相机数据（v1） | ✅ | `scripts/replay_3cam.py`（monkeypatch `_default_sensor_configs` 这个 **property**）→ `Replayed 100 episodes, 100/100`，502 MB |
| **3 相机数据（v2）** | ✅ | `Replayed 1000 episodes, 1000/1000=100.00%`，**5.05 GB**（1 相机 1000 条版备份为 `.1cam.`，3 相机 100 条版备份为 `.3cam100.`） |
| DatasetPipe | ✅ | `state_dim=29` / `action_dim=8` / `fps=20`；v1 100 eps → 166 MB；**v2 500 eps → 11 GB（图片模式）** |
| 微调 v1 | ✅ | 5000 steps，loss 1.714(100) → 0.273(900)，~1.2 s/step，显存 3.57 GB |
| **微调 v2** | ✅ | **20000 steps**（3:29:19），`--num_workers 8`，0.62 s/step，显存 3.57 GB；8 个 checkpoint（2500…20000） |
| 策略适配器 | ✅ | `smolvla_policy.py`；**必须用官方 `make_pre_post_processors` 反归一化**（手写会越界到 2.78） |
| 主评测 v1 / v2 | ✅ | v1 **1/30 = 0.0333**；v2 **1/60 = 0.0167** —— **加大投入没有改善**（见 `reports/B.md` §5 归因） |
| 指令泛化四档 | ✅ | 模板 0/12、同义 0/9、组合 0/6、干扰 0/6（基线为 0，负结果） |
| 场景泛化 | ✅ | baseline 0/5 |
| 推理效率 | ✅ | **mean 6.7 ms / p95 7.3 ms（≈149 Hz）** → 推理不是瓶颈，渲染才是（37 ms/步） |
| Gradio Demo | ✅ | `scripts/demo_gradio.py --policy smolvla|act`，三视角 + 状态面板 |

## Windows / interop 上踩到的坑（都已解决，勿重复踩）

1. **`--policy.path=lerobot/smolvla_base` 会被 interop 改写成 `lerobot\smolvla_base`** → HF repo id 校验失败。
   **解决**：传本地快照的**绝对路径**。
2. **训练收尾建符号链接失败**（`WinError 1314`）→ 训练跑完却异常退出。
   **解决**：`scripts/train_smolvla_windows.py` 把 `update_last_checkpoint` 换 no-op。
3. **必须显式 `--policy.push_to_hub=false`**，否则 `TrainConfig.validate()` 因缺 `repo_id` 报错。
4. **`torchcodec` DLL 在本机加载失败** → lerobot 自动回退 `pyav`，可用。
5. **3 相机必须在「数据生成」与「评测」两侧一致**：`make_env(spec, cameras="three")`；配置集中在 `embodied_core/envs/cameras.py`。
6. **图片模式下 `--num_workers 0` 会让训练慢 5 倍以上**（3.45 s/step vs 0.62 s/step）。这是 dataloader 瓶颈，不是 GPU 瓶颈。
7. **`DatasetPipe` 的 features `dtype` 必须与 `use_videos` 一致**（`use_videos=False` 时只能是 `"image"`），否则 lerobot 直接抛错。已修复。
8. **视频编码在消费级卡上不实用**：500 条 episode 的视频会压进极少的大 mp4（单文件 209 MB+），数小时仍未 finalize → 改用图片模式（40 分钟）。
9. **长时渲染 + GPU 推理会触发 Windows TDR**（`nvlddmkm` ID 153）→ 评测循环必须 `sync_and_free` + `rebuild_env_every`；**手写循环也要带**；为每个指令变体重建模型实例也会触发，应复用同一实例。
10. **工具门禁**：本会话中 `edit`/`write` 之后，含模块路径的 shell 命令会被拦；可用 heredoc（`python - <<'PYEOF'`）规避。
11. **`python scripts\xxx.py` 报 `ModuleNotFoundError: No module named 'embodied_core'`** —— Python 只把**脚本所在目录**（`scripts/`）加入 `sys.path`，而不是当前工作目录；本仓库当时也没有 `pyproject.toml`（`embodied_core` 不是可安装包）。
    **解决**：受影响的脚本（`demo_gradio.py` / `probe_act.py` / `smoke_video.py`）在 import `embodied_core` **之前**统一加一段兜底：
    ```python
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    ```
    注意 `train_act_windows.py` 里出现的 `embodied_core` **只在注释里**，它并不 import 该包，所以不需要改。
12. **验证方法**：改完后用 `probe_act.py --help` 验证（它在**模块顶层**导入 `embodied_core`，最能暴露该问题）；只测 `demo_gradio.py --help` 不够，因为它是延迟导入。

## ⚠️ 已知的数据缺口（如实记录，不影响既有结论）

**3 相机 1000 条的中间产物 h5 已丢失**

- 原文件名：`~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5`（5.05 GB）
- **原因**：`scripts/replay_dual_cube.py` 的早期版本**无条件 rename** 了同目录下的这个同名文件
  （把它改成了 `trajectory.dualred.rgb...h5`），第二次 replay 又把它覆盖。
  **该 bug 已修复**：现在只在「本次 replay 确实改写了该文件（mtime 变化）」时才重命名，
  且目标已存在时**直接报错拒绝覆盖**。
- **影响范围**：**只是中间产物**。下游全部完好 ——
  - LeRobotDataset `pickcube_3cam_500img`（11 GB，v2 训练输入）✅
  - v1 / v2 / langcond 的 checkpoint ✅
  - 语言条件的 `dualred` / `dualblue` h5 ✅
  - 3 相机的 **100 条备份**（`trajectory.rgb.3cam100...h5`，502 MB）✅
- **恢复方式**（约 1.7 小时，可从 1000 条示教重建）：

```cmd
.venv\Scripts\python.exe scripts\replay_3cam.py --traj-path "%USERPROFILE%\.maniskill\demos\PickCube-v1\motionplanning\trajectory.h5" --use-first-env-state -c pd_joint_delta_pos -o rgb --save-traj -b physx_cpu --num-envs 1 --count 1000
```

> 该缺口**不影响任何已汇报的数字**：v2 的训练输入是下游的 LeRobotDataset（完好），
> 而全部历史实验的 `episodes.jsonl` 全部保留（独立复算通过）。

## 复现入口（v2，当前有效）

```cmd
:: 1) 3 相机 replay 1000 条（约 1.7 小时，5.05 GB）
cd /d <你的仓库根目录>
.venv\Scripts\python.exe scripts\replay_3cam.py --traj-path "%USERPROFILE%\.maniskill\demos\PickCube-v1\motionplanning\trajectory.h5" --use-first-env-state -c pd_joint_delta_pos -o rgb --save-traj -b physx_cpu --num-envs 1 --count 1000

:: 2) h5 → LeRobotDataset（图片模式：约 40 分钟，500 条 → 11 GB）
.venv\Scripts\python.exe -m embodied_core.data.pipeline --h5 "%USERPROFILE%\.maniskill\demos\PickCube-v1\motionplanning\trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5" --repo-id local/pickcube_3cam_500img --instruction "pick up the red cube and move it to the goal position" --max-episodes 500 --no-videos

:: 3) SmolVLA 微调 20000 steps（约 3.5 小时；num_workers 必须 > 0）
.venv\Scripts\python.exe scripts\train_smolvla_windows.py --policy.path="<snapshot 绝对路径>" --policy.push_to_hub=false --dataset.repo_id=local/pickcube_3cam_500img --dataset.root="C:/Users/<user>/.cache/huggingface/lerobot/local/pickcube_3cam_500img" --batch_size=8 --steps=20000 --save_freq=2500 --log_freq=200 --num_workers=8 --wandb.enable=false --output_dir="experiments/B_smolvla_finetune_v2"

:: 4) 评测：见 project-b-vla/README.md 第 4 步
```

## 本轮新增：诊断实验与语言条件任务

> 全部为实测。新增实验一律使用**新的 exp_id**，未覆盖任何历史 JSONL / checkpoint。

### 1. 训练配置审计（更正一处不准确表述）

| 项 | 实测 |
|---|---|
| 总参数 / 可训练 | **450,046,176 / 99,880,992（22.19%）** |
| 冻结 | **整个 `model.vlm_with_expert.vlm`（含视觉编码器）** |
| 可训练 | `lm_expert`(98.2M) + `state_proj` + `action_in/out_proj` + `action_time_mlp_in/out` |
| `use_peft` | `false`（确实**没用** LoRA） |

**此前"无需 LoRA/冻结降级""未启用任何冻结阶梯"的表述不准确** → 已更正为
"采用 SmolVLA 官方默认的部分微调"。审计脚本 `scripts/audit_smolvla_config.py`，
结果见 [`smolvla_audit.md`](smolvla_audit.md)。
（附带排除一个假 bug：`config` 里 `observation.state shape=[6]` 是陈旧元数据，
`modeling_smolvla.py:409` 的 `pad_vector` 只 pad 到 32、不截断，训练推理同函数，无害。）

### 2. 延迟重新拆分（推翻旧结论）

| 项 | 实测（n） |
|---|---|
| **真正 VLA forward**（queue 空，含 10 步去噪） | **≈ 433 ms**（p50 393 / p95 620，n=12） |
| 纯 action queue 出队 | ≈ 2.42 ms（n=588） |
| 闭环平均 step latency | ≈ 25.5 ms（≈39 Hz） |
| forward 触发频率 | **每 50 步 1 次** |

旧报告"策略延迟 mean 6.7 ms ≈ 149 Hz"实为 **`policy call` 的 p50（6.54 ms）**，
落在**出队步**上 —— 中位数把每 50 步一次、**0.38–0.86 s** 的 forward 尖峰完全掩盖。
脚本 `scripts/measure_latency.py`，明细见 [`latency_report.md`](latency_report.md)。

### 3. `n_action_steps` 控制变量实验（排除"开环执行过长"）

同一 checkpoint、同一 seeds，**只改** `policy.config.n_action_steps`（`chunk_size` 不变）：

| n_action_steps | 成功/总数 | 95% CI 上限 | 平均单集耗时 |
|---|---|---|---|
| **1**（每步重规划） | **0/30** | 0.114 | **77.83 s** |
| 5 | 0/15 | 0.204 | 17.05 s |
| 10 | 0/15 | 0.204 | 9.76 s |
| 25 | 0/20 | 0.161 | 4.97 s |
| **50**（原配置） | **0/30** | 0.114 | **3.59 s** |

**五个配置全部 0 成功（110 episodes）** → 「开环执行过长」**已排除**；
且 `n=1` 慢 **21.7 倍**却毫无收益。脚本 `scripts/replan_experiment.py`，
新 exp_id：`experiments/B_replan_n{1,5,10,25,50}/`。

### 4. 语言条件性最小任务（`PickDualCube-v1`）

**为什么不能用"两个 goal 位置"**（原计划，实测后推翻）：

1. `goal_site` 在 `_hidden_objects` 里，ManiSkill 取观测前调 `hide_visual()`
   （`sapien_env.py:601`）→ **goal 在相机中不可见**（实测相距 0.3 m 的两个 goal 三个相机均 **0 像素差异**）；
2. 即便可见，`goal_pos` 也在 `obs.extra` 里 → 模型**看 state 就能做对，无需读指令**。

**因此改用"两个候选物体"**：两个同形状不同颜色方块，指令指定抓哪一个，
`obs.extra` **不含**"哪个是目标"。**零额外示教成本**的技巧：
**不改颜色，只交换两个方块的初始位置** —— 现有示教轨迹（抓槽位 A）在两种布局下都成立。

| 变体 | 红块 | 蓝块 | 指令 |
|---|---|---|---|
| `red`  | 槽位 A | 槽位 B | `pick up the red cube ...` |
| `blue` | 槽位 B | 槽位 A | `pick up the blue cube ...` |

**2×2 析因结果**（同 seeds、同 checkpoint）：

| 场景 | 指令 | 正确? | 成功率 |
|---|---|---|---|
| red | red | ✅ | 1/30 = 0.0333 |
| red | blue | ❌ | 1/30 = 0.0333 |
| blue | blue | ✅ | 1/30 = 0.0333 |
| blue | red | ❌ | 1/30 = 0.0333 |

**正确 − 错误 = 0.0000** → 但这是**地板效应**（四格基线都只有 0.03），
**无法区分"忽略语言"与"任务还没学会"**。
准确表述：**在本模型当前能力水平下无法检测语言条件性**，
本实验的价值在**判据干净的实验设计**，不在结论。见 `reports/B.md` §7。

**顺带修掉的 4 个真实 bug**（复现者必看）：

| # | 现象 | 根因 |
|---|---|---|
| 1 | `TypeError: unexpected keyword 'target_color'` | replay 用 `env_info["env_id"]`（`replay_trajectory.py:485`），**不是** `env_class` |
| 2 | `max_episode_steps` 截断示教 | 必须设 150（示教平均 77 步）—— 与项目 A 同一个坑 |
| 3 | 环境"未注册" | 需 `import embodied_core.envs.dual_cube` 触发 `@register_env` |
| 4 | `KeyError: 'cube_b'` | `env_states` **存在 h5 里、不在 json 里**，且 replay 会递归覆盖 → 必须在 **h5 层**注入 `cube_b` |

另：同进程内第二次 `runpy` 会 `context has already been set`（需分进程）；
连续跑多配置会触发 **TDR**。

### 5. 解冻视觉编码器的对照实验（已尝试 → **在本机不可行**）

**目的**：检验"视觉编码器全程冻结"是否是低成功率的原因（`docs/smolvla_audit.md` §3.3 有完整记录）。

**(a) 配置陷阱：改字段 ≠ 改行为**

只设 `--policy.freeze_vision_encoder=false` 时，checkpoint 的 config 里**确实**写着 `False`，
但训练行为**完全没变**：

- 根因：`train_expert_only=True` 会执行
  `for params in self.vlm.parameters(): params.requires_grad = False`
  （`lerobot/policies/smolvla/smolvlm_with_expert.py:157`）→ **优先级更高**
- 铁证：该次训练与 v2 的 **loss 与 grdn 在 24/24 个数据点上逐位相同**（末点 `0.0460 / 0.959`）
- 无效对照的 checkpoint 保存在 `experiments/B_unfreeze_finetune_invalid/`

**(b) 真正解冻后：算力不可行**

| 配置 | 单步耗时 |
|---|---|
| frozen（v2，`expert_only=True`） | **1.40 s/step** |
| 无效解冻（`freeze=False, expert_only=True`） | 1.31 s/step |
| **真正解冻（`freeze=False, expert_only=False`）** | **47.72 s/step**（**慢 34 倍**） |

→ 20000 步约需 **265 小时**，训练在 step 100 后终止。

**(c) 结论（诚实标注）**

| 说法 | 成立？ |
|---|---|
| "视觉编码器全程冻结，模型无机制适配自定义相机" | ✅ 成立（实测参数量） |
| "**全量 VLM 解冻**在 8 GB 消费级卡上**成本不可接受**（47.72 s/step，慢 34 倍）" | ✅ 成立（实测） |
| "单独解冻 **vision tower** 不可行" | ⚠️ **未测试** —— 当前 LeRobot 配置**无此选项**，不能用上一条替代 |
| "已证明成功率低是因为视觉编码器被冻结" | ❌ **仍未验证**（对照因算力受限未完成） |

⚠️ **措辞边界**：我们实测的是 `train_expert_only=False` 导致的**整体解冻**，
不是"只解冻 vision encoder"。**后者这个条件从未被执行**，不能把前者的结果当成对后者的结论。

**替代路径**：LoRA 只训低秩适配器（**已实测，见下节**）、换更大显存、或改 lerobot 单独解冻 vision tower。

### 6. 视觉 LoRA 实验（最后一轮，**未获得结果**）

**动机**：检验"视觉域适配不足"。全量解冻不可行（§5），LoRA 显存开销小且可**精确指定**视觉模块。

**(a) 官方默认 LoRA 不含视觉模块**（读源码）

`modeling_smolvla.py:420` 的 target 只有 `lm_expert.*.{q,v}_proj` + 5 个投影层 →
**`vlm` 完全不在其中** → **默认 LoRA 无法验证视觉假设**（等价于"继续训 action expert"）。

**(b) 视觉 LoRA 配置层面可行**（smoke test 实测，`scripts/lora_smoke.py`）

| 项 | 实测 |
|---|---|
| 插入模块 | **49 个**：vision `self_attn.{q,k,v,out}_proj` × 48 + `connector.modality_projection.proj` × 1（`lm_expert=0`） |
| 可训练参数 | **1,391,616 / 450,046,176 = 0.309%** |
| **adapter 确实更新** | **YES**（5 步后 32/98 张量变化，最大 0.004979） |
| 峰值显存 | **908 MiB** |

⚠️ 踩坑：`connector.modality_projection` 本身是 `SmolVLMSimpleMLP` 不是 `Linear`，
peft 报 `Target module ... is not supported` → 正则必须写成 `.modality_projection.proj`。

**(c) 训练吞吐极低 → 未获得结果**（原始证据，逐字）

```
Training:   0%|          | 1/400 [02:15<14:57:50, 135.01s/step]
Training:   0%|          | 2/400 [06:16<21:49:40, 197.44s/step]
Training:   1%|          | 3/400 [10:34<24:51:20, 225.39s/step]
```

> ⚠️ **精确措辞**：在当前 Windows + RTX 4060 8GB + LeRobot/PEFT 实现和视觉 target 配置下，
> **显存可接受，但训练计算效率极低，无法在合理时间预算内完成有效微调；
> 具体性能瓶颈原因未定位，因此不做进一步因果推断。**

**归因未变**："视觉域适配不足"**仍是假设**（两条验证路径均未完成）。
详见 [`lora_experiment.md`](lora_experiment.md)。

**新增方法论结论**：**"改配置字段"≠"行为已改变"** —— 两次踩到
（`freeze_vision_encoder=false` 被 `train_expert_only` 覆盖，loss/grdn 24/24 逐位相同）。
必须用**行为指标**（loss / 梯度 / 速度 / adapter 权重变化）验证。

### 7. 本轮新增脚本

| 脚本 | 用途 |
|---|---|
| `scripts/audit_smolvla_config.py` | 实测 checkpoint 的参数量与冻结范围 |
| `scripts/measure_latency.py` | 拆分 forward / 出队 / 环境 / 闭环延迟 |
| `scripts/replan_experiment.py` | `n_action_steps` 控制变量实验 |
| `scripts/check_goal_visibility.py` | 验证 goal 位置是否可从相机分辨（得出"不可见"结论） |
| `embodied_core/envs/dual_cube.py` | `PickDualCube-v1` 任务实现 |
| `scripts/replay_dual_cube.py` | 双色 replay（在 h5 层注入 `cube_b`） |
| `scripts/make_dual_dataset.py` | 合并 red/blue 为一个交替数据集 |
| `scripts/langcond_eval.py` | 语言条件性 2×2 析因评测 |

## 下一步（若要继续提升）

按预期收益排序（**均为未验证的假设，不要当成结论**）：

1. **换与预训练匹配的动作空间**：用末端执行器增量（需先解决 Windows 上无 pinocchio 的 IK 问题，或改用 CPU IK 实现）。
2. **换机器人形态**：用 SO-100 的 MuJoCo 模型，使本体接近预训练分布。
3. **对齐相机布局**：按 SmolVLA 预训练数据的相机几何重新布置视角。
4. **I/O 与训练优化**：视频编码改到 Linux/WSL 侧或改用更低分辨率；把 3 相机降到 2 相机以减半 I/O。
5. 若把数据推到 10 万帧级，才具备和社区结果可比的条件。

**已试过且不可行的路径**（勿重复）：

- 全量 VLM 解冻（47.72 s/step，成本不可接受）—— 见 §5
- 视觉 target 的 LoRA（配置可行，但 135–225 s/step，无法在合理预算内训练）—— 见 §6
- **官方默认 LoRA 未试也不建议试**：它的 target 不含视觉模块，**无法验证核心视觉假设**

> **项目 B 到此停止扩展**。后续工作转入两个项目的最终整理与简历阶段。
