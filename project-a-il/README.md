# 项目 A · 机器人操作模仿学习闭环（ManiSkill3 + ACT）

**目标**：证明「数据工程 → 策略训练 → 严谨评测 → 泛化/失败归因」这条闭环能力。
主结果见 [`reports/A_mvp.md`](../reports/A_mvp.md)。

- 任务：`PickCube-v1`（抓取并移动方块）
- 策略：ACT（官方 baseline 的 resnet18 + DETRVAE 结构）
- 控制：关节空间 `pd_joint_delta_pos`（Windows 无 IK，见 `docs/env_decision.md` §8）
- 评测器：`embodied_core.rollout.runner`（与项目 B 共用，保证结果可对照）

## 前置：环境

```cmd
cd /d E:\EmbodiedAI
python -m venv .venv                      :: 或用 conda base 的 python
.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mani_skill
:: CUDA 版 torch 从阿里云镜像取（清华 PyPI 的 torch 是 CPU-only！见 docs/env_decision.md §6）
:: 详见 docs/env_decision.md 的「依赖下载源选型」
```

`third_party/ManiSkill-main/`（官方 ACT baseline 源码）通过 `gh-proxy.com` 拉取，见 `docs/env_decision.md`。

## 一键复现（四步）

### 1. 数据准备（下载官方示教 + 生成 RGB 观测）

```cmd
:: 1a. 下载官方 motionplanning 示教（36.6 MB，走 hf-mirror）
curl -L -o "%TEMP%\PickCube-v1.zip" "https://hf-mirror.com/datasets/haosulab/ManiSkill_Demos/resolve/main/demos/PickCube-v1.zip?download=true"

:: 1b. 解压到 ~/.maniskill/demos（注意：真实 URL 见 docs/env_decision.md 记录）
:: 1c. 关键：补 render_backend=cpu（Windows 必需；原始文件备份为 .orig）
python -c "import json,shutil,pathlib; p=pathlib.Path.home()/'.maniskill/demos/PickCube-v1/motionplanning/trajectory.json'; shutil.copy(p,str(p)+'.orig'); d=json.loads(p.read_text(encoding='utf-8')); d['env_info']['env_kwargs']['render_backend']='cpu'; p.write_text(json.dumps(d,indent=2),encoding='utf-8')"

:: 1d. 全量 replay 生成 RGB 观测（1000 条，约 21 分钟）
.venv\Scripts\python.exe -m mani_skill.trajectory.replay_trajectory ^
  --traj-path "%USERPROFILE%\.maniskill\demos\PickCube-v1\motionplanning\trajectory.h5" ^
  --use-first-env-state -c pd_joint_delta_pos -o rgb --save-traj -b physx_cpu --num-envs 1
```

产物：`trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5`（418 MB）

**数据卡**（分布/覆盖/动作统计，可复核）：

```cmd
.venv\Scripts\python.exe scripts\inspect_demos.py ^
  --h5 "%USERPROFILE%\.maniskill\demos\PickCube-v1\motionplanning\trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5" ^
  --out docs\data_card_pickcube_rgb.md --max-trajectories 100
```

### 2. 训练 ACT（约 50 分钟）

```cmd
.venv\Scripts\python.exe scripts\train_act_windows.py ^
  --env-id PickCube-v1 ^
  --demo-path "%USERPROFILE%\.maniskill\demos\PickCube-v1\motionplanning\trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5" ^
  --num-demos 100 --total-iters 30000 --batch-size 32 --no-include-depth ^
  --num-eval-episodes 2 --num-eval-envs 1 --eval-freq 100000 --log-freq 500 --save-freq 5000 ^
  --no-capture-video --max-episode-steps 150 ^
  --exp-name act-pickcube-rgb-100demos --sim-backend physx_cpu
```

`scripts/train_act_windows.py` 是**必需的包装**：官方 `train_rgbd.py` 构造 env 时不传
`render_backend`（默认 `gpu` 在本机会进程原生崩溃），且它的自带评测依赖 GPU 向量化环境的
`info["final_info"]`（CPU 路径下会 KeyError）。包装脚本用 monkeypatch 注入 Windows 必需配置，
并把评测交给本仓库的评测器。

产物：`third_party/ManiSkill-main/examples/baselines/act/runs/act-pickcube-rgb-100demos/checkpoints/*.pt`

### 3. MVP 评测（150 次 rollout + MP4，约 5 分钟）

```cmd
.venv\Scripts\python.exe -c "from embodied_core.rollout.runner import RolloutConfig, run_rollout; from embodied_core.envs.adapter import EnvSpec; from embodied_core.rollout.act_policy import ActPolicy; p = ActPolicy('third_party/ManiSkill-main/examples/baselines/act/runs/act-pickcube-rgb-100demos/checkpoints/25000.pt', device='cuda', max_timesteps=200); cfg = RolloutConfig(exp_id='A_mvp_pickcube_act', env=EnvSpec(max_episode_steps=150), seeds=(0,1,2), episodes_per_seed=50, max_episode_steps=150, policy_name='act', record_video=True, num_videos=3); s = run_rollout(cfg, policy=p); print(s['num_successes'],'/',s['num_episodes'], s['success_rate'], s['failure_modes'])"
```

结果写入 `experiments/A_mvp_pickcube_act/`：`episodes.jsonl`（逐 episode）、`summary.json`、`videos/*.mp4`

> ⚠️ `EnvSpec(max_episode_steps=150)` 是**必须的**：只设 `RolloutConfig.max_episode_steps`
> 只限制循环，env 内部仍是 PickCube 默认的 50 步，会把 77 步长的示教动作截断
> （实测成功率 0.1 → 0.6）。详见 `reports/A_mvp.md` §4。

### 4. 泛化实验（4 变体 × 30 episodes）

见 `reports/A_mvp.md` §5 的表；变体通过
`env.unwrapped.cube_spawn_half_size`（位姿范围）与
`env.unwrapped.scene.set_ambient_light(...)`（光照）两个接口实现。

```cmd
:: 以「光照变暗」为例（其余变体同理，仅改扰动函数）
.venv\Scripts\python.exe -c "from embodied_core.rollout.runner import RolloutConfig, run_rollout; from embodied_core.envs.adapter import EnvSpec; from embodied_core.rollout.act_policy import ActPolicy; p=ActPolicy('third_party/ManiSkill-main/examples/baselines/act/runs/act-pickcube-rgb-100demos/checkpoints/25000.pt', device='cuda', max_timesteps=200); cfg=RolloutConfig(exp_id='A_gen_dark', env=EnvSpec(max_episode_steps=150), seeds=(10,11,12), episodes_per_seed=10, max_episode_steps=150, policy_name='act', cameras='default', sync_and_free=True, rebuild_env_every=5); print(run_rollout(cfg, policy=p)['success_rate'])"
```

### 5. 第二任务 StackCube-v1（验证管道可移植性）

**唯一改动**：`EnvSpec(task="StackCube-v1", max_episode_steps=200)`。

```cmd
:: 5a. 下载 + 解压 StackCube 示教，并补 render_backend=cpu（同 1b/1c）
curl -L -o "%TEMP%\StackCube-v1.zip" "https://hf-mirror.com/datasets/haosulab/ManiSkill_Demonstrations/resolve/main/demos/StackCube-v1.zip?download=true"

:: 5b. 1 相机 replay 100 条（约 8 分钟 → 366 MB）
.venv\Scripts\python.exe -m mani_skill.trajectory.replay_trajectory --traj-path "%USERPROFILE%\.maniskill\demos\StackCube-v1\motionplanning\trajectory.h5" --use-first-env-state -c pd_joint_delta_pos -o rgb --save-traj -b physx_cpu --num-envs 1 --count 100

:: 5c. 训练（100 demos，30000 iters，约 1 小时）
.venv\Scripts\python.exe scripts\train_act_windows.py --env-id StackCube-v1 --demo-path "%USERPROFILE%\.maniskill\demos\StackCube-v1\motionplanning\trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5" --num-demos 100 --total-iters 30000 --batch-size 32 --no-include-depth --num-eval-episodes 2 --num-eval-envs 1 --eval-freq 100000 --log-freq 500 --save-freq 5000 --no-capture-video --max-episode-steps 200 --exp-name act-stackcube-rgb-100demos --sim-backend physx_cpu

:: 5d. 评测（把上面的 ActPolicy 路径换成 act-stackcube-rgb-100demos/checkpoints/25000.pt，EnvSpec(task='StackCube-v1', max_episode_steps=200)）
```

结果：**13/30 = 0.4333**（95% CI [0.274, 0.608]），记录在 `experiments/A_stackcube_act/`。

### 6. 数据量曲线（50 / 100 / 200 demos）

同口径对照（同一份数据、同一批超参、**同一 checkpoint 步数**，各 30 episodes）：

```cmd
:: 训练（把 N 换成 50 / 100 / 200）
.venv\Scripts\python.exe scripts\train_act_windows.py --env-id PickCube-v1 --demo-path "%USERPROFILE%\.maniskill\demos\PickCube-v1\motionplanning\trajectory.rgb.1cam.pd_joint_delta_pos.physx_cpu.h5" --num-demos N --total-iters 15000 --batch-size 32 --no-include-depth --num-eval-episodes 2 --num-eval-envs 1 --eval-freq 100000 --log-freq 1000 --save-freq 5000 --no-capture-video --max-episode-steps 150 --exp-name act-pickcube-1cam-Ndemos --sim-backend physx_cpu
```

| 示教条数 | 成功率 |
|---|---|
| 50 | 3/30 = **0.1000** |
| 100 | 4/30 = **0.1333** |
| 200 | 21/30 = **0.7000** |

> ⚠️ **注意 checkpoint 排序**：`sorted(glob("*.pt"))` 是**字符串排序**，`'10000.pt' > '5000.pt' > '0.pt'` 并不成立 ——
> `[-1]` 取到的是 `5000.pt`。本表的三个点都是 `5000.pt`，口径一致因此可比；
> 若要取最新一步，请按 **int(stem)** 排序。

### 7. 细粒度失败归因

```cmd
:: 在评测循环里收集信号（is_grasped / 末端位置 / 最终到目标距离），再调用分类器
:: 模块：embodied_core/metrics/failure_modes.py（阈值见文件头部常量）
```

20 episodes 的实测分布：`success 3 / slip 12 / not_reached 4 / collision 1`，
即**主导失败模式是"抓起来后滑落"**，记录在 `experiments/A_failure_modes.json`。

## 代码结构

```
embodied_core/
  envs/adapter.py        EnvSpec（含 Windows 硬约束校验）+ make_env
  rollout/runner.py      评测器：固定 seed、成功率±SE、MP4、JSONL 落盘
  rollout/act_policy.py  ActPolicy：官方 ACT checkpoint → 统一 Policy 接口（零侵入）
  metrics/recorder.py    成功率/Wilson CI/失败模式 → JSONL + summary.json
scripts/
  train_act_windows.py   Windows 上跑官方 ACT baseline 的包装
  inspect_demos.py       数据卡生成
  probe_act.py           动作探针（区分「策略没学会」与「推理有 bug」）
configs/                 实验配置
reports/A_mvp.md         结果与归因
```
