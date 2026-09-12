# SmolVLA 延迟拆分实测（项目 B）

> **方法声明**：全部为本机实测，用 `torch.cuda.synchronize()` + `time.perf_counter()` 计时，
> 每项报 n / mean / p50 / p95 / min / max。
> 脚本：`scripts/measure_latency.py`；原始输出：`experiments/B_latency/latency.json`（不入库）。
> 硬件：Windows 原生 / RTX 4060 Laptop 8 GB / `render_backend=cpu`；checkpoint：
> `experiments/B_smolvla_finetune_v2/checkpoints/020000/pretrained_model`。

## 1. 为什么必须重测

早期报告写过：

> "SmolVLA 策略延迟 **mean 6.7 ms**（≈ **149 Hz**）→ action chunk 生效时 **VLA 推理不是瓶颈**"

这个说法**把两件事混在了一起**。SmolVLA 的 `select_action` 自带 **action queue**
（`chunk_size = n_action_steps = 50`）：

- **每 50 步里只有 1 步真正做 VLA forward**（10 步 flow-matching 去噪），其耗时是**数量级更大**的；
- 其余 49 步只是**从队列里取出预先算好的动作**，耗时极小。

拿这个混合分布的**中位数**当"VLA 推理速度"，会把每 50 步一次的**长尾尖峰完全掩盖**。

## 2. 六项实测结果（forward 样本 n=12）

| # | 项 | n | mean (ms) | p50 | p95 | min | max |
|---|---|---|---|---|---|---|---|
| 1 | `preprocessor` | 30 | 1.56 | 1.62 | 1.98 | 0.98 | 2.09 |
| 2 | **真正 forward（queue 空）** | **12** | **433.24** | **393.02** | **620.02** | 377.47 | 859.28 |
| 3 | **queue 出队（queue 非空）** | 588 | **2.42** | 2.40 | 2.69 | 2.13 | 3.42 |
| 4 | `postprocessor` | 30 | 0.83 | 0.26 | 0.44 | 0.16 | 17.30 |
| 5 | `env.step`（仿真 + CPU 渲染） | 60 | 6.65 | 6.77 | 7.98 | 4.98 | 9.77 |
| 6 | **完整闭环 step** | 120 | **25.49** | 13.68 | 15.52 | 10.22 | **530.87** |
| 6a | 其中 policy 调用 | 120 | 18.48 | **6.54** | 7.37 | 5.05 | **525.40** |

**派生量**：

| 量 | 数值 |
|---|---|
| **真正 VLA forward 延迟**（单次，含 10 步去噪） | **≈ 433 ms**（p50 393 / p95 620） |
| 纯 action queue 出队 | **≈ 2.42 ms** |
| 折算等效 policy 频率（按队列权重平均） | ≈ 90.6 Hz |
| **闭环平均 step latency** | **≈ 25.5 ms（≈ 39 Hz）** |
| forward 触发频率 | **每 50 步 1 次** |
| 环境 step 占闭环比例 | ≈ 26.1% |

> **样本量说明**：初测只跑 4 轮（forward mean 702 ms），复测提高到 12 轮（mean 433 ms）。
> 两者都指向**数百毫秒**这一量级，差异主要来自初测的 warmup 与偶发尖峰。
> 上表为 n=12 的复测结果。

## 3. 关键更正：原来的 6.7 ms 是什么

把第 6a 行与旧报告的"6.7 ms"对照：

```
6a) policy call in loop : p50 = 6.54 ms     ← 与旧报告的 "mean 6.7 ms" 几乎完全一致
```

**所以**：旧报告里的 6.7 ms 实为 **policy 调用的 p50**，而 p50 落在**出队步**上
（出队 2.42 ms + pre/post 处理 ≈ 6.5 ms）。

**中位数把长尾完全掩盖**：每 50 步一次的真实 forward 是 **377–859 ms**，在 p50 里根本看不见。

**修正后的正确表述**（`reports/B.md` §6.2 已按此更新）：

> 闭环平均 step latency ≈ 25.5 ms（≈39 Hz）；
> 但**每 50 步含一次 ≈0.43 s 的 forward 尖峰**（p95 达 0.62 s）。
> 平均闭环延迟与单次 forward 延迟**是两个不同的数字，必须分开陈述**。

## 4. 这对"闭环能否跟得上"意味着什么

结论要**分两层**，不能一句话说完：

1. **平均而言**：闭环 ≈25.5 ms/step，环境占 26.1% —— 单看平均值，闭环并不慢。
2. **但存在周期性尖峰**：每 50 步一次，**≈0.39–0.62 秒**。在那一步上，
   **瞬时控制频率降到约 1.6–2.6 Hz**。对接触丰富的操作任务，
   0.4–0.6 秒的卡顿意味着机器人按"已经过时的观测"算出的动作序列继续盲走。

**因此早期"VLA 推理不是瓶颈"的结论只在平均意义上成立，作为工程结论是不完整的。**
准确表述是：

> **平均闭环 25.5 ms/step，但每 50 步含有一次 ≈0.43 s 的 forward 尖峰；
> 是否构成瓶颈取决于任务对"瞬时控制频率"的容忍度，不能只用平均值下结论。**

这也给出一条**不需要重训就能检验**的方向 —— 缩短 `n_action_steps`（1/5/10/25/50），
让 forward 更频繁但每次开环更短，观察成功率如何变化。

## 5. 复现

```cmd
cd /d E:\EmbodiedAI
:: 默认 12 轮 forward 采样
.venv\Scripts\python.exe scripts\measure_latency.py

:: 调参并落盘
::   --forward-reps 控制 forward 采样数（每轮 reset() 只产生 1 次真 forward，故需多轮）
::   --steps        控制闭环测量步数
.venv\Scripts\python.exe scripts\measure_latency.py --forward-reps 12 --steps 120 --json out.json
```

## 6. 限制

- 环境侧是 **CPU 渲染**（Windows 硬约束，见 `docs/env_decision.md` §8），
  因此第 5 行的 6.65 ms **不能与 GPU 渲染的结果直接比较**。
- forward 延迟波动较大（377–859 ms），与 flow-matching 的 10 步去噪、KV cache 命中情况、
  以及后台进程占用 GPU 都有关；**量级可靠，个位精度不必较真**。
- 本报告只测**延迟**，不涉及成功率；二者关系见 Phase 3 的 `n_action_steps` 实验
  （`experiments/B_replan_n*/`）。

## 7. 相关

- [`smolvla_audit.md`](smolvla_audit.md) — 训练配置审计（实测参数量与冻结范围）
- `reports/B.md` §6.2 — 已按本报告更正"6.7 ms ≈ 149 Hz"的表述
