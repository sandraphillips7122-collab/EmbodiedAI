# 如何验证本仓库的结论

> 本文件回答一个具体问题：**`reports/` 里那些成功率数字，凭什么可信？**
>
> 一句话：**每个对外汇报的数字，都能用独立代码从原始记录复算出来。**

## 为什么需要这份文档

本仓库的两个项目都由 AI 辅助开发，数字也是脚本跑出来的。所以"可信"不能靠自我声明，
必须能**被外部独立复核**。为此设计了三层验证，每层针对不同类型的错误。

## 三层验证

### 第 1 层：独立复算已记录实验（最有说服力）

`tests/test_recorded_experiments.py`

**关键设计：它刻意不 import `embodied_core.metrics`。**

如果复算脚本调用同一个 `summarize()`，那么"公式写错了"这种 bug 会被**一起复现**，
测试照样通过 —— 那只是自己证明自己。所以该文件**另写了一套** Wilson 区间与标准误的实现，
只从最原始的 `experiments/<exp_id>/episodes.jsonl` 逐行读 bool，重新算出：

| 复算量 | 与什么比对 |
|---|---|
| `num_episodes` / `num_successes` | `summary.json` 同名字段 |
| `success_rate` / `success_rate_se` | 同上（容差 1e-9） |
| `success_rate_ci95`（两端） | 同上 |
| `failure_modes` 分布 | 同上（字典相等） |

```cmd
:: 打印"重算 vs 记录"对照表（纯 ASCII 输出，Windows 终端友好）
.venv\Scripts\python.exe tests\test_recorded_experiments.py

:: 或作为测试跑
.venv\Scripts\python.exe -m pytest tests\test_recorded_experiments.py -v
```

**当前结论（实测）**：22 个实验的**全部数字一致**，退出码 0，末尾输出
`[verify] ALL OK - every recorded number was reproduced independently`。

### 第 2 层：单元测试（逻辑正确性与防回归）

```cmd
.venv\Scripts\python.exe -m pytest -q
```

| 文件 | 覆盖内容 | 为什么重要 |
|---|---|---|
| `test_metrics.py` | Wilson 边界（`n=0`→nan、`k=0`、`k=n`、`p=0.5` 手算对照）、SE、JSONL 往返、非法 `failure_mode` 拒绝 | **空输入必须返回 nan 而不是伪装成 0** —— 这是诚实性问题；把"没数据"显示成"成功率 0"会误导 |
| `test_runner_seeds.py` | `_episode_seed` 确定性、块内单射、跨 seed 块不重叠 | 报告里有"同一 checkpoint 换 seed 块得 0.393 vs 0.167"，必须排除 seed 设计 bug 才敢这么说 |
| `test_data_pipeline.py` | `build_state` 展平的 29 维与**字段顺序**（内存 h5py 逐列核对） | 若训练侧与评测侧的 state 顺序不一致，模型照样收敛但永远学不对，且现象与"数据不够"一模一样 |
| `test_failure_modes.py` | `slip` / `not_reached` / `collision` / `timeout` / `no_contact` / `success` 六分支 + 阈值边界 | "60% 失败是抓取滑落"这个结论直接依赖这些分支条件 |
| `test_imports.py` | 9 个模块的导入自检 | 把"`embodied_core` 必须可导入"固定下来，防止 `docs/phase3_progress.md` 第 11 条那个坑复发 |

### 第 3 层：端到端最小复现（真跑仿真）

```cmd
.venv\Scripts\python.exe -m pytest -m slow --runslow tests/test_end_to_end_slow.py -v
```

用 `zero` 策略真跑 1 个 episode（不需要 checkpoint、不需要 GPU），验证
`reset → step → 写 episodes.jsonl → 写 summary.json` 整条落盘路径没被改坏。
**默认跳过**（需要仿真器，约 10 秒），只在显式加 `--runslow` 时运行。

## 已知限制（明确声明）

1. **`experiments/` 不入库**（见 `.gitignore`）。克隆仓库后**第 1 层会自动跳过**
   （`pytest.skip`），因为它依赖真实的实验记录。要复现完整验证，需要自己重跑实验
   （命令见 `project-a-il/README.md` / `project-b-vla/README.md`）。
2. **第 1 层验证的是"记录 → 统计"这一步**，不重新验证仿真过程本身。
   也就是说：它保证"成功率算对了"，不保证"仿真跑对了"。
   仿真的正确性由上游的 ManiSkill 官方环境保证。
3. **失败模式分类是规则启发式**，不是人工标注。单测固定的是"规则按文档执行"，
   不是"规则一定正确"。阈值见 `embodied_core/metrics/failure_modes.py` 头部常量。
4. **没有测试真实策略的质量**（那是实验本身要做的事，不是单测的职责）。

## 依赖

```cmd
:: pytest 装在项目 venv 内（不污染系统环境）
.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "pytest>=8"
```

## 相关

- `pyproject.toml` — `[tool.pytest.ini_options]`：`pythonpath=["."]`、`testpaths`、`slow` marker 声明
- `tests/conftest.py` — `--runslow` 开关 + sys.path 双保险
- `reports/A.md` / `reports/B.md` — 被验证的那些结论的完整上下文
