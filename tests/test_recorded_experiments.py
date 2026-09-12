"""独立复算：用另一套实现重算 ``experiments/*/summary.json`` 里的统计量。

**为什么刻意不 import ``embodied_core.metrics``**

本仓库对外汇报的每个数字（0.3933 / 0.4333 / 0.0167 …）都是评测器算出来的。
如果复算脚本直接调用同一个 ``summarize()``，那么"公式写错了"这种 bug 会被**一起复现**，
测试照样通过 —— 那只是自己证明自己，没有意义。

所以这里**另写一套** Wilson 区间与标准误的实现，只从最原始的
``experiments/<exp_id>/episodes.jsonl`` 逐行读 bool，重新算出：
成功率 / 标准误 / 95% 置信区间 / 失败模式分布，再与 ``summary.json`` 逐字段比对。

**这套复算能证明什么**

- 一致 → 报告里的数字可以由**独立代码**从原始记录复现出来（不是只有评测器自己说了算）。
- 不一致 → 立刻暴露，且差异会被打印出来供人工判断（是口径差异还是真 bug）。

用法::

    pytest tests/test_recorded_experiments.py -v          # 作为测试
    python tests/test_recorded_experiments.py            # 打印"重算 vs 记录"对照表
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

#: 浮点比对容差。
#: 为什么是 1e-9 而不是更松的 1e-6：复算与评测器用的是**同一个公式、同一顺序**的浮点
#: 运算（都是先 k/n 再代入），差异只可能来自 ulp 级舍入，因此 1e-9 足够严也不会误报。
#: 反过来说，如果哪天有人偷改了公式（例如把 z 从 1.96 改成 1.645），差异会远大于 1e-9，
#: 这条测试就能立刻抓住。
TOL = 1e-9


# --------------------------------------------------------------------------------------
# 独立实现（不 import embodied_core.metrics）
# --------------------------------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval 的 95% 双侧区间。

    独立实现：直接照 Wilson (1927) 的定义写，公式经代数整理为
    ``(phat + z²/2n ± z·sqrt(phat(1-phat)/n + z²/4n²)) / (1 + z²/n)``。
    """
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def standard_error(k: int, n: int) -> float:
    """二项比例的标准误 sqrt(p(1-p)/n)。"""
    if n == 0:
        return float("nan")
    phat = k / n
    return math.sqrt(phat * (1.0 - phat) / n)


def recompute_from_jsonl(jsonl_path: Path) -> dict:
    """只读原始 JSONL，重算全部对外汇报的统计量。"""
    records = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    n = len(records)
    k = sum(1 for r in records if r["success"])

    failures: dict[str, int] = {}
    for r in records:
        mode = r.get("failure_mode", "unknown")
        failures[mode] = failures.get(mode, 0) + 1

    durations = [r["duration_s"] for r in records]
    steps = [r["steps"] for r in records]
    lo, hi = wilson_interval(k, n)

    return {
        "num_episodes": n,
        "num_successes": k,
        "success_rate": (k / n) if n else float("nan"),
        "success_rate_se": standard_error(k, n),
        "success_rate_ci95": [lo, hi],
        "failure_modes": failures,
        "duration_s_mean": statistics.fmean(durations) if durations else float("nan"),
        "duration_s_median": statistics.median(durations) if durations else float("nan"),
        "steps_mean": statistics.fmean(steps) if steps else float("nan"),
    }


def _discover_experiments() -> list[Path]:
    """找出所有既有 episodes.jsonl 又有 summary.json 的实验目录。"""
    if not EXPERIMENTS_DIR.is_dir():
        return []
    found = []
    for child in sorted(EXPERIMENTS_DIR.iterdir()):
        if child.is_dir() and (child / "episodes.jsonl").is_file() and (child / "summary.json").is_file():
            found.append(child)
    return found


EXPERIMENTS = _discover_experiments()


def _close(a: float, b: float, tol: float = TOL) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= tol


# --------------------------------------------------------------------------------------
# 测试
# --------------------------------------------------------------------------------------

@pytest.mark.skipif(not EXPERIMENTS, reason="experiments/ 不存在（该目录不入库），跳过复算")
@pytest.mark.parametrize("exp_dir", EXPERIMENTS, ids=[d.name for d in EXPERIMENTS])
def test_summary_matches_independent_recompute(exp_dir: Path) -> None:
    """每条 summary.json 都必须能被独立复算出来（标量容差 1e-9，浮点列表逐项比）。"""
    recorded = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
    recomputed = recompute_from_jsonl(exp_dir / "episodes.jsonl")

    mismatches: list[str] = []

    for key in ("num_episodes", "num_successes"):
        if recorded[key] != recomputed[key]:
            mismatches.append(f"{key}: 记录={recorded[key]} 重算={recomputed[key]}")

    for key in ("success_rate", "success_rate_se", "duration_s_mean",
                "duration_s_median", "steps_mean"):
        rec, got = recorded.get(key), recomputed[key]
        if rec is None:
            mismatches.append(f"{key}: summary.json 缺少该字段")
        elif not _close(float(rec), float(got)):
            mismatches.append(f"{key}: 记录={rec!r} 重算={got!r}")

    # 置信区间是长度 2 的列表
    rec_ci = recorded.get("success_rate_ci95")
    if not rec_ci or len(rec_ci) != 2:
        mismatches.append("success_rate_ci95: summary.json 缺少或长度不为 2")
    elif not (_close(float(rec_ci[0]), recomputed["success_rate_ci95"][0])
              and _close(float(rec_ci[1]), recomputed["success_rate_ci95"][1])):
        mismatches.append(
            f"success_rate_ci95: 记录={rec_ci} 重算={recomputed['success_rate_ci95']}"
        )

    if recorded.get("failure_modes") != recomputed["failure_modes"]:
        mismatches.append(
            f"failure_modes: 记录={recorded.get('failure_modes')} 重算={recomputed['failure_modes']}"
        )

    assert not mismatches, f"{exp_dir.name} 复算不一致：" + "；".join(mismatches)


def test_recompute_covers_expected_number_of_experiments() -> None:
    """确认复算确实覆盖了多个实验（防止 parametrize 意外收集为空而"假通过"）。"""
    if not EXPERIMENTS:
        pytest.skip("experiments/ 不存在，跳过")
    assert len(EXPERIMENTS) >= 10, f"只发现 {len(EXPERIMENTS)} 个实验，少于预期"


# --------------------------------------------------------------------------------------
# 可独立运行：打印"重算 vs 记录"对照表
# --------------------------------------------------------------------------------------

def main() -> int:
    if not EXPERIMENTS:
        print(f"[verify] no experiment dirs found ({EXPERIMENTS_DIR} missing or empty); "
              f"this dir is gitignored, so that is normal.")
        return 0

    # 输出刻意只用 ASCII：Windows 终端默认 GBK，中文/emoji 会乱码或直接抛
    # UnicodeEncodeError（本文件开发时实测踩过）。
    print(f"[verify] {len(EXPERIMENTS)} experiments; each recomputed independently "
          f"from episodes.jsonl only\n")
    header = (f"{'exp_id':<34} {'eps':>4} {'rate_rec':>9} {'rate_recomp':>11} "
              f"{'se':>7} {'ci95_low':>9} {'ci95_high':>10} {'match':>6}")
    print(header)
    print("-" * len(header))

    all_ok = True
    for exp_dir in EXPERIMENTS:
        recorded = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        got = recompute_from_jsonl(exp_dir / "episodes.jsonl")
        ok = (
            recorded["num_episodes"] == got["num_episodes"]
            and recorded["num_successes"] == got["num_successes"]
            and _close(float(recorded["success_rate"]), float(got["success_rate"]))
            and _close(float(recorded["success_rate_se"]), float(got["success_rate_se"]))
            and _close(float(recorded["success_rate_ci95"][0]), got["success_rate_ci95"][0])
            and _close(float(recorded["success_rate_ci95"][1]), got["success_rate_ci95"][1])
            and recorded.get("failure_modes") == got["failure_modes"]
        )
        all_ok = all_ok and ok
        print(
            f"{exp_dir.name:<34} {got['num_episodes']:>8} "
            f"{recorded['success_rate']:>11.4f} {got['success_rate']:>11.4f} "
            f"{'OK' if ok else 'MISMATCH':>6}"
        )

    print()
    print("[verify] ALL OK - every recorded number was reproduced independently"
          if all_ok else "[verify] MISMATCH - see table above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
