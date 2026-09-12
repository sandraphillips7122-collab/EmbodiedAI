"""``embodied_core.metrics.recorder`` 的单元测试。

覆盖对外汇报数字的**计算逻辑本身**：

- Wilson 95% 置信区间的边界与已知值（``n=0`` / ``k=0`` / ``k=n`` / ``p=0.5``）
- 二项标准误
- **空输入必须返回 nan，不能伪装成 0**（这是诚实性问题，不是小事）
- 非法 ``failure_mode`` 必须抛错（防止标签被随意编造）
- JSONL 写入 → 读回 → 重算，闭环一致

与 ``tests/test_recorded_experiments.py`` 的分工：
那个文件是"**独立复算真实实验结果**"（交叉验证），
这个文件是"**单测计算函数本身**"（边界与回归）。
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from embodied_core.metrics.recorder import (
    FAILURE_MODES,
    EpisodeResult,
    ExperimentRecorder,
    summarize,
)

Z95 = 1.959963984540054


# --------------------------------------------------------------------------------------
# Wilson 区间 / 标准误
# --------------------------------------------------------------------------------------

def test_wilson_matches_closed_form_for_half() -> None:
    """p=0.5 时区间应对称，且可由公式手算复核。"""
    n, k = 100, 50
    phat = 0.5
    denom = 1.0 + Z95 * Z95 / n
    center = (phat + Z95 * Z95 / (2 * n)) / denom
    half = Z95 * math.sqrt(phat * (1 - phat) / n + Z95 * Z95 / (4 * n * n)) / denom
    sr = summarize([True] * k + [False] * (n - k))
    assert sr.ci95_low == pytest.approx(center - half, abs=1e-12)
    assert sr.ci95_high == pytest.approx(center + half, abs=1e-12)
    # 对称性与包含点估计
    assert abs((sr.ci95_low - (1 - sr.ci95_high))) < 1e-12
    assert sr.ci95_low < 0.5 < sr.ci95_high


@pytest.mark.parametrize("n", [1, 5, 30, 60, 150])
def test_wilson_contains_point_estimate_and_stays_in_unit_interval(n: int) -> None:
    """对任意 n、任意成功次数，区间必须包含点估计且落在 [0, 1]。

    为什么用 1e-12 容差而不是精确的 ``<=`` 链：
    ``k=0`` 或 ``k=n`` 时中心点与区间端点**理论上**重合，但浮点运算会留下 ulp 级残差
    —— 实测 ``n=150,k=0`` 的下界是 ``1.734723475976807e-18``（而非恰好 0），
    ``n=30,k=30`` 的上界是 ``0.9999999999999999``（而非恰好 1）。
    这些残差远小于任何有实际意义的差别，所以用容差比较；
    端点是否被正确裁到 [0, 1] 由下面那条专门的测试负责。
    """
    eps = 1e-12
    for k in range(n + 1):
        sr = summarize([True] * k + [False] * (n - k))
        assert 0.0 <= sr.ci95_low <= 1.0, (n, k, sr)
        assert 0.0 <= sr.ci95_high <= 1.0, (n, k, sr)
        assert sr.ci95_low <= sr.rate + eps, (n, k, sr)
        assert sr.rate <= sr.ci95_high + eps, (n, k, sr)


def test_wilson_degenerate_endpoints_only_differ_by_ulp() -> None:
    """``k=0`` / ``k=n`` 的退化情形：端点差异必须只在 ulp 级（把浮点行为固定下来）。

    这条测试把"已知的浮点残差"变成**明确契约**：哪天有人改了公式导致残差量级变大
    （例如从 1e-18 变成 1e-6），这里会立刻失败。
    """
    # 150 次全败：下界应 ≈ 0（只允许 ulp 级残差），上界应显著小于 1
    all_fail = summarize([False] * 150)
    assert 0.0 <= all_fail.ci95_low < 1e-12
    assert all_fail.ci95_high < 0.05

    # 30 次全胜：上界应 ≈ 1（ulp 级差异），下界应明显小于 1
    all_ok = summarize([True] * 30)
    assert 1.0 - 1e-12 < all_ok.ci95_high <= 1.0
    assert all_ok.ci95_low > 0.8


def test_wilson_zero_successes_upper_bound_is_below_normal_approximation() -> None:
    """0 成功时 Wilson 上限应显著小于 0（正态近似给不出有限区间），且为正。"""
    sr = summarize([False] * 30)
    assert sr.successes == 0 and sr.rate == 0.0
    assert 0.0 < sr.ci95_high < 0.15
    # 与手算复核：k=0, n=30, z=1.96 -> 上限 = (z²/2n + z·z/2n)/denom 的化简
    denom = 1.0 + Z95 * Z95 / 30
    center = (Z95 * Z95 / (2 * 30)) / denom
    half = (Z95 * math.sqrt(Z95 * Z95 / (4 * 30 * 30))) / denom
    assert sr.ci95_high == pytest.approx(center + half, abs=1e-12)


def test_wilson_all_successes_lower_bound_is_above_zero() -> None:
    """全部成功时下界应显著大于 0（小样本不能宣称 100%）。"""
    sr = summarize([True] * 5)
    assert sr.rate == 1.0
    assert 0.0 < sr.ci95_low < 1.0


def test_standard_error_matches_formula() -> None:
    sr = summarize([True] * 3 + [False] * 7)
    phat = 0.3
    assert sr.rate == pytest.approx(phat, abs=1e-15)
    assert sr.se == pytest.approx(math.sqrt(phat * (1 - phat) / 10), abs=1e-15)


def test_empty_input_returns_nan_not_zero() -> None:
    """空输入必须显式返回 nan：把"没数据"伪装成"成功率 0"是误导。"""
    sr = summarize([])
    assert sr.n == 0 and sr.successes == 0
    assert math.isnan(sr.rate)
    assert math.isnan(sr.se)
    assert math.isnan(sr.ci95_low) and math.isnan(sr.ci95_high)


# --------------------------------------------------------------------------------------
# EpisodeResult 校验
# --------------------------------------------------------------------------------------

def test_invalid_failure_mode_is_rejected() -> None:
    """拼错的失败模式必须直接报错，而不是静默统计进去。"""
    with pytest.raises(ValueError):
        EpisodeResult("e", "t", "p", 0, 0, True, 1, 0.1, failure_mode="made_up")


@pytest.mark.parametrize("mode", FAILURE_MODES)
def test_all_declared_failure_modes_are_accepted(mode: str) -> None:
    result = EpisodeResult("e", "t", "p", 0, 0, True, 1, 0.1, failure_mode=mode)
    assert result.failure_mode == mode


# --------------------------------------------------------------------------------------
# JSONL 往返 + summary 一致性
# --------------------------------------------------------------------------------------

def test_jsonl_roundtrip_and_summary_recompute() -> None:
    """写入 → 读回 → 重算，三者必须自洽。"""
    with tempfile.TemporaryDirectory() as tmp:
        rec = ExperimentRecorder(tmp, "roundtrip", meta={"seeds": [0, 1, 2]})
        n, k = 40, 30
        for i in range(n):
            ok = i < k
            rec.log_episode(
                EpisodeResult(
                    exp_id="roundtrip",
                    task="PickCube-v1",
                    policy="dummy",
                    seed=i % 3,
                    episode_idx=i,
                    success=ok,
                    steps=50 + i,
                    duration_s=1.0 + 0.01 * i,
                    failure_mode="success" if ok else "timeout",
                )
            )
        summary = rec.write_summary()

        # JSONL 条数一致、可回读
        episodes = rec.load_episodes()
        assert len(episodes) == n
        assert sum(1 for e in episodes if e.success) == k

        # summary 的字段可由 episodes 独立重算
        assert summary["num_episodes"] == n
        assert summary["num_successes"] == k
        assert summary["success_rate"] == pytest.approx(k / n, abs=1e-15)
        assert summary["success_rate_se"] == pytest.approx(
            math.sqrt((k / n) * (1 - k / n) / n), abs=1e-15
        )
        assert summary["failure_modes"] == {"success": k, "timeout": n - k}
        assert summary["steps_mean"] == pytest.approx(sum(range(50, 50 + n)) / n, abs=1e-12)

        # meta.json 落盘且内容正确
        meta = json.loads((Path(tmp) / "roundtrip" / "meta.json").read_text(encoding="utf-8"))
        assert meta == {"seeds": [0, 1, 2]}


def test_exp_id_mismatch_is_rejected() -> None:
    """写错实验目录必须报错，避免把 A 实验的结果记到 B 实验名下。"""
    with tempfile.TemporaryDirectory() as tmp:
        rec = ExperimentRecorder(tmp, "exp-A")
        with pytest.raises(ValueError):
            rec.log_episode(EpisodeResult("exp-B", "t", "p", 0, 0, True, 1, 0.1))


def test_loading_missing_jsonl_returns_empty() -> None:
    """目录里还没有 episodes.jsonl 时应返回空列表，而不是抛异常。"""
    with tempfile.TemporaryDirectory() as tmp:
        rec = ExperimentRecorder(tmp, "empty-exp")
        assert rec.load_episodes() == []
